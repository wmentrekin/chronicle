from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from chronicle.cli.app import app
from chronicle.session import SessionManifest
from chronicle.stage1 import GcpStage1Config, build_gcp_stage1_plan
from chronicle.stage1.audio import AudioProbe
from chronicle.stage1.service import DEFAULT_STAGE1_PARAKEET_MODEL, execute_stage1


runner = CliRunner()


def make_manifest(tmp_path: Path) -> SessionManifest:
    return SessionManifest(
        session_id="test-session",
        title="Test Session",
        interview_date=None,
        audio_files=["audio/a.m4a", "audio/b.m4a"],
        participants=["Pat Example"],
        primary_interviewees=["Pat Example"],
        people_likely_discussed=[],
        context_doc="context.md",
        stage1_model_preference="parakeet",
        manifest_path=(tmp_path / "inputs" / "sessions" / "test-session" / "session.yaml").as_posix(),
    )


def test_transcribe_rejects_backend_option() -> None:
    result = runner.invoke(app, ["transcribe", "test-session", "--backend", "auto"])

    assert result.exit_code != 0
    assert "--backend" in result.output

    help_result = runner.invoke(app, ["transcribe", "--help"])
    assert help_result.exit_code == 0
    assert "--backend" not in help_result.output
    assert "--model" not in help_result.output
    assert "--device" not in help_result.output
    assert "--project-id" in help_result.output
    assert "--cpu-only" in help_result.output


def test_transcribe_requires_project_id(monkeypatch) -> None:
    manifest = make_manifest(Path("/tmp"))
    output_dirs = {
        "stage1": Path("/tmp/outputs/test-session/stage1"),
        "runs": Path("/tmp/outputs/test-session/runs"),
    }
    output_dirs["stage1"].mkdir(parents=True, exist_ok=True)
    output_dirs["runs"].mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("chronicle.cli.stage1.require_valid_session", lambda session_id, console_obj, participants_file: manifest)
    monkeypatch.setattr("chronicle.cli.stage1.ensure_output_dirs", lambda session_id: output_dirs)
    monkeypatch.setattr("chronicle.cli.stage1.confirm_overwrite_if_needed", lambda **kwargs: True)

    result = runner.invoke(app, ["transcribe", "test-session"])

    assert result.exit_code != 0
    assert "--project-id" in result.output


def test_transcribe_uses_cloud_stage1_by_default(tmp_path: Path, monkeypatch) -> None:
    session_dir = tmp_path / "inputs" / "sessions" / "test-session"
    session_dir.mkdir(parents=True, exist_ok=True)
    global_dir = tmp_path / "inputs" / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    (global_dir / "participants.yaml").write_text("participants:\n  - name: Pat Example\n", encoding="utf-8")
    manifest = make_manifest(tmp_path)
    output_dirs = {
        "stage1": tmp_path / "outputs" / "test-session" / "stage1",
        "runs": tmp_path / "outputs" / "test-session" / "runs",
    }
    output_dirs["stage1"].mkdir(parents=True, exist_ok=True)
    output_dirs["runs"].mkdir(parents=True, exist_ok=True)

    observed: dict[str, object] = {}

    monkeypatch.setattr("chronicle.cli.stage1.require_valid_session", lambda session_id, console_obj, participants_file: manifest)
    monkeypatch.setattr("chronicle.cli.stage1.ensure_output_dirs", lambda session_id: output_dirs)
    monkeypatch.setattr("chronicle.cli.stage1.confirm_overwrite_if_needed", lambda **kwargs: True)
    monkeypatch.setattr(
        "chronicle.cli.stage1.build_gcp_stage1_plan",
        lambda **kwargs: observed.setdefault("plan_kwargs", kwargs) or object(),
    )
    monkeypatch.setattr(
        "chronicle.cli.stage1.run_gcp_stage1_plan",
        lambda plan, **kwargs: observed.update({"plan": plan, **kwargs}),
    )

    result = runner.invoke(
        app,
        [
            "transcribe",
            "test-session",
                "--project-id",
                "demo-project",
                "--instance-name",
                "demo-instance",
                "--participants-file",
                str(global_dir / "participants.yaml"),
            ],
        )

    assert result.exit_code == 0
    assert observed["keep_instance"] is False
    plan_kwargs = observed["plan_kwargs"]
    assert plan_kwargs["config"].project_id == "demo-project"
    assert plan_kwargs["config"].instance_name == "demo-instance"
    assert plan_kwargs["config"].gpu_enabled is False


def test_execute_stage1_uses_parakeet_only(tmp_path: Path, monkeypatch) -> None:
    manifest = make_manifest(tmp_path)
    stage_dir = tmp_path / "outputs" / "test-session" / "stage1"
    stage_dir.mkdir(parents=True, exist_ok=True)

    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "chronicle.stage1.service.probe_audio_file",
        lambda audio_path: AudioProbe(
            source_audio=audio_path.as_posix(),
            duration_seconds=1.0,
            file_size_bytes=16,
        ),
    )
    monkeypatch.setattr(
        "chronicle.stage1.service.decode_audio_to_mono_16k",
        lambda audio_path: np.zeros(16000, dtype=np.float32),
    )
    monkeypatch.setattr(
        "chronicle.stage1.service.ensure_local_parakeet_model",
        lambda **kwargs: tmp_path / "models" / "parakeet",
    )
    monkeypatch.setattr(
        "chronicle.stage1.service.build_parakeet_pipeline",
        lambda **kwargs: ("processor", object()),
    )

    def fake_transcribe_with_parakeet_pipeline(**kwargs):
        observed.update(kwargs)
        progress_callback = kwargs["progress_callback"]
        progress_callback(1, 1, 1.0)
        return (
            {
                "family": "parakeet",
                "name": kwargs["model_name"],
                "runtime": "parakeet",
                "device": kwargs["device"],
                "notes": ["fake parakeet path"],
            },
            [
                {
                    "segment_id": 1,
                    "start": "00:00:00.000",
                    "end": "00:00:01.000",
                    "text": "Hello from Parakeet.",
                    "decode_status": "ok",
                }
            ],
            [],
        )

    monkeypatch.setattr(
        "chronicle.stage1.service.transcribe_with_parakeet_pipeline",
        fake_transcribe_with_parakeet_pipeline,
    )

    generated_paths, skipped_paths, notes = execute_stage1(
        manifest=manifest,
        stage_dir=stage_dir,
        participants_file=tmp_path / "inputs" / "global" / "participants.yaml",
        force=True,
        model_name=None,
        device="cpu",
        parakeet_chunk_length_s=15,
        parakeet_batch_size=1,
        parakeet_overlap_stride_s=None,
        parakeet_model_dir=tmp_path / "models" / "parakeet",
        local_files_only=True,
    )

    assert skipped_paths == []
    assert generated_paths
    assert observed["model_name"] == DEFAULT_STAGE1_PARAKEET_MODEL
    assert observed["batch_size"] == 1
    assert observed["overlap_stride_s"] is None

    raw_json = stage_dir / "raw_transcript.json"
    raw_md = stage_dir / "raw_transcript.md"
    assert raw_json.exists()
    assert raw_md.exists()

    artifact = json.loads(raw_json.read_text(encoding="utf-8"))
    assert artifact["model"]["family"] == "parakeet"
    assert artifact["transcript_text"].splitlines() == [
        "Hello from Parakeet.",
        "Hello from Parakeet.",
    ]
    assert any("Parakeet model files" in note for note in notes)


def test_build_gcp_stage1_plan_cpu_mode(tmp_path: Path) -> None:
    session_dir = tmp_path / "inputs" / "sessions" / "example-session"
    session_dir.mkdir(parents=True)
    config = GcpStage1Config(
        project_id="demo-project",
        instance_name="stage1-cpu-test",
        session_id="example-session",
        zone="us-central1-a",
        machine_type="e2-standard-8",
        gpu_enabled=False,
    )

    plan = build_gcp_stage1_plan(config=config, local_session_dir=session_dir, worker_user="tester")

    assert "--accelerator" not in plan.commands["vm_create"]
    assert plan.commands["vm_create"][0:5] == [
        "gcloud",
        "compute",
        "instances",
        "create",
        "stage1-cpu-test",
    ]
    assert "/home/tester/chronicle" in " ".join(plan.commands["bootstrap"])
    assert "inputs/global/participants.yaml" in " ".join(plan.commands["upload_participants"])
    assert plan.shell_command("vm_create").startswith("gcloud compute instances create stage1-cpu-test")


def test_transcribe_plan_cli_renders_json(tmp_path: Path) -> None:
    session_dir = tmp_path / "example-session"
    session_dir.mkdir()
    result = runner.invoke(
        app,
        [
            "transcribe-plan",
            "example-session",
            "--project-id",
            "demo-project",
            "--instance-name",
            "demo-instance",
            "--local-session-dir",
            str(session_dir),
            "--worker-user",
            "tester",
            "--zone",
            "us-central1-a",
            "--machine-type",
            "e2-standard-8",
            "--cpu-only",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["config"]["project_id"] == "demo-project"
    assert payload["config"]["gpu_enabled"] is False
    assert payload["worker_user"] == "tester"
    assert payload["commands"]["run_stage1"][0:3] == ["gcloud", "compute", "ssh"]


def test_transcribe_command_cli_renders_shell(tmp_path: Path) -> None:
    session_dir = tmp_path / "example-session"
    session_dir.mkdir()
    result = runner.invoke(
        app,
        [
            "transcribe-command",
            "vm_create",
            "example-session",
            "--project-id",
            "demo-project",
            "--instance-name",
            "demo-instance",
            "--local-session-dir",
            str(session_dir),
            "--worker-user",
            "tester",
            "--zone",
            "us-central1-a",
            "--machine-type",
            "e2-standard-8",
            "--cpu-only",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip().startswith("gcloud compute instances create demo-instance")
