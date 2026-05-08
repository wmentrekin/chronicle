from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from chronicle.cli.app import app
from chronicle.session import SessionManifest
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
        stage1_model_preference="faster-whisper",
        manifest_path=(tmp_path / "inputs" / "sessions" / "test-session" / "session.yaml").as_posix(),
    )


def test_transcribe_rejects_backend_option() -> None:
    result = runner.invoke(app, ["transcribe", "test-session", "--backend", "auto"])

    assert result.exit_code != 0
    assert "--backend" in result.output

    help_result = runner.invoke(app, ["transcribe", "--help"])
    assert help_result.exit_code == 0
    assert "--backend" not in help_result.output
    assert "--compute-type" not in help_result.output
    assert "--beam-size" not in help_result.output
    assert "Parakeet model name." in help_result.output


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
