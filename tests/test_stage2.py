"""Unit tests for Stage 2 acoustic speaker diarization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chronicle.cli.app import app
from chronicle.session import SessionManifest
from chronicle.stage2.artifacts import (
    render_stage2_markdown,
    stage2_output_paths,
    stage2_partial_output_paths,
    write_stage2_artifacts,
)
from chronicle.stage2.service import build_stage2_artifact, execute_stage2


runner = CliRunner()


def make_manifest(tmp_path: Path) -> SessionManifest:
    return SessionManifest(
        session_id="test-session",
        title="Test Session",
        interview_date=None,
        audio_files=["audio/a.m4a", "audio/b.m4a"],
        participants=["Pat Example", "Bill Example"],
        primary_interviewees=["Pat Example", "Bill Example"],
        people_likely_discussed=[],
        context_doc="context.md",
        stage1_model_preference="parakeet",
        manifest_path=(tmp_path / "inputs" / "sessions" / "test-session" / "session.yaml").as_posix(),
    )


def test_stage2_output_paths(tmp_path: Path) -> None:
    stage2_dir = tmp_path / "outputs" / "test-session" / "stage2"
    json_path, md_path = stage2_output_paths(stage2_dir)
    assert json_path == stage2_dir / "diarization.json"
    assert md_path == stage2_dir / "diarization.md"

    partial_json, partial_md = stage2_partial_output_paths(stage2_dir)
    assert partial_json == stage2_dir / "diarization.partial.json"
    assert partial_md == stage2_dir / "diarization.partial.md"


def test_build_stage2_artifact(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    turns = [
        {
            "turn_id": 1,
            "speaker_label": "SPEAKER_00",
            "source_audio": "inputs/sessions/test-session/audio/a.m4a",
            "source_start_seconds": 0.0,
            "source_end_seconds": 5.0,
            "session_start_seconds": 0.0,
            "session_end_seconds": 5.0,
        },
        {
            "turn_id": 2,
            "speaker_label": "SPEAKER_01",
            "source_audio": "inputs/sessions/test-session/audio/a.m4a",
            "source_start_seconds": 5.0,
            "source_end_seconds": 12.5,
            "session_start_seconds": 5.0,
            "session_end_seconds": 12.5,
        },
    ]

    artifact = build_stage2_artifact(
        manifest=manifest,
        audio_artifacts=[
            {
                "audio_index": 1,
                "source_audio": "inputs/sessions/test-session/audio/a.m4a",
                "duration_seconds": 12.5,
                "wall_seconds": 2.0,
                "load_seconds": 0.5,
                "run_seconds": 1.5,
                "speaker_labels": ["SPEAKER_00", "SPEAKER_01"],
                "turn_count": 2,
            }
        ],
        combined_turns=turns,
        notes=["Test stage 2 run."],
        num_speakers=2,
        min_speakers=None,
        max_speakers=None,
        device="cpu",
        stage2_python=Path(".venv/bin/python"),
        vad_model_name="speechbrain/vad-crdnn-libriparty",
        embedding_model_name="speechbrain/spkrec-ecapa-voxceleb",
        total_wall_seconds=2.0,
        total_load_seconds=0.5,
        total_run_seconds=1.5,
        status="complete",
    )

    assert artifact["stage"] == "stage2_audio_diarization"
    assert artifact["session_id"] == "test-session"
    assert artifact["status"] == "complete"
    assert artifact["speaker_labels"] == ["SPEAKER_00", "SPEAKER_01"]
    assert len(artifact["turns"]) == 2

    md_output = render_stage2_markdown(artifact)
    assert "# Anonymous Audio Diarization" in md_output
    assert "SPEAKER_00" in md_output
    assert "SPEAKER_01" in md_output


def test_write_stage2_artifacts(tmp_path: Path) -> None:
    stage_dir = tmp_path / "outputs" / "test-session" / "stage2"
    manifest = make_manifest(tmp_path)
    artifact = build_stage2_artifact(
        manifest=manifest,
        audio_artifacts=[],
        combined_turns=[],
        notes=[],
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
        device="cpu",
        stage2_python=Path(".venv/bin/python"),
        vad_model_name="speechbrain/vad-crdnn-libriparty",
        embedding_model_name="speechbrain/spkrec-ecapa-voxceleb",
        total_wall_seconds=0.0,
        total_load_seconds=0.0,
        total_run_seconds=0.0,
        status="complete",
    )

    write_stage2_artifacts(stage_dir=stage_dir, artifact=artifact, partial=False)

    json_file = stage_dir / "diarization.json"
    md_file = stage_dir / "diarization.md"
    assert json_file.exists()
    assert md_file.exists()
    content = json.loads(json_file.read_text(encoding="utf-8"))
    assert content["session_id"] == "test-session"


def test_diarize_cli_help() -> None:
    result = runner.invoke(app, ["diarize", "--help"])
    assert result.exit_code == 0
    assert "session_id" in result.output.lower() or "session" in result.output.lower()
    assert "--num-speakers" in result.output
    assert "--device" in result.output
