from __future__ import annotations

from pathlib import Path

import pytest

from chronicle.exceptions import StageExecutionError
from chronicle.session import SessionManifest
from chronicle.stage3.alignment import align_transcript_to_diarization
from chronicle.stage3.artifacts import render_stage3_markdown
from chronicle.stage3.identity import normalize_llm_entries
from chronicle.stage3.inputs import load_stage2_artifact
from chronicle.stage3.llm import (
    require_ollama_config,
    resolve_stage3_model,
    run_ollama_speaker_mapping,
)
from chronicle.stage3.manual import validate_manual_speaker_map


def manifest() -> SessionManifest:
    return SessionManifest(
        session_id="test-session",
        title="Test Session",
        interview_date=None,
        audio_files=["audio/a.m4a", "audio/b.m4a"],
        participants=["Pat Example", "Bill Example"],
        primary_interviewees=["Pat Example", "Bill Example"],
        people_likely_discussed=[],
        context_doc="context.md",
        manifest_path="inputs/sessions/test-session/session.yaml",
    )


def stage1_segment(segment_id: int, source_audio: str, start: str, end: str, text: str) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "source_segment_id": segment_id,
        "source_audio": source_audio,
        "source_start": start,
        "source_end": end,
        "start": start,
        "end": end,
        "text": text,
    }


def stage2_turn(turn_id: int, source_audio: str, start: float, end: float, label: str) -> dict[str, object]:
    return {
        "turn_id": turn_id,
        "source_audio": source_audio,
        "source_start_seconds": start,
        "source_end_seconds": end,
        "session_start_seconds": start,
        "session_end_seconds": end,
        "speaker_label": label,
    }


def align(segments: list[dict[str, object]], turns: list[dict[str, object]]):
    return align_transcript_to_diarization(
        manifest=manifest(),
        stage1_artifact={"segments": segments},
        stage2_artifact={"turns": turns},
    )


def test_one_to_one_alignment() -> None:
    blocks, summary = align(
        [stage1_segment(1, "audio/a.m4a", "00:00:00.000", "00:00:05.000", "Hello there.")],
        [stage2_turn(1, "audio/a.m4a", 0.0, 5.0, "SPEAKER_00")],
    )
    assert blocks[0]["speaker_label"] == "SPEAKER_00"
    assert blocks[0]["alignment"]["alignment_confidence"] == "Confirmed"
    assert summary["aligned_block_count"] == 1


def test_multi_file_source_matching_does_not_cross_align() -> None:
    blocks, _ = align(
        [stage1_segment(1, "audio/a.m4a", "00:00:00.000", "00:00:05.000", "File A words.")],
        [stage2_turn(1, "audio/b.m4a", 0.0, 5.0, "SPEAKER_01")],
    )
    assert blocks[0]["speaker_label"] is None
    assert blocks[0]["alignment"]["alignment_confidence"] == "Needs review"


def test_boundary_crossing_segment_becomes_ambiguous() -> None:
    blocks, _ = align(
        [stage1_segment(1, "audio/a.m4a", "00:00:00.000", "00:00:10.000", "Two speakers in one segment.")],
        [
            stage2_turn(1, "audio/a.m4a", 0.0, 5.0, "SPEAKER_00"),
            stage2_turn(2, "audio/a.m4a", 5.0, 10.0, "SPEAKER_01"),
        ],
    )
    assert blocks[0]["speaker_label"] is None
    assert blocks[0]["speaker_label_candidates"] == ["SPEAKER_00", "SPEAKER_01"]
    assert blocks[0]["confidence"] == "Needs review"


def test_weak_overlap_segment_needs_review() -> None:
    blocks, _ = align(
        [stage1_segment(1, "audio/a.m4a", "00:00:00.000", "00:00:10.000", "Mostly unmatched.")],
        [stage2_turn(1, "audio/a.m4a", 9.0, 10.0, "SPEAKER_00")],
    )
    assert blocks[0]["speaker_label"] is None
    assert blocks[0]["alignment"]["alignment_confidence"] == "Needs review"


def test_missing_stage2_artifact_fails(tmp_path: Path) -> None:
    with pytest.raises(StageExecutionError, match="Run `chronicle diarize test-session` first"):
        load_stage2_artifact(manifest(), tmp_path)


def test_manual_override_validation() -> None:
    entries = validate_manual_speaker_map(
        manual_map={"SPEAKER_00": "Pat Example", "SPEAKER_01": "Bill Example"},
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
        participants=["Pat Example", "Bill Example"],
        mode="manual",
    )
    assert [entry["assigned_person"] for entry in entries] == ["Pat Example", "Bill Example"]


def test_invalid_llm_assignment_outside_manifest_fails() -> None:
    with pytest.raises(StageExecutionError, match="not a manifest participant"):
        normalize_llm_entries(
            raw_entries=[
                {
                    "speaker_label": "SPEAKER_00",
                    "assigned_person": "Unknown Person",
                    "confidence": "Likely",
                    "candidate_people": ["Pat Example"],
                }
            ],
            speaker_labels=["SPEAKER_00"],
            participants=["Pat Example"],
            manual_entries=[],
        )


def test_stage3_llm_default_model_is_local_ollama_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHRONICLE_STAGE3_MODEL", raising=False)
    assert resolve_stage3_model(None) == "qwen3:8b"


def test_missing_ollama_model_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chronicle.stage3.llm.list_ollama_models", lambda: {"llama3.1:8b"})
    with pytest.raises(StageExecutionError, match="requires a local Ollama model"):
        require_ollama_config("qwen3:8b")


def test_ollama_speaker_mapping_uses_fake_local_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chronicle.stage3.llm.list_ollama_models", lambda: {"qwen3:8b"})

    requests: list[tuple[str, dict[str, object]]] = []

    def fake_post(path: str, payload: dict[str, object], *, timeout: int = 120) -> dict[str, object]:
        requests.append((path, payload))
        return {
            "message": {
                "content": (
                    '{"speaker_map": ['
                    '{"speaker_label": "SPEAKER_00", "assigned_person": "Pat Example", '
                    '"confidence": "Likely", "candidate_people": ["Pat Example"]}'
                    "]}"
                )
            },
            "prompt_eval_count": 42,
            "eval_count": 18,
        }

    monkeypatch.setattr("chronicle.stage3.llm._ollama_post_json", fake_post)
    speaker_map, usage = run_ollama_speaker_mapping(
        manifest=manifest(),
        context_text="Family interview context.",
        participants_by_name={"Pat Example": {}, "Bill Example": {}},
        evidence_summary={
            "SPEAKER_00": {
                "total_blocks": 1,
                "examples": [{"text": "Hello there."}],
            }
        },
        manual_entries=[],
        model="qwen3:8b",
    )

    assert speaker_map[0]["assigned_person"] == "Pat Example"
    assert usage["provider"] == "ollama"
    assert usage["model"] == "qwen3:8b"
    assert usage["input_tokens"] == 42
    assert usage["output_tokens"] == 18
    assert requests[0][0] == "/api/chat"
    assert requests[0][1]["format"] == "json"


def test_markdown_rendering_from_json() -> None:
    markdown = render_stage3_markdown(
        {
            "session_id": "test-session",
            "mode": "manual",
            "source_stage1_artifact": "outputs/test-session/stage1/raw_transcript.json",
            "source_stage2_artifact": "outputs/test-session/stage2/diarization.json",
            "speaker_map": [
                {
                    "speaker_label": "SPEAKER_00",
                    "assigned_person": "Pat Example",
                    "confidence": "Confirmed",
                }
            ],
            "blocks": [
                {
                    "source_audio": "audio/a.m4a",
                    "speaker": "Pat Example",
                    "speaker_label": "SPEAKER_00",
                    "confidence": "Confirmed",
                    "start_time": "00:00:00.000",
                    "end_time": "00:00:05.000",
                    "text": "Verbatim words.",
                }
            ],
            "notes": [],
        }
    )
    assert "# Identified Conversation" in markdown
    assert "Verbatim words." in markdown
