from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from chronicle.cli.app import app
from chronicle.exceptions import StageExecutionError
from chronicle.paths import repo_relative
from chronicle.session import SessionManifest
from chronicle.stage3.alignment import align_transcript_to_diarization
from chronicle.stage3.artifacts import render_stage3_markdown
from chronicle.stage3.benchmark import (
    choose_stage3_benchmark_recommendation,
    parse_stage3_benchmark_backends,
    render_stage3_benchmark_markdown,
    run_stage3_benchmark,
    score_stage3_assignments,
)
from chronicle.stage3.embeddings import (
    build_embedding_cache_key,
    prepare_cached_embedding,
)
from chronicle.stage3.enrollment import (
    AudioSliceSpec,
    resolve_participant_reference_clips,
    resolve_speaker_audio_slices,
)
from chronicle.stage3.identity import normalize_llm_entries
from chronicle.stage3.inputs import Stage3Inputs, load_participant_records, load_stage2_artifact
from chronicle.stage3.llm import (
    require_ollama_config,
    run_ollama_decomposed_backend,
    resolve_stage3_model,
    run_ollama_speaker_mapping,
)
from chronicle.stage3.manual import validate_manual_speaker_map
from chronicle.stage3.refmatch import (
    assign_similarity_matches,
    run_speechbrain_hybrid,
    run_speechbrain_refmatch,
)
from chronicle.stage3.service import execute_stage3


runner = CliRunner()


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


def test_load_participant_records_parses_voice_references(tmp_path: Path) -> None:
    participants_file = tmp_path / "participants.yaml"
    participants_file.write_text(
        """
participants:
  - canonical_name: Pat Example
  - canonical_name: Bill Example
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    records = load_participant_records(participants_file)

    assert records["Pat Example"]["voice_references"] == []
    assert records["Bill Example"]["voice_references"] == []


def test_load_participant_records_normalizes_optional_voice_references(tmp_path: Path) -> None:
    participants_file = tmp_path / "participants.yaml"
    participants_file.write_text(
        """
participants:
  - canonical_name: Pat Example
    voice_references:
      - inputs/global/voice/pat-reference.wav
      - ./clips/pat-alt.wav
  - canonical_name: Bill Example
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    records = load_participant_records(participants_file)

    assert records["Pat Example"]["voice_references"][0] == "inputs/global/voice/pat-reference.wav"
    assert records["Pat Example"]["voice_references"][1].endswith("/clips/pat-alt.wav")
    assert records["Bill Example"]["voice_references"] == []


def test_load_participant_records_rejects_invalid_voice_references(tmp_path: Path) -> None:
    participants_file = tmp_path / "participants.yaml"
    participants_file.write_text(
        """
participants:
  - canonical_name: Pat Example
    voice_references: inputs/global/voice/pat-reference.wav
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StageExecutionError, match="`voice_references` must be a list of paths"):
        load_participant_records(participants_file)


def test_resolve_participant_reference_clips_rejects_missing_audio(tmp_path: Path) -> None:
    participants_file = tmp_path / "participants.yaml"
    participants_file.write_text("participants: []\n", encoding="utf-8")

    with pytest.raises(StageExecutionError, match="missing audio file"):
        resolve_participant_reference_clips(
            participants_by_name={
                "Pat Example": {
                    "canonical_name": "Pat Example",
                    "voice_references": ["inputs/global/voice/missing.wav"],
                }
            },
            participants_file=participants_file,
        )


def test_resolve_participant_reference_clips_rejects_out_of_repo_audio(tmp_path: Path) -> None:
    participants_file = tmp_path / "participants.yaml"
    participants_file.write_text("participants: []\n", encoding="utf-8")
    outside_audio = tmp_path / "outside.wav"
    outside_audio.write_bytes(b"not-real-audio")

    with pytest.raises(StageExecutionError, match="outside the repository"):
        resolve_participant_reference_clips(
            participants_by_name={
                "Pat Example": {
                    "canonical_name": "Pat Example",
                    "voice_references": [outside_audio.as_posix()],
                }
            },
            participants_file=participants_file,
        )


def test_resolve_speaker_audio_slices_sorts_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chronicle.stage3.enrollment.REPO_ROOT", tmp_path)
    session_dir = tmp_path / "inputs" / "sessions" / "test-session"
    audio_dir = session_dir / "audio"
    audio_dir.mkdir(parents=True)
    first_audio = audio_dir / "b.wav"
    second_audio = audio_dir / "a.wav"
    first_audio.write_bytes(b"a")
    second_audio.write_bytes(b"b")

    resolved = resolve_speaker_audio_slices(
        manifest=SessionManifest(
            session_id="test-session",
            title="Test Session",
            interview_date=None,
            audio_files=["audio/a.wav", "audio/b.wav"],
            participants=["Pat Example"],
            primary_interviewees=["Pat Example"],
            people_likely_discussed=[],
            context_doc="context.md",
            manifest_path=(session_dir / "session.yaml").as_posix(),
        ),
        stage2_artifact={
            "turns": [
                {
                    "turn_id": 2,
                    "speaker_label": "SPEAKER_01",
                    "source_audio": "audio/b.wav",
                    "source_start_seconds": 5.0,
                    "source_end_seconds": 7.0,
                },
                {
                    "turn_id": 1,
                    "speaker_label": "SPEAKER_01",
                    "source_audio": "audio/a.wav",
                    "source_start_seconds": 1.0,
                    "source_end_seconds": 2.0,
                },
            ]
        },
    )

    assert [item.source_audio for item in resolved["SPEAKER_01"]] == [
        repo_relative(second_audio),
        repo_relative(first_audio),
    ]
    assert [item.start_seconds for item in resolved["SPEAKER_01"]] == [1.0, 5.0]


def test_build_embedding_cache_key_is_order_independent() -> None:
    first = AudioSliceSpec(
        owner="Pat Example",
        kind="participant_reference",
        source_audio="inputs/global/voice/pat-1.wav",
        audio_path=Path("/tmp/pat-1.wav"),
    )
    second = AudioSliceSpec(
        owner="Pat Example",
        kind="participant_reference",
        source_audio="inputs/global/voice/pat-2.wav",
        audio_path=Path("/tmp/pat-2.wav"),
    )

    assert build_embedding_cache_key(model_name="model-a", slices=[first, second]) == build_embedding_cache_key(
        model_name="model-a",
        slices=[second, first],
    )


def test_prepare_cached_embedding_writes_only_under_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chronicle.stage3.embeddings.OUTPUTS_ROOT", tmp_path / "outputs")
    cache_root = tmp_path / "outputs" / "test-session" / "runs" / "stage3-embeddings"
    audio_file = tmp_path / "inputs" / "sessions" / "test-session" / "audio" / "speaker.wav"
    audio_file.parent.mkdir(parents=True)
    audio_file.write_bytes(b"placeholder")

    slice_spec = AudioSliceSpec(
        owner="SPEAKER_00",
        kind="speaker_slice",
        source_audio=audio_file.relative_to(tmp_path).as_posix(),
        audio_path=audio_file,
        start_seconds=0.0,
        duration_seconds=1.0,
        turn_id=1,
    )

    decoded_calls: list[tuple[Path, float | None, float | None]] = []

    def fake_decode_audio(path: Path, start_seconds: float | None, duration_seconds: float | None) -> np.ndarray:
        decoded_calls.append((path, start_seconds, duration_seconds))
        return np.array([0.1, -0.2, 0.3], dtype=np.float32)

    extractor_calls: list[tuple[int, int]] = []

    def fake_extractor(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        extractor_calls.append((audio.size, sample_rate))
        return np.array([3.0, 4.0], dtype=np.float32)

    result = prepare_cached_embedding(
        slices=[slice_spec],
        cache_root=cache_root,
        model_name="speechbrain/test-model",
        extractor=fake_extractor,
        audio_decoder=fake_decode_audio,
    )

    assert result.cached is False
    assert decoded_calls == [(audio_file, 0.0, 1.0)]
    assert extractor_calls == [(3, 16000)]
    assert sorted(path.relative_to(tmp_path).as_posix() for path in cache_root.iterdir()) == sorted(
        [
            result.paths.prepared_audio_path.relative_to(tmp_path).as_posix(),
            result.paths.embedding_path.relative_to(tmp_path).as_posix(),
            result.paths.metadata_path.relative_to(tmp_path).as_posix(),
        ]
    )
    assert np.allclose(result.embedding, np.array([0.6, 0.8], dtype=np.float32))

    cached = prepare_cached_embedding(
        slices=[slice_spec],
        cache_root=cache_root,
        model_name="speechbrain/test-model",
        extractor=lambda *_: pytest.fail("cached embedding should not re-run extractor"),
        audio_decoder=lambda *_: pytest.fail("cached embedding should not re-decode audio"),
    )

    assert cached.cached is True
    assert np.allclose(cached.embedding, result.embedding)


def test_assign_similarity_matches_clean_match() -> None:
    entries = assign_similarity_matches(
        similarity_matrix={
            "SPEAKER_00": {"Pat Example": 0.95, "Bill Example": 0.65},
            "SPEAKER_01": {"Pat Example": 0.62, "Bill Example": 0.93},
        }
    )

    assert [entry["assigned_person"] for entry in entries] == ["Pat Example", "Bill Example"]
    assert all(entry["confidence"] == "Confirmed" for entry in entries)


def test_assign_similarity_matches_ambiguous_match_fails() -> None:
    with pytest.raises(StageExecutionError, match="ambiguous match"):
        assign_similarity_matches(
            similarity_matrix={
                "SPEAKER_00": {"Pat Example": 0.81, "Bill Example": 0.79},
            },
            min_margin=0.05,
        )


def test_assign_similarity_matches_low_confidence_fails() -> None:
    with pytest.raises(StageExecutionError, match="low-confidence match"):
        assign_similarity_matches(
            similarity_matrix={
                "SPEAKER_00": {"Pat Example": 0.59, "Bill Example": 0.20},
            },
            min_similarity=0.60,
        )


def test_assign_similarity_matches_duplicate_best_match_conflict_fails() -> None:
    with pytest.raises(StageExecutionError, match="duplicate-best-match conflict"):
        assign_similarity_matches(
            similarity_matrix={
                "SPEAKER_00": {"Pat Example": 0.91, "Bill Example": 0.30},
                "SPEAKER_01": {"Pat Example": 0.90, "Bill Example": 0.20},
            }
        )


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


def test_ollama_decomposed_backend_reports_no_reference_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chronicle.stage3.llm.run_ollama_speaker_mapping",
        lambda **_: (
            [
                {
                    "speaker_label": "SPEAKER_00",
                    "assigned_person": "Pat Example",
                    "confidence": "Likely",
                    "candidate_people": ["Pat Example", "Bill Example"],
                }
            ],
            {
                "backend": "ollama_decomposed",
                "provider": "ollama",
                "model": "qwen3:8b",
                "prompt_version": "stage3-speaker-map-v1",
                "schema_version": "1.0",
                "input_tokens": 42,
                "output_tokens": 18,
                "estimated_cost_usd": None,
                "truncation_or_sampling_notes": [],
            },
        ),
    )

    _, backend_usage, llm_usage = run_ollama_decomposed_backend(
        manifest=manifest(),
        context_text="Family interview context.",
        participants_by_name={"Pat Example": {}, "Bill Example": {}},
        evidence_summary={
            "SPEAKER_00": {"total_blocks": 1},
            "SPEAKER_01": {"total_blocks": 2},
        },
        manual_entries=[
            {
                "speaker_label": "SPEAKER_01",
                "assigned_person": "Bill Example",
                "confidence": "Confirmed",
                "candidate_people": ["Bill Example"],
                "source": "manual",
            }
        ],
        model="qwen3:8b",
    )

    assert llm_usage["provider"] == "ollama"
    assert backend_usage["backend"] == "ollama_decomposed"
    assert backend_usage["workflow"] == "no-reference-baseline"
    assert backend_usage["reference_mode"] == "none"
    assert backend_usage["manual_assignment_count"] == 1
    assert backend_usage["unresolved_speaker_count"] == 1
    assert backend_usage["unresolved_speakers"] == ["SPEAKER_00"]
    assert backend_usage["llm_assignment_count"] == 1
    assert backend_usage["token_usage"] == {"input_tokens": 42, "output_tokens": 18}


def test_markdown_rendering_from_json() -> None:
    markdown = render_stage3_markdown(
        {
            "session_id": "test-session",
            "mode": "manual",
            "backend": "speechbrain_refmatch",
            "source_stage1_artifact": "outputs/test-session/stage1/raw_transcript.json",
            "source_stage2_artifact": "outputs/test-session/stage2/diarization.json",
            "backend_usage": {
                "provider": "speechbrain",
                "model": "speechbrain/spkrec-ecapa-voxceleb",
            },
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
    assert "- **Backend:** speechbrain_refmatch" in markdown
    assert "- **Backend Provider:** speechbrain" in markdown
    assert "Verbatim words." in markdown


def _stub_stage3_inputs() -> Stage3Inputs:
    return Stage3Inputs(
        stage1_path=Path("outputs/test-session/stage1/raw_transcript.json"),
        stage1_artifact={"segments": [{"text": "Hello there."}]},
        stage2_path=Path("outputs/test-session/stage2/diarization.json"),
        stage2_artifact={"speaker_labels": ["SPEAKER_00"], "turns": [{"speaker_label": "SPEAKER_00"}]},
        participants_file=Path("inputs/global/participants.yaml"),
        participants_by_name={
            "Pat Example": {"canonical_name": "Pat Example", "voice_references": []},
            "Bill Example": {"canonical_name": "Bill Example", "voice_references": []},
        },
        context_path=Path("inputs/sessions/test-session/context.md"),
        context_text="Family interview context.",
    )


def _patch_common_stage3_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chronicle.stage3.service.load_stage3_inputs", lambda **_: _stub_stage3_inputs())
    monkeypatch.setattr("chronicle.stage3.service.validate_speaker_count", lambda **_: None)
    monkeypatch.setattr(
        "chronicle.stage3.service.align_transcript_to_diarization",
        lambda **_: (
            [
                {
                    "source_audio": "audio/a.m4a",
                    "speaker_label": "SPEAKER_00",
                    "confidence": "Confirmed",
                    "text": "Hello there.",
                }
            ],
            {"aligned_block_count": 1},
        ),
    )
    monkeypatch.setattr(
        "chronicle.stage3.service.build_evidence_summary",
        lambda **_: ({"SPEAKER_00": {"total_blocks": 1, "examples": [{"text": "Hello there."}]}}, []),
    )
    monkeypatch.setattr("chronicle.stage3.service.load_manual_speaker_map", lambda _: {})


def test_execute_stage3_selects_automatic_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common_stage3_flow(monkeypatch)

    ollama_models: list[str] = []
    monkeypatch.setattr("chronicle.stage3.service.resolve_stage3_model", lambda _: "qwen3:8b")
    monkeypatch.setattr("chronicle.stage3.service.require_ollama_config", lambda model: ollama_models.append(model))
    monkeypatch.setattr("chronicle.stage3.service.validate_manual_speaker_map", lambda **_: [])
    monkeypatch.setattr(
        "chronicle.stage3.service.run_ollama_decomposed_backend",
        lambda **_: (
            [
                {
                    "speaker_label": "SPEAKER_00",
                    "assigned_person": "Pat Example",
                    "confidence": "Likely",
                    "candidate_people": ["Pat Example"],
                }
            ],
            {
                "backend": "ollama_decomposed",
                "provider": "ollama",
                "model": "qwen3:8b",
                "workflow": "no-reference-baseline",
                "reference_mode": "none",
                "prompt_version": "stage3-speaker-map-v1",
                "manual_assignment_count": 0,
                "unresolved_speaker_count": 1,
                "llm_assignment_count": 1,
                "unresolved_speakers": ["SPEAKER_00"],
                "token_usage": {"input_tokens": 42, "output_tokens": 18},
            },
            {
                "backend": "ollama_decomposed",
                "provider": "ollama",
                "model": "qwen3:8b",
                "prompt_version": "stage3-speaker-map-v1",
            },
        ),
    )
    monkeypatch.setattr(
        "chronicle.stage3.service.normalize_llm_entries",
        lambda **_: [
            {
                "speaker_label": "SPEAKER_00",
                "assigned_person": "Pat Example",
                "confidence": "Likely",
                "candidate_people": ["Pat Example"],
                "source": "llm",
            }
        ],
    )
    monkeypatch.setattr(
        "chronicle.stage3.service.apply_speaker_map_to_blocks",
        lambda *, blocks, speaker_map: [dict(blocks[0], speaker="Pat Example", candidate_people=["Pat Example"])],
    )

    _, _, _, metadata = execute_stage3(
        manifest=manifest(),
        stage1_dir=tmp_path / "stage1",
        stage2_dir=tmp_path / "stage2",
        stage3_dir=tmp_path / "stage3",
        participants_file=Path("inputs/global/participants.yaml"),
        force=True,
        mode="llm",
        backend="ollama_decomposed",
    )

    artifact = json.loads((tmp_path / "stage3" / "identified_conversation.json").read_text(encoding="utf-8"))
    assert ollama_models == ["qwen3:8b"]
    assert metadata["backend"] == "ollama_decomposed"
    assert artifact["backend"] == "ollama_decomposed"
    assert artifact["backend_usage"]["provider"] == "ollama"
    assert artifact["backend_usage"]["workflow"] == "no-reference-baseline"
    assert artifact["backend_usage"]["reference_mode"] == "none"
    assert artifact["llm_usage"]["provider"] == "ollama"
    assert metadata["llm_usage"]["provider"] == "ollama"


def test_execute_stage3_rejects_unsupported_backend(tmp_path: Path) -> None:
    with pytest.raises(StageExecutionError, match="Unsupported Stage 3 backend `not-a-backend`"):
        execute_stage3(
            manifest=manifest(),
            stage1_dir=tmp_path / "stage1",
            stage2_dir=tmp_path / "stage2",
            stage3_dir=tmp_path / "stage3",
            participants_file=Path("inputs/global/participants.yaml"),
            force=True,
            mode="llm",
            backend="not-a-backend",
        )


def test_run_speechbrain_refmatch_requires_enrollment_coverage(tmp_path: Path) -> None:
    inputs = Stage3Inputs(
        stage1_path=Path("outputs/test-session/stage1/raw_transcript.json"),
        stage1_artifact={"segments": [{"text": "Hello there."}]},
        stage2_path=Path("outputs/test-session/stage2/diarization.json"),
        stage2_artifact={"speaker_labels": ["SPEAKER_00"], "turns": [stage2_turn(1, "audio/a.m4a", 0.0, 1.0, "SPEAKER_00")]},
        participants_file=Path("inputs/global/participants.yaml"),
        participants_by_name={"Pat Example": {"canonical_name": "Pat Example", "voice_references": []}},
        context_path=Path("inputs/sessions/test-session/context.md"),
        context_text="Family interview context.",
    )

    with pytest.raises(StageExecutionError, match="requires participant `voice_references`"):
        run_speechbrain_refmatch(
            manifest=manifest(),
            inputs=inputs,
            speaker_labels=["SPEAKER_00"],
            participants=["Pat Example"],
            manual_entries=[],
            cache_root=tmp_path / "stage3" / "embeddings",
        )


def test_execute_stage3_routes_speechbrain_refmatch_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common_stage3_flow(monkeypatch)
    monkeypatch.setattr("chronicle.stage3.service.validate_manual_speaker_map", lambda **_: [])
    monkeypatch.setattr(
        "chronicle.stage3.service.run_speechbrain_refmatch",
        lambda **_: (
            [
                {
                    "speaker_label": "SPEAKER_00",
                    "assigned_person": "Pat Example",
                    "confidence": "Confirmed",
                    "candidate_people": ["Pat Example"],
                    "source": "deterministic",
                    "evidence": [],
                    "notes": [],
                }
            ],
            {"provider": "speechbrain", "workflow": "reference-match"},
        ),
    )
    monkeypatch.setattr(
        "chronicle.stage3.service.apply_speaker_map_to_blocks",
        lambda *, blocks, speaker_map: [dict(blocks[0], speaker="Pat Example", candidate_people=["Pat Example"])],
    )

    _, _, _, metadata = execute_stage3(
        manifest=manifest(),
        stage1_dir=tmp_path / "stage1",
        stage2_dir=tmp_path / "stage2",
        stage3_dir=tmp_path / "stage3",
        participants_file=Path("inputs/global/participants.yaml"),
        force=True,
        mode="llm",
        backend="speechbrain_refmatch",
    )

    artifact = json.loads((tmp_path / "stage3" / "identified_conversation.json").read_text(encoding="utf-8"))
    assert metadata["backend"] == "speechbrain_refmatch"
    assert metadata["backend_usage"]["provider"] == "speechbrain"
    assert artifact["backend"] == "speechbrain_refmatch"
    assert artifact["backend_usage"]["provider"] == "speechbrain"
    assert "llm_usage" not in artifact
    assert artifact["speaker_map"][0]["assigned_person"] == "Pat Example"


def test_run_speechbrain_hybrid_skips_ollama_for_high_confidence_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = Stage3Inputs(
        stage1_path=Path("outputs/test-session/stage1/raw_transcript.json"),
        stage1_artifact={"segments": [{"text": "Hello there."}]},
        stage2_path=Path("outputs/test-session/stage2/diarization.json"),
        stage2_artifact={
            "speaker_labels": ["SPEAKER_00", "SPEAKER_01"],
            "turns": [
                stage2_turn(1, "audio/a.m4a", 0.0, 1.0, "SPEAKER_00"),
                stage2_turn(2, "audio/a.m4a", 1.0, 2.0, "SPEAKER_01"),
            ],
        },
        participants_file=Path("inputs/global/participants.yaml"),
        participants_by_name={
            "Pat Example": {"canonical_name": "Pat Example", "voice_references": ["inputs/global/pat.wav"]},
            "Bill Example": {"canonical_name": "Bill Example", "voice_references": ["inputs/global/bill.wav"]},
        },
        context_path=Path("inputs/sessions/test-session/context.md"),
        context_text="Family interview context.",
    )

    monkeypatch.setattr(
        "chronicle.stage3.refmatch.resolve_participant_reference_clips",
        lambda **_: {
            "Pat Example": [AudioSliceSpec("Pat Example", "participant_reference", "inputs/global/pat.wav", tmp_path / "pat.wav")],
            "Bill Example": [AudioSliceSpec("Bill Example", "participant_reference", "inputs/global/bill.wav", tmp_path / "bill.wav")],
        },
    )
    monkeypatch.setattr(
        "chronicle.stage3.refmatch.resolve_speaker_audio_slices",
        lambda **_: {
            "SPEAKER_00": [AudioSliceSpec("SPEAKER_00", "speaker_slice", "audio/a.m4a", tmp_path / "speaker0.wav")],
            "SPEAKER_01": [AudioSliceSpec("SPEAKER_01", "speaker_slice", "audio/a.m4a", tmp_path / "speaker1.wav")],
        },
    )
    embeddings = {
        "Pat Example": np.array([1.0, 0.0], dtype=np.float32),
        "Bill Example": np.array([0.0, 1.0], dtype=np.float32),
        "SPEAKER_00": np.array([0.95, 0.05], dtype=np.float32),
        "SPEAKER_01": np.array([0.05, 0.95], dtype=np.float32),
    }

    monkeypatch.setattr(
        "chronicle.stage3.refmatch.prepare_cached_embedding",
        lambda *, slices, **__: type(
            "EmbeddingResult",
            (),
            {"embedding": embeddings[slices[0].owner]},
        )(),
    )
    monkeypatch.setattr(
        "chronicle.stage3.refmatch.run_ollama_speaker_tiebreak",
        lambda **_: pytest.fail("high-confidence hybrid matches should not invoke Ollama"),
    )

    entries, backend_usage, llm_usage = run_speechbrain_hybrid(
        manifest=manifest(),
        inputs=inputs,
        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
        participants=["Pat Example", "Bill Example"],
        manual_entries=[],
        evidence_summary={},
        cache_root=tmp_path / "stage3" / "embeddings",
        llm_model="qwen3:8b",
    )

    assert [entry["assigned_person"] for entry in entries] == ["Pat Example", "Bill Example"]
    assert all(entry["source"] == "deterministic" for entry in entries)
    assert backend_usage["hybrid"]["llm_call_count"] == 0
    assert llm_usage is None


def test_run_speechbrain_hybrid_bounds_candidates_and_call_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = Stage3Inputs(
        stage1_path=Path("outputs/test-session/stage1/raw_transcript.json"),
        stage1_artifact={"segments": [{"text": "Hello there."}]},
        stage2_path=Path("outputs/test-session/stage2/diarization.json"),
        stage2_artifact={
            "speaker_labels": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"],
            "turns": [
                stage2_turn(1, "audio/a.m4a", 0.0, 1.0, "SPEAKER_00"),
                stage2_turn(2, "audio/a.m4a", 1.0, 2.0, "SPEAKER_01"),
                stage2_turn(3, "audio/a.m4a", 2.0, 3.0, "SPEAKER_02"),
                stage2_turn(4, "audio/a.m4a", 3.0, 4.0, "SPEAKER_03"),
            ],
        },
        participants_file=Path("inputs/global/participants.yaml"),
        participants_by_name={
            "Pat Example": {"canonical_name": "Pat Example", "voice_references": ["inputs/global/pat.wav"]},
            "Bill Example": {"canonical_name": "Bill Example", "voice_references": ["inputs/global/bill.wav"]},
            "Chris Example": {"canonical_name": "Chris Example", "voice_references": ["inputs/global/chris.wav"]},
            "Dana Example": {"canonical_name": "Dana Example", "voice_references": ["inputs/global/dana.wav"]},
            "Eve Example": {"canonical_name": "Eve Example", "voice_references": ["inputs/global/eve.wav"]},
        },
        context_path=Path("inputs/sessions/test-session/context.md"),
        context_text="Family interview context.",
    )

    monkeypatch.setattr(
        "chronicle.stage3.refmatch.resolve_participant_reference_clips",
        lambda **_: {
                "Pat Example": [AudioSliceSpec("Pat Example", "participant_reference", "inputs/global/pat.wav", tmp_path / "pat.wav")],
                "Bill Example": [AudioSliceSpec("Bill Example", "participant_reference", "inputs/global/bill.wav", tmp_path / "bill.wav")],
                "Chris Example": [AudioSliceSpec("Chris Example", "participant_reference", "inputs/global/chris.wav", tmp_path / "chris.wav")],
                "Dana Example": [AudioSliceSpec("Dana Example", "participant_reference", "inputs/global/dana.wav", tmp_path / "dana.wav")],
                "Eve Example": [AudioSliceSpec("Eve Example", "participant_reference", "inputs/global/eve.wav", tmp_path / "eve.wav")],
            },
        )
    monkeypatch.setattr(
        "chronicle.stage3.refmatch.resolve_speaker_audio_slices",
        lambda **_: {
            label: [AudioSliceSpec(label, "speaker_slice", "audio/a.m4a", tmp_path / f"{label}.wav")]
            for label in ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"]
        },
    )
    embeddings = {
        "Pat Example": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "Bill Example": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "Chris Example": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "Dana Example": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "Eve Example": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "SPEAKER_00": np.array([0.95, 0.05, 0.0], dtype=np.float32),
        "SPEAKER_01": np.array([0.70, 0.69, 0.01], dtype=np.float32),
        "SPEAKER_02": np.array([0.01, 0.20, 0.79], dtype=np.float32),
        "SPEAKER_03": np.array([-0.95, 0.05, 0.0], dtype=np.float32),
    }
    monkeypatch.setattr(
        "chronicle.stage3.refmatch.prepare_cached_embedding",
        lambda *, slices, **__: type(
            "EmbeddingResult",
            (),
            {"embedding": embeddings[slices[0].owner]},
        )(),
    )

    tiebreak_calls: list[dict[str, object]] = []

    def fake_tiebreak(**kwargs):
        tiebreak_calls.append(kwargs)
        speaker_label = kwargs["speaker_summary"]["speaker_label"]
        return (
            [
                {
                    "speaker_label": speaker_label,
                    "assigned_person": "Bill Example",
                    "confidence": "Likely",
                        "candidate_people": ["Bill Example", "Eve Example"],
                    }
                ],
                {"provider": "ollama", "model": "qwen3:8b", "prompt_version": "stage3-speaker-map-v1"},
            )

    monkeypatch.setattr("chronicle.stage3.refmatch.run_ollama_speaker_tiebreak", fake_tiebreak)

    entries, backend_usage, llm_usage = run_speechbrain_hybrid(
        manifest=SessionManifest(
                session_id="test-session",
                title="Test Session",
                interview_date=None,
                audio_files=["audio/a.m4a"],
                    participants=["Pat Example", "Bill Example", "Chris Example", "Dana Example", "Eve Example"],
                    primary_interviewees=["Pat Example", "Bill Example", "Chris Example", "Dana Example", "Eve Example"],
                people_likely_discussed=[],
                context_doc="context.md",
                manifest_path="inputs/sessions/test-session/session.yaml",
            ),
            inputs=inputs,
            speaker_labels=["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"],
            participants=["Pat Example", "Bill Example", "Chris Example", "Dana Example", "Eve Example"],
            manual_entries=[],
            evidence_summary={
                "SPEAKER_01": {"examples": [{"text": "Maybe Pat or Bill."}]},
            },
            cache_root=tmp_path / "stage3" / "embeddings",
            llm_model="qwen3:8b",
        )

    assert [entry["speaker_label"] for entry in entries] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"]
    assert {entry["assigned_person"] for entry in entries} == {"Pat Example", "Bill Example", "Chris Example", "Dana Example"}
    assert backend_usage["hybrid"]["llm_call_count"] == 1
    assert llm_usage["provider"] == "ollama"
    assert len(tiebreak_calls) == 1
    assert [candidate["canonical_name"] for candidate in tiebreak_calls[0]["participant_candidates"]] == [
        "Bill Example",
        "Eve Example",
    ]


def test_run_speechbrain_hybrid_fails_closed_when_only_one_candidate_remains_after_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = Stage3Inputs(
        stage1_path=Path("outputs/test-session/stage1/raw_transcript.json"),
        stage1_artifact={"segments": [{"text": "Hello there."}]},
        stage2_path=Path("outputs/test-session/stage2/diarization.json"),
        stage2_artifact={
            "speaker_labels": ["SPEAKER_00", "SPEAKER_01"],
            "turns": [
                stage2_turn(1, "audio/a.m4a", 0.0, 1.0, "SPEAKER_00"),
                stage2_turn(2, "audio/a.m4a", 1.0, 2.0, "SPEAKER_01"),
            ],
        },
        participants_file=Path("inputs/global/participants.yaml"),
        participants_by_name={
            "Pat Example": {"canonical_name": "Pat Example", "voice_references": ["inputs/global/pat.wav"]},
            "Bill Example": {"canonical_name": "Bill Example", "voice_references": ["inputs/global/bill.wav"]},
        },
        context_path=Path("inputs/sessions/test-session/context.md"),
        context_text="Family interview context.",
    )

    monkeypatch.setattr(
        "chronicle.stage3.refmatch.resolve_participant_reference_clips",
        lambda **_: {
            "Pat Example": [AudioSliceSpec("Pat Example", "participant_reference", "inputs/global/pat.wav", tmp_path / "pat.wav")],
            "Bill Example": [AudioSliceSpec("Bill Example", "participant_reference", "inputs/global/bill.wav", tmp_path / "bill.wav")],
        },
    )
    monkeypatch.setattr(
        "chronicle.stage3.refmatch.resolve_speaker_audio_slices",
        lambda **_: {
            "SPEAKER_00": [AudioSliceSpec("SPEAKER_00", "speaker_slice", "audio/a.m4a", tmp_path / "speaker0.wav")],
            "SPEAKER_01": [AudioSliceSpec("SPEAKER_01", "speaker_slice", "audio/a.m4a", tmp_path / "speaker1.wav")],
        },
    )
    embeddings = {
        "Pat Example": np.array([1.0, 0.0], dtype=np.float32),
        "Bill Example": np.array([0.0, 1.0], dtype=np.float32),
        "SPEAKER_00": np.array([0.80, 0.20], dtype=np.float32),
        "SPEAKER_01": np.array([0.79, 0.21], dtype=np.float32),
    }
    monkeypatch.setattr(
        "chronicle.stage3.refmatch.prepare_cached_embedding",
        lambda *, slices, **__: type(
            "EmbeddingResult",
            (),
            {"embedding": embeddings[slices[0].owner]},
        )(),
    )
    monkeypatch.setattr(
        "chronicle.stage3.refmatch.run_ollama_speaker_tiebreak",
        lambda **_: pytest.fail("single remaining candidate conflicts should fail closed"),
    )

    with pytest.raises(StageExecutionError, match="Refusing silent auto-assignment"):
        run_speechbrain_hybrid(
            manifest=manifest(),
            inputs=inputs,
            speaker_labels=["SPEAKER_00", "SPEAKER_01"],
            participants=["Pat Example", "Bill Example"],
            manual_entries=[],
            evidence_summary={},
            cache_root=tmp_path / "stage3" / "embeddings",
            llm_model="qwen3:8b",
        )


def test_execute_stage3_routes_speechbrain_hybrid_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_stage3_flow(monkeypatch)
    monkeypatch.setattr("chronicle.stage3.service.validate_manual_speaker_map", lambda **_: [])
    monkeypatch.setattr("chronicle.stage3.service.resolve_stage3_model", lambda _: "qwen3:8b")
    ollama_models: list[str] = []
    monkeypatch.setattr("chronicle.stage3.service.require_ollama_config", lambda model: ollama_models.append(model))
    monkeypatch.setattr(
        "chronicle.stage3.service.run_speechbrain_hybrid",
        lambda **_: (
            [
                {
                    "speaker_label": "SPEAKER_00",
                    "assigned_person": "Pat Example",
                    "confidence": "Likely",
                    "candidate_people": ["Pat Example", "Bill Example"],
                    "source": "llm",
                    "evidence": [],
                    "notes": [],
                }
            ],
            {"provider": "speechbrain", "workflow": "hybrid-reference-match", "hybrid": {"llm_call_count": 1}},
            {"provider": "ollama", "model": "qwen3:8b", "prompt_version": "stage3-speaker-map-v1"},
        ),
    )
    monkeypatch.setattr(
        "chronicle.stage3.service.apply_speaker_map_to_blocks",
        lambda *, blocks, speaker_map: [dict(blocks[0], speaker="Pat Example", candidate_people=["Pat Example"])],
    )

    _, _, _, metadata = execute_stage3(
        manifest=manifest(),
        stage1_dir=tmp_path / "stage1",
        stage2_dir=tmp_path / "stage2",
        stage3_dir=tmp_path / "stage3",
        participants_file=Path("inputs/global/participants.yaml"),
        force=True,
        mode="llm",
        backend="speechbrain_hybrid",
    )

    artifact = json.loads((tmp_path / "stage3" / "identified_conversation.json").read_text(encoding="utf-8"))
    assert ollama_models == ["qwen3:8b"]
    assert metadata["backend"] == "speechbrain_hybrid"
    assert metadata["model"] == "qwen3:8b"
    assert metadata["provider"] == "ollama"
    assert artifact["backend"] == "speechbrain_hybrid"
    assert artifact["backend_usage"]["workflow"] == "hybrid-reference-match"
    assert artifact["llm_usage"]["provider"] == "ollama"


def test_execute_stage3_manual_ignores_backend_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common_stage3_flow(monkeypatch)
    monkeypatch.setattr(
        "chronicle.stage3.service.validate_manual_speaker_map",
        lambda **_: [
            {
                "speaker_label": "SPEAKER_00",
                "assigned_person": "Pat Example",
                "confidence": "Confirmed",
                "candidate_people": ["Pat Example"],
                "source": "manual",
            }
        ],
    )
    monkeypatch.setattr(
        "chronicle.stage3.service.apply_speaker_map_to_blocks",
        lambda *, blocks, speaker_map: [dict(blocks[0], speaker="Pat Example", candidate_people=["Pat Example"])],
    )
    monkeypatch.setattr(
        "chronicle.stage3.service.require_ollama_config",
        lambda _: pytest.fail("manual mode should not validate automatic backends"),
    )
    monkeypatch.setattr(
        "chronicle.stage3.service.run_ollama_decomposed_backend",
        lambda **_: pytest.fail("manual mode should not call automatic backends"),
    )

    _, _, notes, metadata = execute_stage3(
        manifest=manifest(),
        stage1_dir=tmp_path / "stage1",
        stage2_dir=tmp_path / "stage2",
        stage3_dir=tmp_path / "stage3",
        participants_file=Path("inputs/global/participants.yaml"),
        force=True,
        mode="manual",
        backend="not-a-backend",
    )

    artifact = json.loads((tmp_path / "stage3" / "identified_conversation.json").read_text(encoding="utf-8"))
    assert metadata["backend"] is None
    assert "backend_usage" not in artifact
    assert "llm_usage" not in artifact
    assert artifact["speaker_map"][0]["assigned_person"] == "Pat Example"
    assert notes == []


def test_execute_stage3_align_only_ignores_backend_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_stage3_flow(monkeypatch)
    monkeypatch.setattr("chronicle.stage3.service.validate_manual_speaker_map", lambda **_: [])
    monkeypatch.setattr(
        "chronicle.stage3.service.require_ollama_config",
        lambda _: pytest.fail("align-only mode should not validate automatic backends"),
    )
    monkeypatch.setattr(
        "chronicle.stage3.service.run_ollama_decomposed_backend",
        lambda **_: pytest.fail("align-only mode should not call automatic backends"),
    )

    _, _, notes, metadata = execute_stage3(
        manifest=manifest(),
        stage1_dir=tmp_path / "stage1",
        stage2_dir=tmp_path / "stage2",
        stage3_dir=tmp_path / "stage3",
        participants_file=Path("inputs/global/participants.yaml"),
        force=True,
        mode="align-only",
        backend="not-a-backend",
    )

    artifact = json.loads((tmp_path / "stage3" / "aligned_transcript.json").read_text(encoding="utf-8"))
    assert metadata["backend"] is None
    assert "backend_usage" not in artifact
    assert "llm_usage" not in artifact
    assert artifact["speaker_map"] == []
    assert "Align-only mode did not assign real people" in notes[0]


def test_identify_cli_passes_backend_option_and_records_resolved_run_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_execute: dict[str, object] = {}
    captured_run: dict[str, object] = {}
    directories = {
        "stage1": tmp_path / "stage1",
        "stage2": tmp_path / "stage2",
        "stage3": tmp_path / "stage3",
        "stage4": tmp_path / "stage4",
        "runs": tmp_path / "runs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("chronicle.cli.stage3.require_valid_session", lambda *args, **kwargs: manifest())
    monkeypatch.setattr("chronicle.cli.stage3.ensure_output_dirs", lambda session_id: directories)
    monkeypatch.setattr("chronicle.cli.stage3.render_stage_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr("chronicle.cli.stage3.confirm_overwrite_if_needed", lambda **kwargs: kwargs["force"])
    monkeypatch.setattr("chronicle.cli.stage3.resolve_context_path", lambda manifest: Path("inputs/sessions/test-session/context.md"))

    def fake_execute_stage3(**kwargs):
        captured_execute.update(kwargs)
        return (
            ["outputs/test-session/stage3/identified_conversation.json"],
            [],
            [],
            {
                "mode": "llm",
                "backend": "speechbrain_hybrid",
                "model": "qwen3:8b",
                "provider": "ollama",
                "prompt_version": "stage3-speaker-map-v1",
                "schema_version": "1.0",
                "source_stage1_artifact": "outputs/test-session/stage1/raw_transcript.json",
                "source_stage2_artifact": "outputs/test-session/stage2/diarization.json",
                "participants_file": "inputs/global/participants.yaml",
                "context_doc": "inputs/sessions/test-session/context.md",
                "speaker_map_path": None,
                "backend_usage": {
                    "provider": "speechbrain",
                    "workflow": "hybrid-reference-match",
                    "hybrid": {"llm_call_count": 1},
                },
                "llm_usage": {
                    "provider": "ollama",
                    "model": "qwen3:8b",
                    "prompt_version": "stage3-speaker-map-v1",
                },
            },
        )

    def fake_write_run_metadata(**kwargs):
        captured_run.update(kwargs)
        run_path = kwargs["runs_dir"] / "stage3.20260424T000000Z.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text("{}", encoding="utf-8")
        return run_path

    monkeypatch.setattr("chronicle.cli.stage3.execute_stage3", fake_execute_stage3)
    monkeypatch.setattr("chronicle.cli.stage3.write_run_metadata", fake_write_run_metadata)

    result = runner.invoke(
        app,
        [
            "identify",
            "test-session",
            "--participants-file",
            "inputs/global/participants.yaml",
            "--backend",
            "speechbrain_hybrid",
        ],
    )

    assert result.exit_code == 0
    assert captured_execute["backend"] == "speechbrain_hybrid"
    assert captured_run["config"]["backend"] == "speechbrain_hybrid"
    assert captured_run["config"]["backend_requested"] == "speechbrain_hybrid"
    assert captured_run["config"]["backend_usage"]["workflow"] == "hybrid-reference-match"
    assert captured_run["config"]["llm_usage"]["provider"] == "ollama"
    assert "stage3" not in captured_run["config"]


def test_identify_cli_records_default_backend_when_not_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_run: dict[str, object] = {}
    directories = {
        "stage1": tmp_path / "stage1",
        "stage2": tmp_path / "stage2",
        "stage3": tmp_path / "stage3",
        "stage4": tmp_path / "stage4",
        "runs": tmp_path / "runs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("chronicle.cli.stage3.require_valid_session", lambda *args, **kwargs: manifest())
    monkeypatch.setattr("chronicle.cli.stage3.ensure_output_dirs", lambda session_id: directories)
    monkeypatch.setattr("chronicle.cli.stage3.render_stage_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr("chronicle.cli.stage3.confirm_overwrite_if_needed", lambda **kwargs: kwargs["force"])
    monkeypatch.setattr("chronicle.cli.stage3.resolve_context_path", lambda manifest: Path("inputs/sessions/test-session/context.md"))
    monkeypatch.setattr(
        "chronicle.cli.stage3.execute_stage3",
        lambda **kwargs: (
            ["outputs/test-session/stage3/identified_conversation.json"],
            [],
            [],
            {
                "mode": "llm",
                "backend": "ollama_decomposed",
                "model": "qwen3:8b",
                "provider": "ollama",
                "prompt_version": "stage3-speaker-map-v1",
                "schema_version": "1.0",
                "source_stage1_artifact": "outputs/test-session/stage1/raw_transcript.json",
                "source_stage2_artifact": "outputs/test-session/stage2/diarization.json",
                "participants_file": "inputs/global/participants.yaml",
                "context_doc": "inputs/sessions/test-session/context.md",
                "speaker_map_path": None,
                "backend_usage": {
                    "provider": "ollama",
                    "workflow": "no-reference-baseline",
                },
                "llm_usage": {
                    "provider": "ollama",
                    "model": "qwen3:8b",
                    "prompt_version": "stage3-speaker-map-v1",
                },
            },
        ),
    )
    monkeypatch.setattr(
        "chronicle.cli.stage3.write_run_metadata",
        lambda **kwargs: captured_run.update(kwargs) or kwargs["runs_dir"] / "stage3.20260424T000000Z.json",
    )

    result = runner.invoke(
        app,
        [
            "identify",
            "test-session",
            "--participants-file",
            "inputs/global/participants.yaml",
        ],
    )

    assert result.exit_code == 0
    assert captured_run["config"]["backend"] == "ollama_decomposed"
    assert captured_run["config"]["backend_requested"] is None
    assert captured_run["config"]["model"] == "qwen3:8b"


def test_parse_stage3_benchmark_backends_deduplicates_and_validates() -> None:
    assert parse_stage3_benchmark_backends("speechbrain_refmatch,ollama_decomposed,speechbrain_refmatch") == [
        "speechbrain_refmatch",
        "ollama_decomposed",
    ]

    with pytest.raises(StageExecutionError, match="Unsupported Stage 3 backend"):
        parse_stage3_benchmark_backends("not-a-backend")


def test_score_stage3_assignments_reports_exact_accuracy_and_mismatches() -> None:
    result = score_stage3_assignments(
        truth_map={"SPEAKER_00": "Pat Example", "SPEAKER_01": "Bill Example"},
        speaker_map_entries=[
            {"speaker_label": "SPEAKER_00", "assigned_person": "Pat Example"},
            {"speaker_label": "SPEAKER_01", "assigned_person": "Patricia Wrong"},
        ],
    )

    assert result["correct_assignments"] == 1
    assert result["total_assignments"] == 2
    assert result["exact_assignment_accuracy"] == 50.0
    assert result["mismatches"] == [
        {
            "speaker_label": "SPEAKER_01",
            "expected_person": "Bill Example",
            "predicted_person": "Patricia Wrong",
        }
    ]


def test_choose_stage3_benchmark_recommendation_prefers_faster_when_accuracy_is_within_threshold() -> None:
    recommendation = choose_stage3_benchmark_recommendation(
        [
            {
                "backend": "ollama_decomposed",
                "status": "success",
                "exact_assignment_accuracy": 97.0,
                "runtime_seconds": 7.5,
            },
            {
                "backend": "speechbrain_refmatch",
                "status": "success",
                "exact_assignment_accuracy": 95.5,
                "runtime_seconds": 1.4,
            },
            {
                "backend": "speechbrain_hybrid",
                "status": "success",
                "exact_assignment_accuracy": 92.0,
                "runtime_seconds": 3.0,
            },
        ]
    )

    assert recommendation["recommended_backend"] == "speechbrain_refmatch"
    assert recommendation["basis"] == "within_accuracy_threshold_prefer_faster_or_lighter"


def test_run_stage3_benchmark_writes_reports_and_isolates_backend_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    runs_dir = tmp_path / "runs"
    stage1_dir.mkdir()
    stage2_dir.mkdir()
    runs_dir.mkdir()

    truth_file = tmp_path / "truth.yaml"
    truth_file.write_text(
        "speaker_map:\n  SPEAKER_00: Pat Example\n  SPEAKER_01: Bill Example\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "chronicle.stage3.benchmark.load_stage3_truth_map",
        lambda **kwargs: {"SPEAKER_00": "Pat Example", "SPEAKER_01": "Bill Example"},
    )

    def fake_execute_stage3(**kwargs):
        backend = kwargs["backend"]
        stage3_dir = kwargs["stage3_dir"]
        stage3_dir.mkdir(parents=True, exist_ok=True)
        if backend == "speechbrain_hybrid":
            raise StageExecutionError("hybrid benchmark failure")

        artifact = {
            "speaker_map": (
                [
                    {"speaker_label": "SPEAKER_00", "assigned_person": "Pat Example"},
                    {"speaker_label": "SPEAKER_01", "assigned_person": "Bill Example"},
                ]
                if backend == "speechbrain_refmatch"
                else [
                    {"speaker_label": "SPEAKER_00", "assigned_person": "Pat Example"},
                    {"speaker_label": "SPEAKER_01", "assigned_person": "Pat Example"},
                ]
            )
        }
        (stage3_dir / "identified_conversation.json").write_text(json.dumps(artifact), encoding="utf-8")
        return (
            [repo_relative(stage3_dir / "identified_conversation.json")],
            [],
            [f"{backend} notes"],
            {
                "backend_usage": {
                    "workflow": backend,
                    "enrollment_coverage": {
                        "required_participants": ["Pat Example", "Bill Example"],
                        "available_participants": ["Pat Example", "Bill Example"],
                        "missing_participants": [],
                    },
                },
                "llm_usage": {"provider": "ollama"} if backend == "ollama_decomposed" else None,
            },
        )

    monkeypatch.setattr("chronicle.stage3.benchmark.execute_stage3", fake_execute_stage3)

    report, json_path, markdown_path = run_stage3_benchmark(
        manifest=manifest(),
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        runs_dir=runs_dir,
        participants_file=Path("inputs/global/participants.yaml"),
        truth_file=truth_file,
        started_at_label="20260424T000000Z",
        backends=["speechbrain_refmatch", "speechbrain_hybrid", "ollama_decomposed"],
        cpu_feasibility_notes=["CPU-only run on 8 GB machine."],
    )

    assert json_path.exists()
    assert markdown_path.exists()
    assert report["recommendation"]["recommended_backend"] == "speechbrain_refmatch"
    assert [result["status"] for result in report["results"]] == ["success", "failed", "success"]
    assert report["results"][0]["exact_assignment_accuracy"] == 100.0
    assert report["results"][1]["error"] == "hybrid benchmark failure"
    assert report["results"][0]["enrollment_coverage"]["available_participants"] == ["Pat Example", "Bill Example"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Recommended backend: `speechbrain_refmatch`" in markdown
    assert "hybrid benchmark failure" in markdown
    assert "CPU-only run on 8 GB machine." in markdown


def test_run_stage3_benchmark_isolates_unexpected_backend_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    runs_dir = tmp_path / "runs"
    stage1_dir.mkdir()
    stage2_dir.mkdir()
    runs_dir.mkdir()

    truth_file = tmp_path / "truth.yaml"
    truth_file.write_text(
        "speaker_map:\n  SPEAKER_00: Pat Example\n  SPEAKER_01: Bill Example\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "chronicle.stage3.benchmark.load_stage3_truth_map",
        lambda **kwargs: {"SPEAKER_00": "Pat Example", "SPEAKER_01": "Bill Example"},
    )

    def fake_execute_stage3(**kwargs):
        backend = kwargs["backend"]
        stage3_dir = kwargs["stage3_dir"]
        stage3_dir.mkdir(parents=True, exist_ok=True)
        if backend == "speechbrain_hybrid":
            raise RuntimeError("unexpected benchmark crash")
        artifact = {
            "speaker_map": [
                {"speaker_label": "SPEAKER_00", "assigned_person": "Pat Example"},
                {"speaker_label": "SPEAKER_01", "assigned_person": "Bill Example"},
            ]
        }
        (stage3_dir / "identified_conversation.json").write_text(json.dumps(artifact), encoding="utf-8")
        return (
            [repo_relative(stage3_dir / "identified_conversation.json")],
            [],
            [],
            {"backend_usage": {}, "llm_usage": None},
        )

    monkeypatch.setattr("chronicle.stage3.benchmark.execute_stage3", fake_execute_stage3)

    report, _, _ = run_stage3_benchmark(
        manifest=manifest(),
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        runs_dir=runs_dir,
        participants_file=Path("inputs/global/participants.yaml"),
        truth_file=truth_file,
        started_at_label="20260424T010000Z",
        backends=["speechbrain_refmatch", "speechbrain_hybrid", "ollama_decomposed"],
    )

    assert [result["status"] for result in report["results"]] == ["success", "failed", "success"]
    assert report["results"][1]["error"] == "Unexpected error: unexpected benchmark crash"


def test_render_stage3_benchmark_markdown_includes_diagnostics() -> None:
    markdown = render_stage3_benchmark_markdown(
        {
            "session_id": "test-session",
            "truth_file": "inputs/sessions/test-session/benchmark-truth.yaml",
            "backends": ["speechbrain_refmatch"],
            "cpu_feasibility_notes": ["CPU-only run."],
            "recommendation": {
                "recommended_backend": "speechbrain_refmatch",
                "basis": "highest_accuracy",
                "winning_accuracy": 100.0,
                "winning_runtime_seconds": 1.25,
            },
            "results": [
                {
                    "backend": "speechbrain_refmatch",
                    "status": "success",
                    "runtime_seconds": 1.25,
                    "exact_assignment_accuracy": 100.0,
                    "correct_assignments": 2,
                    "total_assignments": 2,
                    "mismatches": [],
                    "enrollment_coverage": {
                        "required_participants": ["Pat Example", "Bill Example"],
                        "available_participants": ["Pat Example", "Bill Example"],
                        "missing_participants": [],
                    },
                    "cpu_feasibility_notes": ["CPU-only run."],
                }
            ],
        }
    )

    assert "Exact assignment accuracy: 100.000% (2/2)" in markdown
    assert "Enrollment coverage: available=2, required=2, missing=0" in markdown
    assert "CPU feasibility notes: CPU-only run." in markdown


def test_benchmark_stage3_cli_passes_expected_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    truth_file = tmp_path / "truth.yaml"
    truth_file.write_text("speaker_map: {}\n", encoding="utf-8")
    directories = {
        "stage1": tmp_path / "stage1",
        "stage2": tmp_path / "stage2",
        "stage3": tmp_path / "stage3",
        "stage4": tmp_path / "stage4",
        "runs": tmp_path / "runs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("chronicle.cli.benchmark.require_valid_session", lambda *args, **kwargs: manifest())
    monkeypatch.setattr("chronicle.cli.benchmark.ensure_output_dirs", lambda session_id: directories)

    def fake_run_stage3_benchmark(**kwargs):
        captured.update(kwargs)
        json_path = kwargs["runs_dir"] / "stage3-benchmark.20260424T000000Z.json"
        markdown_path = kwargs["runs_dir"] / "stage3-benchmark.20260424T000000Z.md"
        json_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("# report\n", encoding="utf-8")
        return (
            {
                "results": [
                    {
                        "backend": "speechbrain_refmatch",
                        "status": "success",
                        "exact_assignment_accuracy": 100.0,
                        "runtime_seconds": 1.0,
                    }
                ],
                "recommendation": {
                    "recommended_backend": "speechbrain_refmatch",
                    "basis": "highest_accuracy",
                },
            },
            json_path,
            markdown_path,
        )

    monkeypatch.setattr("chronicle.cli.benchmark.run_stage3_benchmark", fake_run_stage3_benchmark)

    result = runner.invoke(
        app,
        [
            "benchmark-stage3",
            "test-session",
            "--participants-file",
            "inputs/global/participants.yaml",
            "--truth-file",
            str(truth_file),
            "--backends",
            "speechbrain_refmatch,ollama_decomposed",
            "--cpu-note",
            "CPU-only run.",
        ],
    )

    assert result.exit_code == 0
    assert captured["backends"] == ["speechbrain_refmatch", "ollama_decomposed"]
    assert captured["truth_file"] == truth_file
    assert captured["cpu_feasibility_notes"] == ["CPU-only run."]
