"""Input loading and normalization helpers for Stage 3."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..exceptions import StageExecutionError
from ..paths import DEFAULT_PARTICIPANTS_FILE, REPO_ROOT, repo_relative
from ..session import SessionManifest, resolve_context_path
from ..stage1.artifacts import parse_timestamp_seconds, session_stage1_output_paths
from ..stage2.artifacts import stage2_output_paths
from ..utils import load_yaml


@dataclass(frozen=True)
class Stage3Inputs:
    stage1_path: Path
    stage1_artifact: dict[str, Any]
    stage2_path: Path
    stage2_artifact: dict[str, Any]
    participants_file: Path
    participants_by_name: dict[str, dict[str, Any]]
    context_path: Path
    context_text: str
    participant_voice_references: dict[str, list[str]] = field(default_factory=dict)


def normalize_source_audio(value: object, manifest: SessionManifest | None = None) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""

    path = Path(raw_value)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(REPO_ROOT / path)
        if manifest and manifest.manifest_path:
            candidates.append(Path(manifest.manifest_path).resolve().parent / path)
        if manifest:
            candidates.append(REPO_ROOT / "inputs" / "sessions" / manifest.session_id / path)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.exists():
                return repo_relative(resolved)
        except OSError:
            continue

    normalized = raw_value.replace("\\", "/").lstrip("./")
    marker = f"inputs/sessions/{manifest.session_id}/" if manifest else "inputs/sessions/"
    marker_index = normalized.find(marker)
    if marker_index >= 0:
        return normalized[marker_index:]
    if manifest and normalized.startswith("audio/"):
        return f"inputs/sessions/{manifest.session_id}/{normalized}"
    return normalized


def normalize_voice_references(value: object, *, participants_file: Path) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise StageExecutionError(
            f"`voice_references` must be a list of paths when provided in {repo_relative(participants_file)}"
        )

    normalized: list[str] = []
    for raw_path in value:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise StageExecutionError(
                f"`voice_references` entries must be non-empty strings in {repo_relative(participants_file)}"
            )
        path = Path(raw_path.strip())
        candidates = [path] if path.is_absolute() else []
        if not path.is_absolute():
            normalized_raw_path = path.as_posix().lstrip("./")
            if normalized_raw_path.startswith(("inputs/", "outputs/", "src/", "tests/", "docs/", "examples/")):
                candidates.append(REPO_ROOT / path)
            else:
                candidates.append(participants_file.resolve().parent / path)
                candidates.append(REPO_ROOT / path)

        normalized_path = path.as_posix()
        for candidate in candidates:
            try:
                normalized_path = repo_relative(candidate)
                break
            except OSError:
                continue
        normalized.append(normalized_path)
    return normalized


def load_participant_records(participants_file: Path = DEFAULT_PARTICIPANTS_FILE) -> dict[str, dict[str, Any]]:
    payload = load_yaml(participants_file)
    participants = payload.get("participants")
    if not isinstance(participants, list):
        raise StageExecutionError(
            f"`participants` list missing or invalid in {repo_relative(participants_file)}"
        )

    records: dict[str, dict[str, Any]] = {}
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        canonical_name = participant.get("canonical_name")
        if isinstance(canonical_name, str) and canonical_name.strip():
            normalized_participant = dict(participant)
            normalized_participant["voice_references"] = normalize_voice_references(
                participant.get("voice_references"),
                participants_file=participants_file,
            )
            records[canonical_name.strip()] = normalized_participant
    return records


def load_json_artifact(path: Path, missing_message: str) -> dict[str, Any]:
    if not path.exists():
        raise StageExecutionError(missing_message)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageExecutionError(f"Invalid JSON artifact: {repo_relative(path)}") from exc
    if not isinstance(payload, dict):
        raise StageExecutionError(f"Artifact must be a JSON object: {repo_relative(path)}")
    return payload


def load_stage1_artifact(manifest: SessionManifest, stage1_dir: Path) -> tuple[Path, dict[str, Any]]:
    path, _ = session_stage1_output_paths(stage1_dir)
    payload = load_json_artifact(
        path,
        "Stage 3 requires Stage 1 output. Missing: "
        f"{repo_relative(path)}. Run `chronicle transcribe {manifest.session_id}` first.",
    )
    if payload.get("session_id") != manifest.session_id:
        raise StageExecutionError(
            f"Stage 1 artifact session_id does not match `{manifest.session_id}`: {repo_relative(path)}"
        )
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise StageExecutionError(f"Stage 1 artifact has no transcript segments: {repo_relative(path)}")
    return path, payload


def load_stage2_artifact(manifest: SessionManifest, stage2_dir: Path) -> tuple[Path, dict[str, Any]]:
    path, _ = stage2_output_paths(stage2_dir)
    payload = load_json_artifact(
        path,
        "Stage 3 requires Stage 2 output. Missing: "
        f"{repo_relative(path)}. Run `chronicle diarize {manifest.session_id}` first.",
    )
    if payload.get("session_id") != manifest.session_id:
        raise StageExecutionError(
            f"Stage 2 artifact session_id does not match `{manifest.session_id}`: {repo_relative(path)}"
        )
    if payload.get("status", "complete") != "complete":
        raise StageExecutionError(f"Stage 2 artifact is not complete: {repo_relative(path)}")
    if not isinstance(payload.get("turns"), list) or not payload["turns"]:
        raise StageExecutionError(f"Stage 2 artifact has no diarization turns: {repo_relative(path)}")
    if not isinstance(payload.get("speaker_labels"), list) or not payload["speaker_labels"]:
        raise StageExecutionError(f"Stage 2 artifact has no speaker labels: {repo_relative(path)}")
    return path, payload


def load_stage3_inputs(
    *,
    manifest: SessionManifest,
    stage1_dir: Path,
    stage2_dir: Path,
    participants_file: Path,
) -> Stage3Inputs:
    stage1_path, stage1_artifact = load_stage1_artifact(manifest, stage1_dir)
    stage2_path, stage2_artifact = load_stage2_artifact(manifest, stage2_dir)
    participants_by_name = load_participant_records(participants_file)
    missing_participants = [name for name in manifest.participants if name not in participants_by_name]
    if missing_participants:
        raise StageExecutionError(
            "Session manifest participants are missing from participants file: "
            + ", ".join(missing_participants)
        )
    context_path = resolve_context_path(manifest)
    context_text = context_path.read_text(encoding="utf-8")
    return Stage3Inputs(
        stage1_path=stage1_path,
        stage1_artifact=stage1_artifact,
        stage2_path=stage2_path,
        stage2_artifact=stage2_artifact,
        participants_file=participants_file,
        participants_by_name=participants_by_name,
        context_path=context_path,
        context_text=context_text,
        participant_voice_references={
            name: list(record.get("voice_references") or [])
            for name, record in participants_by_name.items()
        },
    )


def stage1_segment_seconds(segment: dict[str, Any]) -> tuple[Optional[float], Optional[float], str]:
    source_start = parse_timestamp_seconds(segment.get("source_start"))
    source_end = parse_timestamp_seconds(segment.get("source_end"))
    if source_start is not None and source_end is not None:
        return source_start, source_end, "source_relative"
    return parse_timestamp_seconds(segment.get("start")), parse_timestamp_seconds(segment.get("end")), "session_relative_fallback"
