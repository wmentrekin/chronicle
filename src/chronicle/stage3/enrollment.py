"""Enrollment and speaker-audio preparation helpers for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..exceptions import StageExecutionError
from ..paths import REPO_ROOT, repo_relative
from ..session import SessionManifest, resolve_audio_path


@dataclass(frozen=True)
class AudioSliceSpec:
    owner: str
    kind: str
    source_audio: str
    audio_path: Path
    start_seconds: float | None = None
    duration_seconds: float | None = None
    turn_id: int | None = None

    @property
    def end_seconds(self) -> float | None:
        if self.start_seconds is None or self.duration_seconds is None:
            return None
        return self.start_seconds + self.duration_seconds

    def as_cache_fragment(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "kind": self.kind,
            "source_audio": self.source_audio,
            "start_seconds": None if self.start_seconds is None else round(self.start_seconds, 3),
            "duration_seconds": None if self.duration_seconds is None else round(self.duration_seconds, 3),
            "turn_id": self.turn_id,
        }


def _ensure_repo_file(path: Path, *, context: str) -> Path:
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise StageExecutionError(f"{context}: could not resolve path `{path.as_posix()}`.") from exc

    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise StageExecutionError(
            f"{context}: `{resolved.as_posix()}` is outside the repository and cannot be used for local Stage 3 enrollment."
        ) from exc

    if not resolved.exists():
        raise StageExecutionError(f"{context}: missing audio file `{repo_relative(resolved)}`.")
    if not resolved.is_file():
        raise StageExecutionError(f"{context}: expected a file but found `{repo_relative(resolved)}`.")
    return resolved


def resolve_participant_reference_clips(
    *,
    participants_by_name: dict[str, dict[str, Any]],
    participants_file: Path,
) -> dict[str, list[AudioSliceSpec]]:
    resolved: dict[str, list[AudioSliceSpec]] = {}
    for participant_name in sorted(participants_by_name):
        record = participants_by_name[participant_name]
        clips: list[AudioSliceSpec] = []
        for raw_reference in sorted(str(path) for path in (record.get("voice_references") or [])):
            reference_path = Path(raw_reference)
            candidate = reference_path if reference_path.is_absolute() else REPO_ROOT / reference_path
            clip_path = _ensure_repo_file(
                candidate,
                context=(
                    f"Participant `{participant_name}` voice reference from "
                    f"{repo_relative(participants_file)}"
                ),
            )
            clips.append(
                AudioSliceSpec(
                    owner=participant_name,
                    kind="participant_reference",
                    source_audio=repo_relative(clip_path),
                    audio_path=clip_path,
                )
            )
        resolved[participant_name] = clips
    return resolved


def resolve_speaker_audio_slices(
    *,
    manifest: SessionManifest,
    stage2_artifact: dict[str, Any],
) -> dict[str, list[AudioSliceSpec]]:
    grouped: dict[str, list[AudioSliceSpec]] = {}
    turns = stage2_artifact.get("turns")
    if not isinstance(turns, list):
        raise StageExecutionError("Stage 2 artifact has no diarization turns to prepare Stage 3 speaker slices.")

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker_label = str(turn.get("speaker_label") or "").strip()
        if not speaker_label:
            raise StageExecutionError("Stage 2 diarization turn is missing `speaker_label`.")

        source_audio = str(turn.get("source_audio") or "").strip()
        if not source_audio:
            raise StageExecutionError(f"Stage 2 diarization turn for `{speaker_label}` is missing `source_audio`.")

        try:
            start_seconds = float(turn["source_start_seconds"])
            end_seconds = float(turn["source_end_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StageExecutionError(
                f"Stage 2 diarization turn for `{speaker_label}` has invalid source-relative timing."
            ) from exc

        duration_seconds = round(end_seconds - start_seconds, 3)
        if duration_seconds <= 0:
            raise StageExecutionError(
                f"Stage 2 diarization turn for `{speaker_label}` must have a positive duration."
            )

        audio_path = _ensure_repo_file(
            resolve_audio_path(manifest, source_audio),
            context=f"Stage 2 diarization turn audio for `{speaker_label}`",
        )
        grouped.setdefault(speaker_label, []).append(
            AudioSliceSpec(
                owner=speaker_label,
                kind="speaker_slice",
                source_audio=repo_relative(audio_path),
                audio_path=audio_path,
                start_seconds=round(start_seconds, 3),
                duration_seconds=duration_seconds,
                turn_id=int(turn["turn_id"]) if turn.get("turn_id") is not None else None,
            )
        )

    return {
        speaker_label: sorted(
            slices,
            key=lambda item: (
                item.source_audio,
                item.start_seconds if item.start_seconds is not None else -1.0,
                item.duration_seconds if item.duration_seconds is not None else -1.0,
                item.turn_id if item.turn_id is not None else -1,
            ),
        )
        for speaker_label, slices in sorted(grouped.items())
    }
