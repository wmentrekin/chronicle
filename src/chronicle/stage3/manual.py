"""Manual speaker-map loading and validation for Stage 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import StageExecutionError
from ..paths import repo_relative
from ..utils import load_yaml
from .schemas import make_speaker_map_entry


def load_manual_speaker_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise StageExecutionError(f"Manual speaker map not found: {repo_relative(path)}")
    payload = load_yaml(path)
    speaker_map = payload.get("speaker_map")
    if not isinstance(speaker_map, dict):
        raise StageExecutionError(
            f"Manual speaker map must contain a top-level `speaker_map` mapping: {repo_relative(path)}"
        )
    return {str(label): str(person) for label, person in speaker_map.items()}


def validate_manual_speaker_map(
    *,
    manual_map: dict[str, str],
    speaker_labels: list[str],
    participants: list[str],
    mode: str,
) -> list[dict[str, Any]]:
    known_labels = set(speaker_labels)
    known_people = set(participants)
    seen_people: set[str] = set()
    entries: list[dict[str, Any]] = []

    for speaker_label, assigned_person in manual_map.items():
        if speaker_label not in known_labels:
            raise StageExecutionError(f"Manual speaker map references unknown speaker label: {speaker_label}")
        if assigned_person not in known_people:
            raise StageExecutionError(
                "Manual speaker map assignments must use canonical manifest participants only. "
                f"Invalid assignment for {speaker_label}: {assigned_person}"
            )
        if assigned_person in seen_people:
            raise StageExecutionError(
                f"Manual speaker map assigns `{assigned_person}` to more than one speaker label."
            )
        seen_people.add(assigned_person)
        entries.append(
            make_speaker_map_entry(
                speaker_label=speaker_label,
                assigned_person=assigned_person,
                confidence="Confirmed",
                candidate_people=[assigned_person],
                source="manual",
                evidence=[{"type": "manual_override", "summary": "Provided by manual speaker-map file."}],
            )
        )

    if mode == "manual":
        missing = sorted(known_labels - set(manual_map))
        if missing:
            raise StageExecutionError(
                "Manual mode requires a complete speaker map. Missing assignments for: "
                + ", ".join(missing)
            )

    return sorted(entries, key=lambda entry: entry["speaker_label"])
