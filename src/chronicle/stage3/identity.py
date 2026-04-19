"""Speaker identity validation and application for Stage 3."""

from __future__ import annotations

from typing import Any

from ..exceptions import StageExecutionError
from .schemas import CONFIDENCE_LABELS, IDENTITY_SOURCES, make_speaker_map_entry


def validate_speaker_count(*, speaker_labels: list[str], participants: list[str], mode: str) -> None:
    if mode == "align-only":
        return
    if len(speaker_labels) != len(participants):
        raise StageExecutionError(
            "Stage 3 identity assignment expects the number of Stage 2 speakers to match "
            "`manifest.participants` for the first implementation. "
            f"Stage 2 speakers: {len(speaker_labels)} ({', '.join(speaker_labels)}). "
            f"Manifest participants: {len(participants)} ({', '.join(participants)}). "
            "Use `chronicle identify <session_id> --mode align-only` to inspect alignment, "
            "or rerun Stage 2 with speaker-count constraints."
        )


def normalize_llm_entries(
    *,
    raw_entries: list[dict[str, Any]],
    speaker_labels: list[str],
    participants: list[str],
    manual_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manual_labels = {entry["speaker_label"] for entry in manual_entries}
    normalized: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise StageExecutionError("OpenAI speaker-map entries must be JSON objects.")
        speaker_label = str(raw_entry.get("speaker_label") or "")
        if speaker_label in manual_labels:
            continue
        normalized.append(
            make_speaker_map_entry(
                speaker_label=speaker_label,
                assigned_person=str(raw_entry.get("assigned_person") or ""),
                confidence=str(raw_entry.get("confidence") or "Needs review"),
                candidate_people=[
                    str(person)
                    for person in raw_entry.get("candidate_people", [])
                    if isinstance(person, str)
                ],
                source="llm",
                evidence=[
                    item for item in raw_entry.get("evidence", []) if isinstance(item, dict)
                ],
                notes=[
                    str(note)
                    for note in raw_entry.get("notes", [])
                    if isinstance(note, str)
                ],
            )
        )
    return validate_speaker_map_entries(
        entries=manual_entries + normalized,
        speaker_labels=speaker_labels,
        participants=participants,
        require_complete=True,
    )


def validate_speaker_map_entries(
    *,
    entries: list[dict[str, Any]],
    speaker_labels: list[str],
    participants: list[str],
    require_complete: bool,
) -> list[dict[str, Any]]:
    known_labels = set(speaker_labels)
    known_people = set(participants)
    seen_labels: set[str] = set()
    seen_people: set[str] = set()
    validated: list[dict[str, Any]] = []

    for entry in entries:
        speaker_label = entry.get("speaker_label")
        assigned_person = entry.get("assigned_person")
        confidence = entry.get("confidence")
        source = entry.get("source")
        if speaker_label not in known_labels:
            raise StageExecutionError(f"Speaker map references unknown Stage 2 speaker label: {speaker_label}")
        if speaker_label in seen_labels:
            raise StageExecutionError(f"Speaker map contains duplicate entry for {speaker_label}.")
        if assigned_person not in known_people:
            raise StageExecutionError(
                f"Speaker map assignment for {speaker_label} is not a manifest participant: {assigned_person}"
            )
        if assigned_person in seen_people:
            raise StageExecutionError(
                f"Speaker map assigns `{assigned_person}` to more than one anonymous speaker."
            )
        if confidence not in CONFIDENCE_LABELS:
            raise StageExecutionError(f"Invalid confidence label for {speaker_label}: {confidence}")
        if source not in IDENTITY_SOURCES:
            raise StageExecutionError(f"Invalid speaker-map source for {speaker_label}: {source}")

        candidate_people = entry.get("candidate_people")
        if not isinstance(candidate_people, list):
            raise StageExecutionError(f"`candidate_people` must be a list for {speaker_label}.")
        invalid_candidates = [person for person in candidate_people if person not in known_people]
        if invalid_candidates:
            raise StageExecutionError(
                f"Speaker map for {speaker_label} includes non-participant candidates: "
                + ", ".join(str(person) for person in invalid_candidates)
            )

        seen_labels.add(str(speaker_label))
        seen_people.add(str(assigned_person))
        validated.append(entry)

    if require_complete:
        missing = sorted(known_labels - seen_labels)
        if missing:
            raise StageExecutionError("Speaker map is incomplete. Missing: " + ", ".join(missing))

    return sorted(validated, key=lambda item: item["speaker_label"])


def apply_speaker_map_to_blocks(
    *,
    blocks: list[dict[str, Any]],
    speaker_map: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_label = {entry["speaker_label"]: entry for entry in speaker_map}
    identified_blocks: list[dict[str, Any]] = []
    for block in blocks:
        copied = dict(block)
        alignment_confidence = copied.get("alignment", {}).get("alignment_confidence")
        speaker_label = copied.get("speaker_label")
        entry = by_label.get(speaker_label)

        if alignment_confidence == "Needs review" or not entry:
            copied["speaker"] = None
            copied["candidate_people"] = []
            copied["confidence"] = "Needs review"
            notes = list(copied.get("notes") or [])
            if speaker_label and not entry:
                notes.append("[speaker assignment uncertain] No valid speaker-map entry was available.")
            copied["notes"] = notes
            identified_blocks.append(copied)
            continue

        copied["speaker"] = entry["assigned_person"]
        copied["candidate_people"] = list(entry.get("candidate_people") or [entry["assigned_person"]])
        copied["confidence"] = str(entry.get("confidence") or "Needs review")
        identified_blocks.append(copied)
    return identified_blocks
