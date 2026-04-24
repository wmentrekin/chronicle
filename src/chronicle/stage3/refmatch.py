"""SpeechBrain reference-matching backend for Stage 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import StageExecutionError
from ..paths import repo_relative
from ..session import SessionManifest
from .embeddings import DEFAULT_STAGE3_SPEECHBRAIN_MODEL, prepare_cached_embedding
from .enrollment import resolve_participant_reference_clips, resolve_speaker_audio_slices
from .identity import validate_speaker_map_entries
from .inputs import Stage3Inputs
from .llm import run_ollama_speaker_tiebreak
from .schemas import (
    DEFAULT_HYBRID_MAX_CANDIDATES,
    DEFAULT_REFMATCH_CONFIRMED_MARGIN,
    DEFAULT_REFMATCH_CONFIRMED_SIMILARITY,
    DEFAULT_REFMATCH_MIN_MARGIN,
    DEFAULT_REFMATCH_MIN_SIMILARITY,
    make_speaker_map_entry,
)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = np.asarray(left, dtype=np.float32)
    right_vector = np.asarray(right, dtype=np.float32)
    left_norm = np.linalg.norm(left_vector)
    right_norm = np.linalg.norm(right_vector)
    if left_norm == 0 or right_norm == 0:
        raise StageExecutionError("SpeechBrain refmatch received a zero-length embedding vector.")
    return float(np.dot(left_vector, right_vector) / (left_norm * right_norm))


def build_similarity_matrix(
    *,
    speaker_embeddings: dict[str, np.ndarray],
    participant_embeddings: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    return {
        speaker_label: {
            participant_name: round(cosine_similarity(speaker_vector, participant_vector), 6)
            for participant_name, participant_vector in sorted(participant_embeddings.items())
        }
        for speaker_label, speaker_vector in sorted(speaker_embeddings.items())
    }


def assign_similarity_matches(
    *,
    similarity_matrix: dict[str, dict[str, float]],
    min_similarity: float = DEFAULT_REFMATCH_MIN_SIMILARITY,
    min_margin: float = DEFAULT_REFMATCH_MIN_MARGIN,
    confirmed_similarity: float = DEFAULT_REFMATCH_CONFIRMED_SIMILARITY,
    confirmed_margin: float = DEFAULT_REFMATCH_CONFIRMED_MARGIN,
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    assigned_people: dict[str, str] = {}

    for speaker_label in sorted(similarity_matrix):
        scores = similarity_matrix[speaker_label]
        if not scores:
            raise StageExecutionError(f"SpeechBrain refmatch has no participant scores for `{speaker_label}`.")

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_person, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best_score - second_score if len(ranked) > 1 else best_score

        if best_score < min_similarity:
            raise StageExecutionError(
                f"SpeechBrain refmatch low-confidence match for `{speaker_label}`: "
                f"{best_person} scored {best_score:.3f}, below threshold {min_similarity:.3f}."
            )
        if len(ranked) > 1 and margin < min_margin:
            raise StageExecutionError(
                f"SpeechBrain refmatch ambiguous match for `{speaker_label}`: "
                f"top margin {margin:.3f} is below threshold {min_margin:.3f}."
            )
        if best_person in assigned_people:
            raise StageExecutionError(
                "SpeechBrain refmatch duplicate-best-match conflict: "
                f"`{best_person}` is the top match for both `{assigned_people[best_person]}` and `{speaker_label}`."
            )

        assigned_people[best_person] = speaker_label
        confidence = (
            "Confirmed"
            if best_score >= confirmed_similarity and margin >= confirmed_margin
            else "Likely"
        )
        assignments.append(
            make_speaker_map_entry(
                speaker_label=speaker_label,
                assigned_person=best_person,
                confidence=confidence,
                candidate_people=[participant_name for participant_name, _ in ranked],
                source="deterministic",
                evidence=[
                    {
                        "type": "embedding_similarity",
                        "top_similarity": round(best_score, 6),
                        "margin": round(margin, 6),
                    }
                ],
                notes=[
                    f"Top similarity {best_score:.3f}",
                    f"Top margin {margin:.3f}",
                ],
            )
        )

    return assignments


def _rank_similarity_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _describe_similarity_choices(
    *,
    speaker_label: str,
    scores: dict[str, float],
    min_similarity: float,
    min_margin: float,
) -> dict[str, Any]:
    if not scores:
        raise StageExecutionError(f"SpeechBrain refmatch has no participant scores for `{speaker_label}`.")
    ranked = _rank_similarity_scores(scores)
    best_person, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = best_score - second_score if len(ranked) > 1 else best_score
    return {
        "speaker_label": speaker_label,
        "ranked": ranked,
        "best_person": best_person,
        "best_score": best_score,
        "margin": margin,
        "is_low_confidence": best_score < min_similarity,
        "is_low_margin": len(ranked) > 1 and margin < min_margin,
    }


def _build_hybrid_candidate_subset(
    *,
    ranked: list[tuple[str, float]],
    unavailable_people: set[str],
    max_candidates: int,
) -> list[str]:
    available = [name for name, _ in ranked if name not in unavailable_people]
    return available[:max_candidates]


def _normalize_hybrid_llm_entry(
    *,
    raw_entry: dict[str, Any],
    speaker_label: str,
    candidate_people: list[str],
    scores: dict[str, float],
) -> dict[str, Any]:
    assigned_person = str(raw_entry.get("assigned_person") or "")
    if assigned_person not in candidate_people:
        raise StageExecutionError(
            f"Hybrid Ollama tie-break returned invalid assignment for `{speaker_label}`: {assigned_person}"
        )

    confidence = str(raw_entry.get("confidence") or "Likely")
    notes = [str(note) for note in raw_entry.get("notes", []) if isinstance(note, str)]
    evidence = [item for item in raw_entry.get("evidence", []) if isinstance(item, dict)]
    evidence.append(
        {
            "type": "embedding_similarity",
            "candidate_scores": [
                {"assigned_person": person, "similarity": round(scores[person], 6)}
                for person in candidate_people
            ],
        }
    )
    return make_speaker_map_entry(
        speaker_label=speaker_label,
        assigned_person=assigned_person,
        confidence=confidence,
        candidate_people=candidate_people,
        source="llm",
        evidence=evidence,
        notes=notes,
    )


def run_speechbrain_refmatch(
    *,
    manifest: SessionManifest,
    inputs: Stage3Inputs,
    speaker_labels: list[str],
    participants: list[str],
    manual_entries: list[dict[str, Any]],
    cache_root: Path,
    model_name: str = DEFAULT_STAGE3_SPEECHBRAIN_MODEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manual_labels = {str(entry["speaker_label"]) for entry in manual_entries}
    manual_people = {str(entry["assigned_person"]) for entry in manual_entries}
    unresolved_speakers = [label for label in speaker_labels if label not in manual_labels]
    unresolved_people = [name for name in participants if name not in manual_people]

    reference_clips = resolve_participant_reference_clips(
        participants_by_name=inputs.participants_by_name,
        participants_file=inputs.participants_file,
    )
    missing_people = [name for name in unresolved_people if not reference_clips.get(name)]
    if missing_people:
        raise StageExecutionError(
            "SpeechBrain refmatch requires participant `voice_references` for all unresolved participants. Missing: "
            + ", ".join(sorted(missing_people))
        )

    speaker_slices = resolve_speaker_audio_slices(
        manifest=manifest,
        stage2_artifact=inputs.stage2_artifact,
    )
    missing_speakers = [label for label in unresolved_speakers if not speaker_slices.get(label)]
    if missing_speakers:
        raise StageExecutionError(
            "SpeechBrain refmatch could not prepare speaker audio slices for: "
            + ", ".join(sorted(missing_speakers))
        )

    participant_embeddings = {
        participant_name: prepare_cached_embedding(
            slices=reference_clips[participant_name],
            cache_root=cache_root / "participant-enrollment",
            model_name=model_name,
        ).embedding
        for participant_name in unresolved_people
    }
    speaker_embeddings = {
        speaker_label: prepare_cached_embedding(
            slices=speaker_slices[speaker_label],
            cache_root=cache_root / "speaker-profiles",
            model_name=model_name,
        ).embedding
        for speaker_label in unresolved_speakers
    }

    similarity_matrix = build_similarity_matrix(
        speaker_embeddings=speaker_embeddings,
        participant_embeddings=participant_embeddings,
    )
    auto_entries = assign_similarity_matches(similarity_matrix=similarity_matrix)
    validated_entries = validate_speaker_map_entries(
        entries=manual_entries + auto_entries,
        speaker_labels=speaker_labels,
        participants=participants,
        require_complete=True,
    )

    usage = {
        "provider": "speechbrain",
        "model": model_name,
        "workflow": "reference-match",
        "cache_root": repo_relative(cache_root),
        "thresholds": {
            "min_similarity": DEFAULT_REFMATCH_MIN_SIMILARITY,
            "min_margin": DEFAULT_REFMATCH_MIN_MARGIN,
            "confirmed_similarity": DEFAULT_REFMATCH_CONFIRMED_SIMILARITY,
            "confirmed_margin": DEFAULT_REFMATCH_CONFIRMED_MARGIN,
        },
        "enrollment_coverage": {
            "required_participants": unresolved_people,
            "available_participants": sorted(participant_embeddings),
            "missing_participants": missing_people,
        },
        "speakers": [
            {
                "speaker_label": speaker_label,
                "top_match": max(scores.items(), key=lambda item: (item[1], item[0]))[0],
                "top_similarity": max(scores.values()),
                "candidate_people": [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))],
            }
            for speaker_label, scores in sorted(similarity_matrix.items())
        ],
    }
    return validated_entries, usage


def run_speechbrain_hybrid(
    *,
    manifest: SessionManifest,
    inputs: Stage3Inputs,
    speaker_labels: list[str],
    participants: list[str],
    manual_entries: list[dict[str, Any]],
    evidence_summary: dict[str, Any],
    cache_root: Path,
    model_name: str = DEFAULT_STAGE3_SPEECHBRAIN_MODEL,
    llm_model: str,
    min_similarity: float = DEFAULT_REFMATCH_MIN_SIMILARITY,
    min_margin: float = DEFAULT_REFMATCH_MIN_MARGIN,
    confirmed_similarity: float = DEFAULT_REFMATCH_CONFIRMED_SIMILARITY,
    confirmed_margin: float = DEFAULT_REFMATCH_CONFIRMED_MARGIN,
    max_tiebreak_candidates: int = DEFAULT_HYBRID_MAX_CANDIDATES,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    manual_labels = {str(entry["speaker_label"]) for entry in manual_entries}
    manual_people = {str(entry["assigned_person"]) for entry in manual_entries}
    unresolved_speakers = [label for label in speaker_labels if label not in manual_labels]
    unresolved_people = [name for name in participants if name not in manual_people]

    reference_clips = resolve_participant_reference_clips(
        participants_by_name=inputs.participants_by_name,
        participants_file=inputs.participants_file,
    )
    missing_people = [name for name in unresolved_people if not reference_clips.get(name)]
    if missing_people:
        raise StageExecutionError(
            "SpeechBrain hybrid requires participant `voice_references` for all unresolved participants. Missing: "
            + ", ".join(sorted(missing_people))
        )

    speaker_slices = resolve_speaker_audio_slices(
        manifest=manifest,
        stage2_artifact=inputs.stage2_artifact,
    )
    missing_speakers = [label for label in unresolved_speakers if not speaker_slices.get(label)]
    if missing_speakers:
        raise StageExecutionError(
            "SpeechBrain hybrid could not prepare speaker audio slices for: "
            + ", ".join(sorted(missing_speakers))
        )

    participant_embeddings = {
        participant_name: prepare_cached_embedding(
            slices=reference_clips[participant_name],
            cache_root=cache_root / "participant-enrollment",
            model_name=model_name,
        ).embedding
        for participant_name in unresolved_people
    }
    speaker_embeddings = {
        speaker_label: prepare_cached_embedding(
            slices=speaker_slices[speaker_label],
            cache_root=cache_root / "speaker-profiles",
            model_name=model_name,
        ).embedding
        for speaker_label in unresolved_speakers
    }

    similarity_matrix = build_similarity_matrix(
        speaker_embeddings=speaker_embeddings,
        participant_embeddings=participant_embeddings,
    )
    similarity_details = {
        speaker_label: _describe_similarity_choices(
            speaker_label=speaker_label,
            scores=scores,
            min_similarity=min_similarity,
            min_margin=min_margin,
        )
        for speaker_label, scores in sorted(similarity_matrix.items())
    }

    duplicate_groups: dict[str, list[dict[str, Any]]] = {}
    for detail in similarity_details.values():
        duplicate_groups.setdefault(str(detail["best_person"]), []).append(detail)
    conflict_winners = {
        person: sorted(
            group,
            key=lambda item: (-float(item["best_score"]), -float(item["margin"]), item["speaker_label"]),
        )[0]["speaker_label"]
        for person, group in duplicate_groups.items()
        if len(group) > 1
    }

    auto_entries: list[dict[str, Any]] = []
    pending_tiebreaks: list[dict[str, Any]] = []
    assigned_people = set(manual_people)

    for speaker_label in sorted(similarity_details):
        detail = similarity_details[speaker_label]
        ranked = detail["ranked"]
        best_person = str(detail["best_person"])
        best_score = float(detail["best_score"])
        margin = float(detail["margin"])

        if detail["is_low_confidence"]:
            raise StageExecutionError(
                f"SpeechBrain hybrid low-confidence match for `{speaker_label}`: "
                f"{best_person} scored {best_score:.3f}, below threshold {min_similarity:.3f}."
            )

        if best_person in conflict_winners and conflict_winners[best_person] != speaker_label:
            pending_tiebreaks.append(detail)
            continue

        if detail["is_low_margin"]:
            pending_tiebreaks.append(detail)
            continue

        if best_person in assigned_people:
            pending_tiebreaks.append(detail)
            continue

        assigned_people.add(best_person)
        confidence = (
            "Confirmed"
            if best_score >= confirmed_similarity and margin >= confirmed_margin
            else "Likely"
        )
        auto_entries.append(
            make_speaker_map_entry(
                speaker_label=speaker_label,
                assigned_person=best_person,
                confidence=confidence,
                candidate_people=[participant_name for participant_name, _ in ranked],
                source="deterministic",
                evidence=[
                    {
                        "type": "embedding_similarity",
                        "top_similarity": round(best_score, 6),
                        "margin": round(margin, 6),
                    }
                ],
                notes=[
                    f"Top similarity {best_score:.3f}",
                    f"Top margin {margin:.3f}",
                ],
            )
        )

    llm_usage: dict[str, Any] | None = None
    llm_call_count = 0
    for detail in sorted(
        pending_tiebreaks,
        key=lambda item: (item["speaker_label"], -float(item["best_score"]), float(item["margin"])),
    ):
        speaker_label = str(detail["speaker_label"])
        candidate_people = _build_hybrid_candidate_subset(
            ranked=detail["ranked"],
            unavailable_people=assigned_people,
            max_candidates=max_tiebreak_candidates,
        )
        if len(candidate_people) == 0:
            raise StageExecutionError(
                f"SpeechBrain hybrid exhausted candidate people for `{speaker_label}` after deterministic assignments."
            )
        if len(candidate_people) == 1:
            raise StageExecutionError(
                "SpeechBrain hybrid unresolved conflict for "
                f"`{speaker_label}` left only one candidate after one-to-one filtering. "
                "Refusing silent auto-assignment."
            )

        raw_entries, last_usage = run_ollama_speaker_tiebreak(
            manifest=manifest,
            context_text=inputs.context_text,
            participant_candidates=[
                {
                    "canonical_name": name,
                    "metadata": inputs.participants_by_name.get(name, {}),
                    "similarity": round(similarity_matrix[speaker_label][name], 6),
                }
                for name in candidate_people
            ],
            unavailable_people=sorted(assigned_people),
            speaker_summary={
                "speaker_label": speaker_label,
                **(evidence_summary.get(speaker_label) or {}),
                "candidate_people": candidate_people,
                "similarity_scores": {
                    name: round(similarity_matrix[speaker_label][name], 6)
                    for name in candidate_people
                },
            },
            model=llm_model,
        )
        if len(raw_entries) != 1:
            raise StageExecutionError(
                f"Hybrid Ollama tie-break must return exactly one speaker-map entry for `{speaker_label}`."
            )
        entry = _normalize_hybrid_llm_entry(
            raw_entry=raw_entries[0],
            speaker_label=speaker_label,
            candidate_people=candidate_people,
            scores=similarity_matrix[speaker_label],
        )
        auto_entries.append(entry)
        assigned_people.add(entry["assigned_person"])
        llm_usage = dict(last_usage)
        llm_call_count += 1

    validated_entries = validate_speaker_map_entries(
        entries=manual_entries + auto_entries,
        speaker_labels=speaker_labels,
        participants=participants,
        require_complete=True,
    )
    backend_usage = {
        "provider": "speechbrain",
        "model": model_name,
        "workflow": "hybrid-reference-match",
        "cache_root": repo_relative(cache_root),
        "thresholds": {
            "min_similarity": min_similarity,
            "min_margin": min_margin,
            "confirmed_similarity": confirmed_similarity,
            "confirmed_margin": confirmed_margin,
        },
        "hybrid": {
            "llm_model": llm_model,
            "max_tiebreak_candidates": max_tiebreak_candidates,
            "llm_call_count": llm_call_count,
        },
        "enrollment_coverage": {
            "required_participants": unresolved_people,
            "available_participants": sorted(participant_embeddings),
            "missing_participants": missing_people,
        },
        "speakers": [
            {
                "speaker_label": speaker_label,
                "top_match": detail["best_person"],
                "top_similarity": detail["best_score"],
                "margin": detail["margin"],
                "candidate_people": [name for name, _ in detail["ranked"]],
                "required_tiebreak": any(item["speaker_label"] == speaker_label for item in pending_tiebreaks),
            }
            for speaker_label, detail in sorted(similarity_details.items())
        ],
    }
    return validated_entries, backend_usage, llm_usage
