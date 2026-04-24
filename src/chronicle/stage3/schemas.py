"""Schema constants and helpers for Stage 3 artifacts."""

from __future__ import annotations

from typing import Any

from ..exceptions import StageExecutionError


SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "stage3-speaker-map-v1"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_BACKEND = "ollama_decomposed"
DEFAULT_MAX_INPUT_TOKENS = 12_000
DEFAULT_MAX_OUTPUT_TOKENS = 600
DEFAULT_MAX_EVIDENCE_EXAMPLES_PER_SPEAKER = 6
DEFAULT_MAX_EVIDENCE_CHARS = 300
DEFAULT_REFMATCH_MIN_SIMILARITY = 0.60
DEFAULT_REFMATCH_MIN_MARGIN = 0.05
DEFAULT_REFMATCH_CONFIRMED_SIMILARITY = 0.80
DEFAULT_REFMATCH_CONFIRMED_MARGIN = 0.15
DEFAULT_HYBRID_MAX_CANDIDATES = 2

MODES = {"llm", "manual", "align-only"}
AUTOMATIC_MODES = {"llm"}
BACKENDS = {"ollama_decomposed", "speechbrain_refmatch", "speechbrain_hybrid"}
CONFIDENCE_LABELS = {"Confirmed", "Likely", "Unclear", "Needs review"}
IDENTITY_SOURCES = {"manual", "llm", "deterministic"}

MATERIAL_OVERLAP_RATIO = 0.20
MATERIAL_OVERLAP_SECONDS = 1.0
CONFIDENT_OVERLAP_RATIO = 0.80
WEAK_TOTAL_OVERLAP_RATIO = 0.50
MERGE_GAP_SECONDS = 2.0


def resolve_stage3_backend(backend: str | None) -> str:
    resolved = (backend or DEFAULT_BACKEND).strip()
    if resolved in BACKENDS:
        return resolved
    raise StageExecutionError(
        f"Unsupported Stage 3 backend `{resolved}`. Expected one of: {', '.join(sorted(BACKENDS))}"
    )


def backend_uses_llm(backend: str) -> bool:
    return backend in {"ollama_decomposed", "speechbrain_hybrid"}


def mode_uses_automatic_backend(mode: str) -> bool:
    return mode in AUTOMATIC_MODES


def empty_llm_usage(model: str, notes: list[str] | None = None) -> dict[str, Any]:
    return {
        "backend": "ollama_decomposed",
        "provider": "ollama",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost_usd": None,
        "truncation_or_sampling_notes": list(notes or []),
    }


def build_ollama_backend_usage(
    *,
    model: str,
    llm_usage: dict[str, Any],
    manual_entries: list[dict[str, Any]],
    evidence_summary: dict[str, Any],
    llm_entry_count: int,
) -> dict[str, Any]:
    manual_labels = {str(entry.get("speaker_label", "")) for entry in manual_entries}
    unresolved_speakers = sorted(
        str(label) for label in evidence_summary if str(label) and str(label) not in manual_labels
    )
    return {
        "backend": "ollama_decomposed",
        "provider": "ollama",
        "model": model,
        "workflow": "no-reference-baseline",
        "reference_mode": "none",
        "prompt_version": llm_usage.get("prompt_version"),
        "manual_assignment_count": len(manual_entries),
        "unresolved_speaker_count": len(unresolved_speakers),
        "llm_assignment_count": llm_entry_count,
        "unresolved_speakers": unresolved_speakers,
        "token_usage": {
            "input_tokens": llm_usage.get("input_tokens"),
            "output_tokens": llm_usage.get("output_tokens"),
        },
    }


def make_speaker_map_entry(
    *,
    speaker_label: str,
    assigned_person: str,
    confidence: str,
    candidate_people: list[str],
    source: str,
    evidence: list[dict[str, Any]] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "speaker_label": speaker_label,
        "assigned_person": assigned_person,
        "confidence": confidence,
        "candidate_people": candidate_people,
        "source": source,
        "evidence": list(evidence or []),
        "notes": list(notes or []),
    }
