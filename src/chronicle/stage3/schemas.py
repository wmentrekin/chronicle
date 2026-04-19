"""Schema constants and helpers for Stage 3 artifacts."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "stage3-speaker-map-v1"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_INPUT_TOKENS = 12_000
DEFAULT_MAX_OUTPUT_TOKENS = 1_200
DEFAULT_MAX_EVIDENCE_EXAMPLES_PER_SPEAKER = 12
DEFAULT_MAX_EVIDENCE_CHARS = 500

MODES = {"llm", "manual", "align-only"}
CONFIDENCE_LABELS = {"Confirmed", "Likely", "Unclear", "Needs review"}
IDENTITY_SOURCES = {"manual", "llm", "deterministic"}

MATERIAL_OVERLAP_RATIO = 0.20
MATERIAL_OVERLAP_SECONDS = 1.0
CONFIDENT_OVERLAP_RATIO = 0.80
WEAK_TOTAL_OVERLAP_RATIO = 0.50
MERGE_GAP_SECONDS = 2.0


def empty_llm_usage(model: str, notes: list[str] | None = None) -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost_usd": None,
        "truncation_or_sampling_notes": list(notes or []),
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
