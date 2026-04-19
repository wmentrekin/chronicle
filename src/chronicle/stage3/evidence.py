"""Deterministic evidence extraction for Stage 3 speaker mapping."""

from __future__ import annotations

from typing import Any

from ..session import SessionManifest
from .schemas import DEFAULT_MAX_EVIDENCE_CHARS, DEFAULT_MAX_EVIDENCE_EXAMPLES_PER_SPEAKER
from .segmentation import classify_stage3_segment_type


def build_evidence_summary(
    *,
    manifest: SessionManifest,
    blocks: list[dict[str, Any]],
    speaker_labels: list[str],
    max_examples_per_speaker: int = DEFAULT_MAX_EVIDENCE_EXAMPLES_PER_SPEAKER,
    max_example_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
) -> tuple[dict[str, Any], list[str]]:
    summaries: dict[str, Any] = {}
    notes: list[str] = []

    for speaker_label in speaker_labels:
        speaker_blocks = [block for block in blocks if block.get("speaker_label") == speaker_label]
        examples: list[dict[str, Any]] = []
        question_count = 0
        response_count = 0
        speaking_seconds = 0.0

        for block in speaker_blocks:
            block_type = classify_stage3_segment_type(str(block.get("text") or ""))
            if block_type == "question":
                question_count += 1
            else:
                response_count += 1
            speaking_seconds += float(block.get("alignment", {}).get("overlap_seconds") or 0.0)

            if len(examples) >= max_examples_per_speaker:
                continue
            example_text = str(block.get("text") or "").strip()
            if not example_text:
                continue
            example_type = "question_like" if block_type == "question" else "response_like"
            examples.append(
                {
                    "example_id": f"{speaker_label}-example-{len(examples) + 1}",
                    "type": example_type,
                    "text": example_text[:max_example_chars],
                    "source_block_ids": [block.get("block_id")],
                    "source_segment_ids": list(block.get("source_segment_ids") or []),
                    "source_turn_ids": list(block.get("source_turn_ids") or []),
                    "notes": ["example truncated"] if len(example_text) > max_example_chars else [],
                }
            )

        if len(speaker_blocks) > max_examples_per_speaker:
            notes.append(
                f"Evidence examples for {speaker_label} sampled to {max_examples_per_speaker} block(s)."
            )

        summaries[speaker_label] = {
            "total_speaking_seconds": round(speaking_seconds, 3),
            "turn_count": len(speaker_blocks),
            "question_like_block_count": question_count,
            "response_like_block_count": response_count,
            "candidate_people": list(manifest.participants),
            "evidence_examples": examples,
            "notes": [],
        }

    return summaries, notes
