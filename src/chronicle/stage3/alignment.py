"""Source-aware Stage 1 to Stage 2 alignment."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from ..session import SessionManifest
from ..utils import format_timestamp, normalize_text
from .inputs import normalize_source_audio, stage1_segment_seconds
from .schemas import (
    CONFIDENT_OVERLAP_RATIO,
    MATERIAL_OVERLAP_RATIO,
    MATERIAL_OVERLAP_SECONDS,
    MERGE_GAP_SECONDS,
    WEAK_TOTAL_OVERLAP_RATIO,
)


def overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _turn_seconds(turn: dict[str, Any], timing_basis: str) -> tuple[Optional[float], Optional[float]]:
    if timing_basis == "source_relative":
        return turn.get("source_start_seconds"), turn.get("source_end_seconds")
    return turn.get("session_start_seconds"), turn.get("session_end_seconds")


def _material_overlaps(
    segment_duration: float,
    overlaps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        item
        for item in overlaps
        if item["overlap_seconds"] >= MATERIAL_OVERLAP_SECONDS
        or item["overlap_seconds"] / segment_duration >= MATERIAL_OVERLAP_RATIO
    ]


def _make_block(
    *,
    block_id: int,
    segment: dict[str, Any],
    source_audio: str,
    start_seconds: Optional[float],
    end_seconds: Optional[float],
    text: str,
    speaker_label: Optional[str],
    speaker_label_candidates: list[str],
    source_turn_ids: list[int],
    overlap_total: float,
    segment_coverage_ratio: float,
    turn_coverage_ratio: float,
    alignment_confidence: str,
    timing_basis: str,
    notes: list[str],
) -> dict[str, Any]:
    segment_id = segment.get("segment_id", segment.get("source_segment_id"))
    return {
        "block_id": block_id,
        "speaker_label": speaker_label,
        "speaker_label_candidates": speaker_label_candidates,
        "speaker": None,
        "confidence": alignment_confidence,
        "candidate_people": [],
        "start_time": format_timestamp(start_seconds),
        "end_time": format_timestamp(end_seconds),
        "source_audio": source_audio,
        "text": text,
        "source_turn_ids": source_turn_ids,
        "source_segment_ids": [segment.get("source_segment_id", segment_id)],
        "source_stage1_segment_ids": [segment_id],
        "alignment": {
            "overlap_seconds": round(overlap_total, 3),
            "segment_coverage_ratio": round(segment_coverage_ratio, 4),
            "turn_coverage_ratio": round(turn_coverage_ratio, 4),
            "alignment_confidence": alignment_confidence,
            "timing_basis": timing_basis,
            "notes": notes,
        },
        "notes": list(notes),
    }


def align_transcript_to_diarization(
    *,
    manifest: SessionManifest,
    stage1_artifact: dict[str, Any],
    stage2_artifact: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    turns_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in stage2_artifact.get("turns", []):
        if not isinstance(turn, dict):
            continue
        source_audio = normalize_source_audio(turn.get("source_audio"), manifest)
        turns_by_source[source_audio].append(turn)

    blocks: list[dict[str, Any]] = []
    source_summaries: dict[str, dict[str, Any]] = {}
    ambiguous_count = 0
    needs_review_count = 0

    for segment in stage1_artifact.get("segments", []):
        if not isinstance(segment, dict):
            continue
        text = normalize_text(str(segment.get("text") or ""))
        if not text:
            continue

        source_audio = normalize_source_audio(segment.get("source_audio"), manifest)
        start_seconds, end_seconds, timing_basis = stage1_segment_seconds(segment)
        summary = source_summaries.setdefault(
            source_audio,
            {
                "source_audio": source_audio,
                "stage1_segment_count": 0,
                "stage2_turn_count": len(turns_by_source.get(source_audio, [])),
                "aligned_block_count": 0,
                "notes": [],
            },
        )
        summary["stage1_segment_count"] += 1

        if start_seconds is None or end_seconds is None or end_seconds <= start_seconds:
            notes = ["[source timing mismatch] Stage 1 segment has invalid timing."]
            block = _make_block(
                block_id=len(blocks) + 1,
                segment=segment,
                source_audio=source_audio,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                text=text,
                speaker_label=None,
                speaker_label_candidates=[],
                source_turn_ids=[],
                overlap_total=0.0,
                segment_coverage_ratio=0.0,
                turn_coverage_ratio=0.0,
                alignment_confidence="Needs review",
                timing_basis=timing_basis,
                notes=notes,
            )
            blocks.append(block)
            ambiguous_count += 1
            needs_review_count += 1
            continue

        segment_duration = end_seconds - start_seconds
        overlaps: list[dict[str, Any]] = []
        for turn in turns_by_source.get(source_audio, []):
            turn_start, turn_end = _turn_seconds(turn, timing_basis)
            if not isinstance(turn_start, (int, float)) or not isinstance(turn_end, (int, float)):
                continue
            overlap = overlap_seconds(start_seconds, end_seconds, float(turn_start), float(turn_end))
            if overlap <= 0:
                continue
            overlaps.append(
                {
                    "turn": turn,
                    "overlap_seconds": overlap,
                    "turn_duration": max(0.001, float(turn_end) - float(turn_start)),
                }
            )

        overlaps.sort(key=lambda item: item["overlap_seconds"], reverse=True)
        total_overlap = sum(item["overlap_seconds"] for item in overlaps)
        segment_coverage = total_overlap / segment_duration if segment_duration > 0 else 0.0
        material = _material_overlaps(segment_duration, overlaps)
        notes: list[str] = []
        speaker_label: Optional[str] = None
        candidates: list[str] = []
        turn_ids = [int(item["turn"].get("turn_id")) for item in overlaps if item["turn"].get("turn_id") is not None]
        turn_coverage = 0.0
        confidence = "Needs review"

        if not overlaps or segment_coverage < WEAK_TOTAL_OVERLAP_RATIO:
            notes.append("[alignment weak] Stage 1 segment has weak or no Stage 2 overlap.")
            candidates = sorted(
                {
                    str(item["turn"].get("speaker_label"))
                    for item in overlaps
                    if item["turn"].get("speaker_label")
                }
            )
        elif len(material) >= 2:
            top = overlaps[0]
            top_speaker = str(top["turn"].get("speaker_label"))
            top_ratio = top["overlap_seconds"] / segment_duration if segment_duration > 0 else 0.0
            if top_ratio >= 0.65:
                speaker_label = top_speaker
                candidates = [top_speaker]
                confidence = "Likely"
                turn_coverage = top["overlap_seconds"] / top["turn_duration"]
                notes.append(f"[dominant speaker assignment] Dominant turn `{top_speaker}` covers {top_ratio*100:.0f}% of segment.")
            else:
                notes.append("[multiple candidate speakers] Stage 1 segment crosses diarized speaker turns.")
                candidates = sorted(
                    {
                        str(item["turn"].get("speaker_label"))
                        for item in material
                        if item["turn"].get("speaker_label")
                    }
                )
                ambiguous_count += 1
        else:
            top = overlaps[0]
            speaker_label = str(top["turn"].get("speaker_label"))
            candidates = [speaker_label]
            turn_coverage = top["overlap_seconds"] / top["turn_duration"]
            if top["overlap_seconds"] / segment_duration >= CONFIDENT_OVERLAP_RATIO:
                confidence = "Confirmed"
            else:
                confidence = "Likely"

        if confidence == "Needs review":
            needs_review_count += 1

        if overlaps and turn_coverage == 0.0:
            turn_coverage = overlaps[0]["overlap_seconds"] / overlaps[0]["turn_duration"]

        block = _make_block(
            block_id=len(blocks) + 1,
            segment=segment,
            source_audio=source_audio,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            text=text,
            speaker_label=speaker_label,
            speaker_label_candidates=candidates,
            source_turn_ids=turn_ids,
            overlap_total=total_overlap,
            segment_coverage_ratio=segment_coverage,
            turn_coverage_ratio=turn_coverage,
            alignment_confidence=confidence,
            timing_basis=timing_basis,
            notes=notes,
        )
        blocks.append(block)
        summary["aligned_block_count"] += 1

    merged_blocks = merge_adjacent_blocks(blocks)
    alignment_summary = {
        "block_count": len(merged_blocks),
        "aligned_block_count": len([block for block in merged_blocks if block.get("speaker_label")]),
        "ambiguous_block_count": ambiguous_count,
        "needs_review_block_count": len(
            [
                block
                for block in merged_blocks
                if block.get("alignment", {}).get("alignment_confidence") == "Needs review"
            ]
        )
        or needs_review_count,
        "stage1_segment_count": len(stage1_artifact.get("segments") or []),
        "stage2_turn_count": len(stage2_artifact.get("turns") or []),
        "source_audio_files": list(source_summaries.values()),
        "notes": [],
    }
    return merged_blocks, alignment_summary


def merge_adjacent_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for block in blocks:
        if not merged:
            merged.append(block)
            continue
        previous = merged[-1]
        previous_end = _timestamp_to_seconds(previous.get("end_time"))
        current_start = _timestamp_to_seconds(block.get("start_time"))
        same_identity = (
            previous.get("speaker_label") == block.get("speaker_label")
            and previous.get("speaker_label") is not None
            and previous.get("source_audio") == block.get("source_audio")
            and previous.get("alignment", {}).get("alignment_confidence") == block.get("alignment", {}).get("alignment_confidence")
        )
        close_enough = (
            previous_end is not None
            and current_start is not None
            and 0 <= current_start - previous_end <= MERGE_GAP_SECONDS
        )
        if not same_identity or not close_enough:
            merged.append(block)
            continue
        previous["end_time"] = block.get("end_time")
        previous["text"] = normalize_text(f"{previous.get('text', '')} {block.get('text', '')}")
        previous["source_turn_ids"] = sorted(set(previous.get("source_turn_ids", [])) | set(block.get("source_turn_ids", [])))
        previous["source_segment_ids"].extend(block.get("source_segment_ids", []))
        previous["source_stage1_segment_ids"].extend(block.get("source_stage1_segment_ids", []))
        previous["alignment"]["overlap_seconds"] = round(
            float(previous["alignment"].get("overlap_seconds", 0.0))
            + float(block["alignment"].get("overlap_seconds", 0.0)),
            3,
        )
        previous["notes"] = sorted(set(previous.get("notes", [])) | set(block.get("notes", [])))
        previous["alignment"]["notes"] = sorted(
            set(previous.get("alignment", {}).get("notes", [])) | set(block.get("alignment", {}).get("notes", []))
        )

    for index, block in enumerate(merged, 1):
        block["block_id"] = index
    return merged


def _timestamp_to_seconds(value: object) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    try:
        hours, minutes, seconds_text = value.split(":")
        seconds, milliseconds = seconds_text.split(".")
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0
    except ValueError:
        return None
