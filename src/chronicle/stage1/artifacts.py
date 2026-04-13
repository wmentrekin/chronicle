"""Artifact helpers for Stage 1."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from ..exceptions import StageExecutionError
from ..paths import repo_relative
from ..session import SessionManifest
from ..utils import format_timestamp


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    return f"{value:.1f}{units[unit_index]}"


def format_duration_summary(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def build_stage1_audio_summary(audio_probes: list[Any]) -> str:
    total_duration = sum(probe.duration_seconds for probe in audio_probes)
    total_size = sum(probe.file_size_bytes for probe in audio_probes)
    lines = [
        f"- Audio files: {len(audio_probes)}",
        f"- Total duration: {format_duration_summary(total_duration)}",
        f"- Total size: {format_bytes(total_size)}",
    ]
    for probe in audio_probes:
        lines.append(
            f"- {probe.source_audio}: {format_duration_summary(probe.duration_seconds)} | {format_bytes(probe.file_size_bytes)}"
        )
    return "\n".join(lines)


def parse_timestamp_seconds(value: Optional[str]) -> Optional[float]:
    if not value or not isinstance(value, str):
        return None
    try:
        hours_text, minutes_text, seconds_text = value.split(":")
        seconds_part, millis_part = seconds_text.split(".")
        return (
            int(hours_text) * 3600
            + int(minutes_text) * 60
            + int(seconds_part)
            + int(millis_part) / 1000.0
        )
    except (ValueError, AttributeError):
        return None


def write_stage1_markdown(
    path: Path,
    manifest: SessionManifest,
    artifact: dict[str, Any],
) -> None:
    source_audio_files = artifact.get("source_audio_files") or []
    lines = [
        "# Raw Transcript",
        "",
        f"- **Session ID:** {manifest.session_id}",
        f"- **Source audio files:** {len(source_audio_files)}",
        f"- **Language:** {artifact['language']}",
        f"- **Model family:** {artifact['model']['family']}",
        f"- **Model name:** {artifact['model']['name']}",
        f"- **Runtime:** {artifact['model']['runtime']}",
        f"- **Device:** {artifact['model']['device']}",
        "",
        "## Transcript",
        "",
    ]

    current_audio: Optional[str] = None
    for segment in artifact["segments"]:
        source_audio = segment.get("source_audio")
        if source_audio != current_audio:
            current_audio = source_audio
            if current_audio:
                lines.append(f"### {current_audio}")
                lines.append("")
        start = segment.get("start")
        decode_status = segment.get("decode_status")
        text = segment.get("text") or "[inaudible]"
        if decode_status == "unk":
            text = "`<unk>` [needs review]"
        elif decode_status == "empty":
            text = "[needs review: empty transcription chunk]"
        if start:
            lines.append(f"[{start}] {text}")
        else:
            lines.append(text)
        lines.append("")

    lines.extend(["## Notes", ""])
    lines.append("- Raw transcription only. No speaker assignment has been performed.")
    if artifact.get("notes"):
        for note in artifact["notes"]:
            lines.append(f"- {note}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def session_stage1_output_paths(stage_dir: Path) -> tuple[Path, Path]:
    return (
        stage_dir / "raw_transcript.json",
        stage_dir / "raw_transcript.md",
    )


def legacy_stage1_output_paths(stage_dir: Path, audio_file: str) -> tuple[Path, Path]:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(audio_file).stem).strip("-") or "audio"
    return (
        stage_dir / f"{stem}.raw_transcript.json",
        stage_dir / f"{stem}.raw_transcript.md",
    )


def build_session_stage1_artifact(
    manifest: SessionManifest,
    audio_artifacts: list[dict[str, Any]],
    model_info: dict[str, Any],
    notes: list[str],
) -> dict[str, Any]:
    combined_segments: list[dict[str, Any]] = []
    combined_word_timestamps: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    audio_summaries: list[dict[str, Any]] = []
    offset_seconds = 0.0

    for audio_artifact in audio_artifacts:
        source_audio = str(audio_artifact.get("source_audio", "")).strip()
        segments = audio_artifact.get("segments") or []
        transcript_text = str(audio_artifact.get("transcript_text", "")).strip()
        if transcript_text:
            transcript_parts.append(transcript_text)

        max_local_end = 0.0
        first_global_start: Optional[str] = None
        last_global_end: Optional[str] = None
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            local_start_seconds = parse_timestamp_seconds(segment.get("start"))
            local_end_seconds = parse_timestamp_seconds(segment.get("end"))
            global_start = (
                format_timestamp(offset_seconds + local_start_seconds)
                if local_start_seconds is not None
                else None
            )
            global_end = (
                format_timestamp(offset_seconds + local_end_seconds)
                if local_end_seconds is not None
                else None
            )
            if first_global_start is None and global_start is not None:
                first_global_start = global_start
            if global_end is not None:
                last_global_end = global_end
            if local_end_seconds is not None:
                max_local_end = max(max_local_end, local_end_seconds)
            elif local_start_seconds is not None:
                max_local_end = max(max_local_end, local_start_seconds)

            combined_segments.append(
                {
                    "segment_id": len(combined_segments) + 1,
                    "start": global_start,
                    "end": global_end,
                    "text": segment.get("text"),
                    "decode_status": segment.get("decode_status"),
                    "avg_logprob": segment.get("avg_logprob"),
                    "no_speech_prob": segment.get("no_speech_prob"),
                    "compression_ratio": segment.get("compression_ratio"),
                    "source_audio": source_audio,
                    "source_segment_id": segment.get("segment_id"),
                    "source_start": segment.get("start"),
                    "source_end": segment.get("end"),
                }
            )

        audio_summaries.append(
            {
                "source_audio": source_audio,
                "segment_count": len([segment for segment in segments if isinstance(segment, dict)]),
                "offset_start": format_timestamp(offset_seconds),
                "offset_end": format_timestamp(offset_seconds + max_local_end),
                "transcript_char_count": len(transcript_text),
                "global_first_segment_start": first_global_start,
                "global_last_segment_end": last_global_end,
            }
        )
        offset_seconds += max_local_end

        for word_timestamp in audio_artifact.get("word_timestamps") or []:
            if not isinstance(word_timestamp, dict):
                continue
            combined_word_timestamps.append(word_timestamp)

    artifact_notes = list(notes)
    artifact_notes.append(
        "Session-level timestamps are sequential across source audio files and are offset using each file's local segment timings."
    )

    return {
        "stage": "stage1_transcription",
        "session_id": manifest.session_id,
        "source_audio_files": [summary["source_audio"] for summary in audio_summaries],
        "audio_files": audio_summaries,
        "normalized_audio": None,
        "model": model_info,
        "language": manifest.language,
        "transcript_text": "\n".join(part for part in transcript_parts if part).strip(),
        "segments": combined_segments,
        "word_timestamps": combined_word_timestamps,
        "notes": artifact_notes,
    }


def load_existing_audio_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StageExecutionError(f"Invalid Stage 1 artifact format: {repo_relative(path)}")
    return payload
