"""Artifact helpers for Stage 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import format_timestamp, write_json


def stage2_output_paths(stage_dir: Path) -> tuple[Path, Path]:
    return (
        stage_dir / "diarization.json",
        stage_dir / "diarization.md",
    )


def write_stage2_artifacts(
    *,
    stage_dir: Path,
    artifact: dict[str, Any],
) -> list[Path]:
    json_path, markdown_path = stage2_output_paths(stage_dir)
    write_json(json_path, artifact)
    markdown_path.write_text(render_stage2_markdown(artifact), encoding="utf-8")
    return [json_path, markdown_path]


def render_stage2_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# Anonymous Audio Diarization",
        "",
        f"- **Session ID:** {artifact['session_id']}",
        f"- **Backend:** {artifact['backend']}",
        f"- **Audio files:** {len(artifact['audio_files'])}",
        f"- **Speaker labels:** {', '.join(artifact['speaker_labels']) if artifact['speaker_labels'] else 'none'}",
        f"- **Turn count:** {len(artifact['turns'])}",
        "",
        "## Turns",
        "",
    ]

    for turn in artifact["turns"]:
        session_start = format_timestamp(turn["session_start_seconds"]) or "00:00:00.000"
        session_end = format_timestamp(turn["session_end_seconds"]) or "00:00:00.000"
        source_start = format_timestamp(turn["source_start_seconds"]) or "00:00:00.000"
        source_end = format_timestamp(turn["source_end_seconds"]) or "00:00:00.000"
        lines.append(
            (
                f"- [{session_start} - {session_end}] `{turn['speaker_label']}` "
                f"({turn['source_audio']} @ {source_start} - {source_end})"
            )
        )

    lines.append("")
    return "\n".join(lines)
