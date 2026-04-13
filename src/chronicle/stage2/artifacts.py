"""Artifact helpers for Stage 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..session import SessionManifest


def stage2_output_paths(stage_dir: Path) -> tuple[Path, Path]:
    return (
        stage_dir / "diarized_conversation.json",
        stage_dir / "diarized_conversation.md",
    )


def write_stage2_markdown(
    path: Path,
    manifest: SessionManifest,
    artifact: dict[str, Any],
) -> None:
    lines = [
        "# Diarized Conversation",
        "",
        f"- **Session ID:** {manifest.session_id}",
        f"- **Stage:** {artifact['stage']}",
        "",
        "## Transcript",
        "",
    ]

    current_audio: Optional[str] = None
    for block in artifact["blocks"]:
        if block["source_audio"] != current_audio:
            current_audio = block["source_audio"]
            lines.extend([f"### {current_audio}", ""])

        timestamp = ""
        if block.get("start_time") and block.get("end_time"):
            timestamp = f" [{block['start_time']} - {block['end_time']}]"
        lines.append(f"**{block['speaker']}** `[{block['confidence']}]`{timestamp}")
        lines.append(block["text"] or "[inaudible]")
        lines.append("")
        if block.get("candidate_speakers"):
            lines.append("Candidate speakers: " + ", ".join(block["candidate_speakers"]))
            lines.append("")
        if block.get("notes"):
            lines.append("Notes:")
            for note in block["notes"]:
                lines.append(f"- {note}")
            lines.append("")

    if artifact.get("notes"):
        lines.extend(["## Notes", ""])
        for note in artifact["notes"]:
            lines.append(f"- {note}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
