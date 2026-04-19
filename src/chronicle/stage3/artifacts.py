"""Artifact helpers for Stage 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import write_json


def stage3_output_paths(stage_dir: Path) -> tuple[Path, Path]:
    return (
        stage_dir / "identified_conversation.json",
        stage_dir / "identified_conversation.md",
    )


def stage3_align_only_output_paths(stage_dir: Path) -> tuple[Path, Path]:
    return (
        stage_dir / "aligned_transcript.json",
        stage_dir / "aligned_transcript.md",
    )


def output_paths_for_mode(stage_dir: Path, mode: str) -> tuple[Path, Path]:
    if mode == "align-only":
        return stage3_align_only_output_paths(stage_dir)
    return stage3_output_paths(stage_dir)


def write_stage3_artifacts(*, stage_dir: Path, mode: str, artifact: dict[str, Any]) -> list[Path]:
    json_path, markdown_path = output_paths_for_mode(stage_dir, mode)
    write_json(json_path, artifact)
    markdown_path.write_text(render_stage3_markdown(artifact), encoding="utf-8")
    return [json_path, markdown_path]


def render_stage3_markdown(artifact: dict[str, Any]) -> str:
    align_only = artifact.get("mode") == "align-only"
    title = "Aligned Transcript" if align_only else "Identified Conversation"
    lines = [
        f"# {title}",
        "",
        f"- **Session ID:** {artifact['session_id']}",
        f"- **Mode:** {artifact['mode']}",
        f"- **Source Stage 1:** {artifact['source_stage1_artifact']}",
        f"- **Source Stage 2:** {artifact['source_stage2_artifact']}",
        "",
    ]

    if not align_only:
        lines.extend(["## Speaker Map", ""])
        for entry in artifact.get("speaker_map", []):
            lines.append(
                f"- `{entry['speaker_label']}`: {entry['assigned_person']} `[{entry['confidence']}]`"
            )
        lines.append("")

    lines.extend(["## Transcript", ""])
    current_audio: str | None = None
    for block in artifact.get("blocks", []):
        source_audio = block.get("source_audio")
        if source_audio != current_audio:
            current_audio = source_audio
            lines.extend([f"### {current_audio}", ""])

        speaker_label = block.get("speaker_label")
        candidates = block.get("speaker_label_candidates") or []
        label_text = speaker_label or " / ".join(candidates) or "UNKNOWN_ANONYMOUS"
        timestamp = ""
        if block.get("start_time") and block.get("end_time"):
            timestamp = f"`[{block['start_time']} - {block['end_time']}]`"

        if align_only:
            speaker_text = "Anonymous speaker"
            confidence = block.get("alignment", {}).get("alignment_confidence", block.get("confidence"))
            lines.append(f"**{speaker_text}** `[{confidence} alignment]` `{label_text}` {timestamp}".rstrip())
        else:
            speaker_text = block.get("speaker") or "Needs review"
            lines.append(
                f"**{speaker_text}** `[{block.get('confidence', 'Needs review')}]` `{label_text}` {timestamp}".rstrip()
            )

        lines.append(block.get("text") or "[inaudible]")
        if block.get("candidate_people"):
            lines.append("")
            lines.append("Candidate people: " + ", ".join(block["candidate_people"]))
        if block.get("notes"):
            lines.append("")
            lines.append("Notes: " + "; ".join(block["notes"]))
        lines.append("")

    if artifact.get("notes"):
        lines.extend(["## Notes", ""])
        for note in artifact["notes"]:
            lines.append(f"- {note}")

    return "\n".join(lines).rstrip() + "\n"
