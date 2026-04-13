"""Stage 2 semantic diarization orchestration."""

from __future__ import annotations

from pathlib import Path

from ..exceptions import StageExecutionError
from ..paths import repo_relative
from ..session import SessionManifest
from ..utils import write_json
from .artifacts import stage2_output_paths, write_stage2_markdown
from .assignment import assign_stage2_blocks, reconcile_stage2_question_targets
from .inputs import load_participant_records, load_stage1_segments, load_stage2_context_text
from .segmentation import build_stage2_candidate_blocks


def execute_stage2(
    manifest: SessionManifest,
    stage1_dir: Path,
    stage2_dir: Path,
    participants_file: Path,
    force: bool,
) -> tuple[list[str], list[str], list[str]]:
    json_path, markdown_path = stage2_output_paths(stage2_dir)
    if not force and json_path.exists() and markdown_path.exists():
        return [], [repo_relative(json_path), repo_relative(markdown_path)], [
            "Stage 2 artifacts already exist; rerun with `--force` to overwrite them."
        ]

    participants_by_name = load_participant_records(participants_file)
    context_text = load_stage2_context_text(manifest)
    stage1_segments, stage1_artifacts = load_stage1_segments(manifest, stage1_dir)
    candidate_blocks = build_stage2_candidate_blocks(stage1_segments)
    assigned_blocks, stage_notes = assign_stage2_blocks(
        manifest=manifest,
        candidate_blocks=candidate_blocks,
        participants_by_name=participants_by_name,
        context_text=context_text,
    )
    reconcile_stage2_question_targets(assigned_blocks, manifest.primary_interviewees)

    artifact_payload = {
        "stage": "stage2_semantic_diarization",
        "session_id": manifest.session_id,
        "participants_file": repo_relative(participants_file),
        "source_stage1_artifacts": stage1_artifacts,
        "block_count": len(assigned_blocks),
        "blocks": assigned_blocks,
        "notes": stage_notes,
    }
    write_json(json_path, artifact_payload)
    write_stage2_markdown(markdown_path, manifest, artifact_payload)
    return [repo_relative(json_path), repo_relative(markdown_path)], [], stage_notes
