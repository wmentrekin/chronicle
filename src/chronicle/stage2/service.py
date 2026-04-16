"""Stage 2 anonymous audio diarization planning helpers."""

from __future__ import annotations

from pathlib import Path

from .artifacts import stage2_output_paths


def planned_stage2_artifacts(stage_dir: Path) -> list[Path]:
    return list(stage2_output_paths(stage_dir))
