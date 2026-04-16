"""Artifact helpers for Stage 2."""

from __future__ import annotations

from pathlib import Path
def stage2_output_paths(stage_dir: Path) -> tuple[Path, Path]:
    return (
        stage_dir / "diarization.json",
        stage_dir / "diarization.md",
    )
