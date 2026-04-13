"""Stage 3 planning helpers."""

from __future__ import annotations

from pathlib import Path

from ..session import SessionManifest
from ..utils import slugify_stem


def planned_stage3_artifacts(manifest: SessionManifest, stage_dir: Path) -> list[Path]:
    return [
        stage_dir / f"{slugify_stem(Path(name))}.chronology.json"
        for name in manifest.primary_interviewees
    ]

