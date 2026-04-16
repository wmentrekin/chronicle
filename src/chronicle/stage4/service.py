"""Stage 4 planning helpers."""

from __future__ import annotations

from pathlib import Path

from ..session import SessionManifest
from ..utils import slugify_stem


def planned_stage4_artifacts(manifest: SessionManifest, stage_dir: Path) -> list[Path]:
    return [
        stage_dir / f"{slugify_stem(Path(name))}.organized.json"
        for name in manifest.primary_interviewees
    ]
