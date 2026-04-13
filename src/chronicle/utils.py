"""Shared utility helpers."""

from __future__ import annotations

import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-") or "audio"


def format_timestamp(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_run_metadata(
    runs_dir: Path,
    stage_name: str,
    status: str,
    input_paths: list[str],
    output_paths: list[str],
    config: dict[str, Any],
    notes: list[str],
    started_at: datetime,
) -> Path:
    finished_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    run_path = runs_dir / f"{stage_name}.{run_id}.json"
    payload = {
        "run_id": run_id,
        "stage": stage_name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "status": status,
        "input_paths": input_paths,
        "output_paths": output_paths,
        "config": config,
        "notes": notes,
    }
    write_json(run_path, payload)
    return run_path

