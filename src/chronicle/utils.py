"""Shared utility helpers."""

from __future__ import annotations

import os
import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .paths import REPO_ROOT


def load_local_env(env_path: Path | None = None) -> None:
    path = env_path or (REPO_ROOT / ".env")
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


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


def estimate_gcp_cost(
    wall_seconds: float,
    *,
    gpu_enabled: bool = False,
    gpu_type: str = "nvidia-l4",
    machine_type: str = "e2-standard-8",
) -> float:
    """Estimate GCP Compute Engine cost in USD for a given wall-clock execution duration."""
    hourly_rate = 0.26
    if gpu_enabled:
        if "t4" in gpu_type.lower():
            hourly_rate = 0.38
        else:
            hourly_rate = 0.70
    elif "g2" in machine_type.lower():
        hourly_rate = 0.70
    elif "n1" in machine_type.lower():
        hourly_rate = 0.38

    total_hours = wall_seconds / 3600.0
    return round(total_hours * hourly_rate, 4)
