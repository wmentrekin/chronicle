"""Shared CLI helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..exceptions import StageExecutionError
from ..paths import repo_relative, session_manifest_path
from ..session import SessionManifest, resolve_audio_path, resolve_context_path
from ..stage4.service import planned_stage4_artifacts


console = Console()


def confirm_overwrite_if_needed(
    *,
    stage_label: str,
    force: bool,
    output_paths: list[Path],
) -> bool:
    existing_paths = [path for path in output_paths if path.exists()]
    if force or not existing_paths:
        return force

    if not sys.stdin.isatty():
        console.print(
            Panel(
                (
                    f"{stage_label} artifacts already exist and this shell is non-interactive. "
                    "Skipping overwrite. Rerun with `--force` to overwrite them explicitly."
                ),
                title=f"{stage_label} Skip",
            )
        )
        return False

    overwrite = typer.confirm(
        (
            f"{stage_label} artifacts already exist for this session.\n"
            f"Overwrite {len(existing_paths)} existing file(s)?"
        ),
        default=False,
    )
    if not overwrite:
        console.print(
            Panel(
                "Keeping existing artifacts unchanged.",
                title=f"{stage_label} Skip",
            )
        )
    return overwrite


def install_chronicle_link(link_path: Path) -> Path:
    chronicle_executable = Path(sys.executable).resolve().parent / "chronicle"
    if not chronicle_executable.exists():
        raise StageExecutionError(
            f"Could not find the Chronicle executable beside the current Python interpreter at {chronicle_executable}."
        )
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(chronicle_executable)
    return link_path


def render_stage_plan(
    manifest: SessionManifest, stage_name: str, directories: dict[str, Path]
) -> None:
    table = Table(title=f"{stage_name.upper()} Plan")
    table.add_column("Item", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Session ID", manifest.session_id)
    table.add_row("Manifest", repo_relative(Path(manifest.manifest_path or session_manifest_path(manifest.session_id))))
    table.add_row("Context doc", repo_relative(resolve_context_path(manifest)))
    table.add_row(
        "Audio files",
        ", ".join(repo_relative(resolve_audio_path(manifest, audio_file)) for audio_file in manifest.audio_files),
    )
    table.add_row("Runs dir", directories["runs"].as_posix())
    table.add_row("Stage dir", directories[stage_name].as_posix())
    if stage_name == "stage1":
        table.add_row(
            "Planned artifact",
            (directories["stage1"] / "raw_transcript.json").as_posix(),
        )
    elif stage_name == "stage2":
        table.add_row(
            "Planned artifact",
            (directories["stage2"] / "diarization.json").as_posix(),
        )
    elif stage_name == "stage3":
        table.add_row(
            "Planned artifact",
            (directories["stage3"] / "identified_conversation.json").as_posix(),
        )
    elif stage_name == "stage4":
        table.add_row(
            "Planned artifacts",
            ", ".join(path.name for path in planned_stage4_artifacts(manifest, directories["stage4"])),
        )
    console.print(table)
