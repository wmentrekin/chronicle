"""Stage 3 and status CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from ..exceptions import SessionValidationError
from ..paths import DEFAULT_PARTICIPANTS_FILE, ensure_output_dirs
from ..session import require_valid_session
from .common import console, render_stage_plan


def register(app: typer.Typer) -> None:
    @app.command("chronology")
    def chronology_command(
        session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
        participants_file: Path = typer.Option(
            DEFAULT_PARTICIPANTS_FILE,
            "--participants-file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ) -> None:
        """Validate inputs and prepare Stage 3 output locations."""
        try:
            manifest = require_valid_session(session_id, console, participants_file)
        except SessionValidationError as exc:
            raise typer.Exit(code=1) from exc

        directories = ensure_output_dirs(manifest.session_id)
        render_stage_plan(manifest, "stage3", directories)
        console.print(
            Panel(
                "Stage 3 validates the session and prepares planned artifact paths, but the implementation is not wired yet.",
                title="Skeleton Only",
            )
        )

    @app.command("run")
    def run_command(
        session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
        participants_file: Path = typer.Option(
            DEFAULT_PARTICIPANTS_FILE,
            "--participants-file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ) -> None:
        """Validate a session and show the current pipeline status."""
        try:
            manifest = require_valid_session(session_id, console, participants_file)
        except SessionValidationError as exc:
            raise typer.Exit(code=1) from exc

        directories = ensure_output_dirs(manifest.session_id)
        table = Table(title="Pipeline Status")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Output dir", style="white")
        table.add_row("transcribe", "implemented", directories["stage1"].as_posix())
        table.add_row("diarize", "implemented", directories["stage2"].as_posix())
        table.add_row("chronology", "validated, not implemented", directories["stage3"].as_posix())
        console.print(table)
        console.print(
            Panel(
                (
                    "Current CLI commands are `validate`, `transcribe`, `diarize`, `chronology`, and `run`. "
                    "Session input is resolved from `inputs/sessions/<session_id>/`."
                ),
                title="Current Workflow",
            )
        )
