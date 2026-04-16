"""Stage 4 CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel

from ..exceptions import SessionValidationError
from ..paths import DEFAULT_PARTICIPANTS_FILE, ensure_output_dirs
from ..session import require_valid_session
from .common import console, render_stage_plan


def register(app: typer.Typer) -> None:
    @app.command("organize")
    def organize_command(
        session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
        participants_file: Path = typer.Option(
            DEFAULT_PARTICIPANTS_FILE,
            "--participants-file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ) -> None:
        """Validate inputs and prepare Stage 4 organization output locations."""
        try:
            manifest = require_valid_session(session_id, console, participants_file)
        except SessionValidationError as exc:
            raise typer.Exit(code=1) from exc

        directories = ensure_output_dirs(manifest.session_id)
        render_stage_plan(manifest, "stage4", directories)
        console.print(
            Panel(
                "Stage 4 now means verbatim organization. The command surface is aligned, but the implementation is not wired yet.",
                title="Skeleton Only",
            )
        )
