"""Validation CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from ..exceptions import SessionValidationError
from ..paths import DEFAULT_PARTICIPANTS_FILE
from ..session import render_validation_report, validate_session
from .common import console


def register(app: typer.Typer) -> None:
    @app.command("validate")
    def validate_command(
        session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
        participants_file: Path = typer.Option(
            DEFAULT_PARTICIPANTS_FILE,
            "--participants-file",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Canonical participants metadata file.",
        ),
        as_json: bool = typer.Option(False, "--json", help="Emit the validation report as JSON."),
        strict: bool = typer.Option(False, "--strict", help="Treat warnings as a non-zero exit."),
    ) -> None:
        """Validate a session bundle."""
        try:
            report = validate_session(session_id, participants_file)
        except SessionValidationError as exc:
            raise typer.Exit(code=1) from exc
        render_validation_report(report, console, as_json=as_json)
        if not report.ok or (strict and report.warnings):
            raise typer.Exit(code=1)
