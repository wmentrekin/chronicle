"""Stage 2 CLI commands."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.panel import Panel

from ..exceptions import SessionValidationError, StageExecutionError
from ..paths import DEFAULT_PARTICIPANTS_FILE, ensure_output_dirs, repo_relative, session_manifest_path
from ..session import require_valid_session, resolve_audio_path, resolve_context_path
from ..stage2.service import execute_stage2
from ..utils import write_run_metadata
from .common import confirm_overwrite_if_needed, console, render_stage_plan


def register(app: typer.Typer) -> None:
    @app.command("diarize")
    def diarize_command(
        session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
        participants_file: Path = typer.Option(
            DEFAULT_PARTICIPANTS_FILE,
            "--participants-file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
        force: bool = typer.Option(False, "--force", help="Overwrite existing stage 2 artifacts."),
    ) -> None:
        """Run Stage 2 semantic diarization."""
        try:
            manifest = require_valid_session(session_id, console, participants_file)
        except SessionValidationError as exc:
            raise typer.Exit(code=1) from exc

        directories = ensure_output_dirs(manifest.session_id)
        render_stage_plan(manifest, "stage2", directories)
        force = confirm_overwrite_if_needed(
            stage_label="Stage 2",
            force=force,
            output_paths=[
                directories["stage2"] / "diarized_conversation.json",
                directories["stage2"] / "diarized_conversation.md",
            ],
        )

        started_at = datetime.now(timezone.utc)
        run_notes: list[str] = []
        output_paths: list[str] = []
        status = "failed"
        try:
            output_paths, skipped_paths, stage_notes = execute_stage2(
                manifest=manifest,
                stage1_dir=directories["stage1"],
                stage2_dir=directories["stage2"],
                participants_file=participants_file,
                force=force,
            )
            run_notes.extend(stage_notes)
            if skipped_paths and not output_paths:
                status = "partial"
                console.print(
                    Panel(
                        "Stage 2 artifacts already exist. Use `--force` to regenerate them.",
                        title="Stage 2 Skip",
                    )
                )
            else:
                status = "success"
                console.print(
                    Panel(
                        f"Stage 2 wrote {len(output_paths)} artifact file(s) for session `{manifest.session_id}`.",
                        title="Stage 2 Complete",
                    )
                )
        except KeyboardInterrupt as exc:
            run_notes.append("Stage 2 run was interrupted before artifacts were written.")
            console.print(Panel("Stage 2 run interrupted.", title="Stage 2 Interrupted", style="yellow"))
            raise typer.Exit(code=130) from exc
        except StageExecutionError as exc:
            run_notes.append(str(exc))
            console.print(Panel(str(exc), title="Stage 2 Failed", style="red"))
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            run_notes.append(f"Unhandled stage 2 error: {exc}")
            console.print(Panel(str(exc), title="Stage 2 Failed", style="red"))
            raise typer.Exit(code=1) from exc
        finally:
            run_path = write_run_metadata(
                runs_dir=directories["runs"],
                stage_name="stage2",
                status=status,
                input_paths=[repo_relative(Path(manifest.manifest_path or session_manifest_path(manifest.session_id)))]
                + [repo_relative(resolve_context_path(manifest))]
                + [repo_relative(resolve_audio_path(manifest, audio_file)) for audio_file in manifest.audio_files],
                output_paths=output_paths,
                config={
                    "force": force,
                    "participants_file": repo_relative(participants_file),
                },
                notes=run_notes,
                started_at=started_at,
            )
            console.print(f"Run metadata: {run_path.as_posix()}")
