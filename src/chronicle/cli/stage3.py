"""Stage 3 and status CLI commands."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from ..exceptions import SessionValidationError, StageExecutionError
from ..paths import DEFAULT_PARTICIPANTS_FILE, ensure_output_dirs, repo_relative, session_manifest_path
from ..session import require_valid_session, resolve_context_path
from ..stage3.service import execute_stage3
from ..stage3.schemas import DEFAULT_BACKEND
from ..utils import write_run_metadata
from .common import confirm_overwrite_if_needed, console, render_stage_plan


def register(app: typer.Typer) -> None:
    @app.command("identify")
    def identify_command(
        session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
        participants_file: Path = typer.Option(
            DEFAULT_PARTICIPANTS_FILE,
            "--participants-file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
        force: bool = typer.Option(False, "--force", help="Overwrite existing stage 3 artifacts."),
        mode: str = typer.Option("llm", "--mode", help="Stage 3 mode: llm, manual, or align-only."),
        model: Optional[str] = typer.Option(
            None,
            "--model",
            help="Ollama model for llm mode. Overrides CHRONICLE_STAGE3_MODEL.",
        ),
        backend: Optional[str] = typer.Option(
            None,
            "--backend",
            help=(
                "Automatic Stage 3 backend for identity assignment. "
                f"Defaults to `{DEFAULT_BACKEND}` and applies only to automatic modes."
            ),
        ),
        speaker_map: Optional[Path] = typer.Option(
            None,
            "--speaker-map",
            exists=False,
            dir_okay=False,
            readable=True,
            help="Manual speaker-map YAML for manual mode or llm overrides.",
        ),
    ) -> None:
        """Run Stage 3 speaker identification over Stage 1 and Stage 2 outputs."""
        try:
            manifest = require_valid_session(session_id, console, participants_file)
        except SessionValidationError as exc:
            raise typer.Exit(code=1) from exc

        directories = ensure_output_dirs(manifest.session_id)
        render_stage_plan(manifest, "stage3", directories)
        output_names = (
            ("aligned_transcript.json", "aligned_transcript.md")
            if mode == "align-only"
            else ("identified_conversation.json", "identified_conversation.md")
        )
        force = confirm_overwrite_if_needed(
            stage_label="Stage 3",
            force=force,
            output_paths=[
                directories["stage3"] / output_names[0],
                directories["stage3"] / output_names[1],
            ],
        )

        started_at = datetime.now(timezone.utc)
        run_notes: list[str] = []
        output_paths: list[str] = []
        stage_metadata: dict[str, object] = {}
        status = "failed"
        try:
            if mode == "llm" and (backend is None or backend == DEFAULT_BACKEND):
                console.print(
                    Panel(
                        (
                            "Stage 3 LLM mode uses local Ollama. Transcript excerpts, evidence summaries, "
                            "session context, and participant metadata stay on this machine."
                        ),
                        title="Stage 3 Privacy Notice",
                    )
                )
            console.print("Loading Stage 1 and Stage 2 artifacts...")
            output_paths, skipped_paths, stage_notes, stage_metadata = execute_stage3(
                manifest=manifest,
                stage1_dir=directories["stage1"],
                stage2_dir=directories["stage2"],
                stage3_dir=directories["stage3"],
                participants_file=participants_file,
                force=force,
                mode=mode,
                model=model,
                backend=backend,
                speaker_map_path=speaker_map,
            )
            run_notes.extend(stage_notes)
            if skipped_paths and not output_paths:
                status = "partial"
                console.print(
                    Panel(
                        "Stage 3 artifacts already exist. Use `--force` to regenerate them.",
                        title="Stage 3 Skip",
                    )
                )
            else:
                status = "success"
                console.print(
                    Panel(
                        f"Stage 3 wrote {len(output_paths)} artifact file(s) for session `{manifest.session_id}`.",
                        title="Stage 3 Complete",
                    )
                )
        except KeyboardInterrupt as exc:
            run_notes.append("Stage 3 run was interrupted before artifacts were written.")
            console.print(Panel("Stage 3 run interrupted.", title="Stage 3 Interrupted", style="yellow"))
            raise typer.Exit(code=130) from exc
        except StageExecutionError as exc:
            run_notes.append(str(exc))
            console.print(Panel(str(exc), title="Stage 3 Failed", style="red"))
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            run_notes.append(f"Unhandled stage 3 error: {exc}")
            console.print(Panel(str(exc), title="Stage 3 Failed", style="red"))
            raise typer.Exit(code=1) from exc
        finally:
            input_paths = [
                repo_relative(Path(manifest.manifest_path or session_manifest_path(manifest.session_id))),
                repo_relative(resolve_context_path(manifest)),
                repo_relative(participants_file),
                repo_relative(directories["stage1"] / "raw_transcript.json"),
                repo_relative(directories["stage2"] / "diarization.json"),
            ]
            if speaker_map is not None:
                input_paths.append(repo_relative(speaker_map))
            run_path = write_run_metadata(
                runs_dir=directories["runs"],
                stage_name="stage3",
                status=status,
                input_paths=input_paths,
                output_paths=output_paths,
                config={
                    "force": force,
                    "mode": mode,
                    "backend": stage_metadata.get("backend"),
                    "backend_requested": backend,
                    "model": stage_metadata.get("model"),
                    "model_requested": model,
                    "provider": stage_metadata.get("provider"),
                    "prompt_version": stage_metadata.get("prompt_version"),
                    "speaker_map": repo_relative(speaker_map) if speaker_map else None,
                    "participants_file": repo_relative(participants_file),
                    "schema_version": stage_metadata.get("schema_version"),
                    "source_stage1_artifact": stage_metadata.get("source_stage1_artifact"),
                    "source_stage2_artifact": stage_metadata.get("source_stage2_artifact"),
                    "context_doc": stage_metadata.get("context_doc"),
                    "speaker_map_path": stage_metadata.get("speaker_map_path"),
                    "backend_usage": stage_metadata.get("backend_usage"),
                    "llm_usage": stage_metadata.get("llm_usage"),
                },
                notes=run_notes,
                started_at=started_at,
            )
            console.print(f"Run metadata: {run_path.as_posix()}")

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
        table.add_row("identify", "implemented", directories["stage3"].as_posix())
        table.add_row("organize", "validated, not implemented", directories["stage4"].as_posix())
        console.print(table)
        console.print(
            Panel(
                (
                    "Current CLI commands are `validate`, `transcribe`, `diarize`, `identify`, `organize`, and `run`. "
                    "Session input is resolved from `inputs/sessions/<session_id>/`."
                ),
                title="Current Workflow",
            )
        )
