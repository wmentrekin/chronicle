"""Stage 1 CLI commands."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel

from ..exceptions import SessionValidationError, StageExecutionError
from ..paths import (
    DEFAULT_PARAKEET_MODEL_DIR,
    DEFAULT_PARTICIPANTS_FILE,
    DEFAULT_STAGE1_CACHE_DIR,
    ensure_output_dirs,
    repo_relative,
    session_manifest_path,
)
from ..session import require_valid_session, resolve_audio_path, resolve_context_path
from ..stage1.service import DEFAULT_STAGE1_BACKEND, execute_stage1
from ..utils import write_run_metadata
from .common import confirm_overwrite_if_needed, console, render_stage_plan


def register(app: typer.Typer) -> None:
    @app.command("transcribe")
    def transcribe_command(
        session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
        participants_file: Path = typer.Option(
            DEFAULT_PARTICIPANTS_FILE,
            "--participants-file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
        backend: str = typer.Option(
            DEFAULT_STAGE1_BACKEND,
            "--backend",
            help="Stage 1 backend: parakeet, faster-whisper, or auto.",
        ),
        model_name: Optional[str] = typer.Option(None, "--model", help="Override the backend-specific model name."),
        force: bool = typer.Option(False, "--force", help="Overwrite existing stage 1 artifacts."),
        device: str = typer.Option("cpu", "--device", help="Execution device for the selected backend."),
        compute_type: str = typer.Option("int8", "--compute-type", help="faster-whisper compute type."),
        cpu_threads: int = typer.Option(
            max(1, (os.cpu_count() or 1) - 1),
            "--cpu-threads",
            help="CPU threads to use for faster-whisper.",
        ),
        local_files_only: bool = typer.Option(
            True,
            "--local-files-only/--allow-download",
            help="Only use models already available locally.",
        ),
        download_root: Path = typer.Option(
            DEFAULT_STAGE1_CACHE_DIR,
            "--download-root",
            file_okay=False,
            dir_okay=True,
            help="Model cache directory for stage 1 backends.",
        ),
        parakeet_model_dir: Path = typer.Option(
            DEFAULT_PARAKEET_MODEL_DIR,
            "--parakeet-model-dir",
            file_okay=False,
            dir_okay=True,
            help="Chronicle-managed local directory for Parakeet model files.",
        ),
        beam_size: int = typer.Option(5, "--beam-size", help="faster-whisper beam size."),
        vad_filter: bool = typer.Option(
            True,
            "--vad-filter/--no-vad-filter",
            help="Enable or disable faster-whisper VAD filtering.",
        ),
        parakeet_chunk_length_s: int = typer.Option(
            15,
            "--parakeet-chunk-length-s",
            help="Chunk length in seconds for the Parakeet backend.",
        ),
        parakeet_batch_size: int = typer.Option(
            4,
            "--parakeet-batch-size",
            help="Number of Parakeet chunks to send per pipeline call.",
        ),
        experimental_overlap: bool = typer.Option(
            False,
            "--experimental-overlap/--no-experimental-overlap",
            help="Enable experimental Parakeet overlap mode using half-window stride.",
        ),
    ) -> None:
        """Run Stage 1 transcription."""
        try:
            manifest = require_valid_session(session_id, console, participants_file)
        except SessionValidationError as exc:
            raise typer.Exit(code=1) from exc

        directories = ensure_output_dirs(manifest.session_id)
        render_stage_plan(manifest, "stage1", directories)
        force = confirm_overwrite_if_needed(
            stage_label="Stage 1",
            force=force,
            output_paths=[
                directories["stage1"] / "raw_transcript.json",
                directories["stage1"] / "raw_transcript.md",
            ],
        )

        started_at = datetime.now(timezone.utc)
        run_notes: list[str] = []
        output_paths: list[str] = []
        status = "failed"
        try:
            output_paths, skipped_paths, stage_notes = execute_stage1(
                manifest=manifest,
                stage_dir=directories["stage1"],
                participants_file=participants_file,
                force=force,
                backend=backend,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                local_files_only=local_files_only,
                download_root=download_root,
                beam_size=beam_size,
                vad_filter=vad_filter,
                parakeet_chunk_length_s=parakeet_chunk_length_s,
                parakeet_batch_size=parakeet_batch_size,
                parakeet_overlap_stride_s=(parakeet_chunk_length_s / 2.0) if experimental_overlap else None,
                parakeet_model_dir=parakeet_model_dir,
                console=console,
            )
            run_notes.extend(stage_notes)
            status = "partial" if skipped_paths and not output_paths else "success"
            console.print(
                Panel(
                    f"Stage 1 wrote {len(output_paths)} artifact file(s) for session `{manifest.session_id}`.",
                    title="Stage 1 Complete",
                )
            )
        except KeyboardInterrupt as exc:
            run_notes.append("Stage 1 run was interrupted before artifacts were written.")
            console.print(Panel("Stage 1 run interrupted.", title="Stage 1 Interrupted", style="yellow"))
            raise typer.Exit(code=130) from exc
        except StageExecutionError as exc:
            run_notes.append(str(exc))
            console.print(Panel(str(exc), title="Stage 1 Failed", style="red"))
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            run_notes.append(f"Unhandled stage 1 error: {exc}")
            console.print(Panel(str(exc), title="Stage 1 Failed", style="red"))
            raise typer.Exit(code=1) from exc
        finally:
            run_path = write_run_metadata(
                runs_dir=directories["runs"],
                stage_name="stage1",
                status=status,
                input_paths=[repo_relative(Path(manifest.manifest_path or session_manifest_path(manifest.session_id)))]
                + [repo_relative(resolve_audio_path(manifest, audio_file)) for audio_file in manifest.audio_files]
                + [repo_relative(resolve_context_path(manifest))],
                output_paths=output_paths,
                config={
                    "backend": backend,
                    "model_name": model_name,
                    "device": device,
                    "compute_type": compute_type,
                    "cpu_threads": cpu_threads,
                    "local_files_only": local_files_only,
                    "download_root": download_root.as_posix(),
                    "parakeet_model_dir": parakeet_model_dir.as_posix(),
                    "beam_size": beam_size,
                    "vad_filter": vad_filter,
                    "parakeet_chunk_length_s": parakeet_chunk_length_s,
                    "parakeet_batch_size": parakeet_batch_size,
                    "experimental_overlap": experimental_overlap,
                    "parakeet_overlap_stride_s": (parakeet_chunk_length_s / 2.0) if experimental_overlap else None,
                },
                notes=run_notes,
                started_at=started_at,
            )
            console.print(f"Run metadata: {run_path.as_posix()}")
