"""Stage 1 CLI commands."""

from __future__ import annotations

import getpass
import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.panel import Panel

from ..exceptions import SessionValidationError, StageExecutionError
from ..paths import (
    DEFAULT_PARAKEET_MODEL_DIR,
    DEFAULT_PARTICIPANTS_FILE,
    OUTPUTS_ROOT,
    ensure_output_dirs,
    input_session_dir,
    repo_relative,
    session_manifest_path,
)
from ..stage1 import GcpStage1Config, build_gcp_stage1_plan, run_gcp_stage1_plan
from ..session import require_valid_session, resolve_audio_path, resolve_context_path
from ..stage1.service import DEFAULT_STAGE1_PARAKEET_MODEL, execute_stage1
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
        project_id: str | None = typer.Option(
            None,
            "--project-id",
            help="Google Cloud project id for the Stage 1 worker.",
        ),
        instance_name: str | None = typer.Option(
            None,
            "--instance-name",
            help="Compute Engine instance name. Defaults to a session-derived name.",
        ),
        zone: str = typer.Option(
            "us-central1-a",
            "--zone",
            help="Google Cloud zone for the Stage 1 worker.",
        ),
        machine_type: str = typer.Option(
            "e2-standard-8",
            "--machine-type",
            help="Compute Engine machine type for the Stage 1 worker.",
        ),
        gpu_enabled: bool = typer.Option(
            False,
            "--gpu-enabled/--cpu-only",
            help="Use a GPU-backed worker instead of the default CPU worker.",
        ),
        gpu_type: str = typer.Option(
            "nvidia-l4",
            "--gpu-type",
            help="GPU accelerator type when --gpu-enabled is set.",
        ),
        worker_user: str = typer.Option(
            getpass.getuser(),
            "--worker-user",
            help="Linux username on the worker VM.",
        ),
        backend: str = typer.Option(
            "whisper",
            "--backend",
            help="Transcription backend (`whisper` or `parakeet`).",
        ),
        keep_instance: bool = typer.Option(
            False,
            "--keep-instance/--teardown",
            help="Keep the worker instance running after Stage 1 completes.",
        ),
        local_worker: bool = typer.Option(False, "--local-worker", hidden=True),
        model_name: str | None = typer.Option(None, "--model", help="Override model name/id."),
        force: bool = typer.Option(False, "--force", hidden=True),
        device: str = typer.Option("cpu", "--device", hidden=True),
        local_files_only: bool = typer.Option(True, "--local-files-only/--allow-download", hidden=True),
        parakeet_model_dir: Path = typer.Option(
            DEFAULT_PARAKEET_MODEL_DIR,
            "--parakeet-model-dir",
            file_okay=False,
            dir_okay=True,
            hidden=True,
        ),
        parakeet_chunk_length_s: int = typer.Option(15, "--parakeet-chunk-length-s", hidden=True),
        parakeet_batch_size: int = typer.Option(4, "--parakeet-batch-size", hidden=True),
        experimental_overlap: bool = typer.Option(False, "--experimental-overlap/--no-experimental-overlap", hidden=True),
    ) -> None:
        """Run Stage 1 transcription on the default cloud worker."""
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

        if not local_worker:
            if not project_id:
                console.print(Panel("`--project-id` is required for Stage 1 orchestration.", title="Stage 1 Failed", style="red"))
                raise typer.Exit(code=1)
            cloud_config = GcpStage1Config(
                project_id=project_id,
                instance_name=instance_name or f"chronicle-stage1-{manifest.session_id}",
                session_id=manifest.session_id,
                backend=backend,
                zone=zone,
                machine_type=machine_type,
                gpu_enabled=gpu_enabled,
                gpu_type=gpu_type,
                model_name=model_name or ("large-v3-turbo" if backend == "whisper" else DEFAULT_STAGE1_PARAKEET_MODEL),
                local_output_dir=OUTPUTS_ROOT.as_posix(),
                local_participants_file=participants_file.as_posix(),
            )
            plan = build_gcp_stage1_plan(
                config=cloud_config,
                local_session_dir=input_session_dir(manifest.session_id),
                worker_user=worker_user,
            )
            try:
                run_gcp_stage1_plan(plan, console=console, keep_instance=keep_instance)
                console.print(
                    Panel(
                        f"Stage 1 completed for session `{manifest.session_id}`.",
                        title="Stage 1 Complete",
                    )
                )
            except StageExecutionError as exc:
                console.print(Panel(str(exc), title="Stage 1 Failed", style="red"))
                raise typer.Exit(code=1) from exc
            return

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
                model_name=model_name,
                device=device,
                backend=backend,
                local_files_only=local_files_only,
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
                    "local_files_only": local_files_only,
                    "parakeet_model_dir": parakeet_model_dir.as_posix(),
                    "parakeet_chunk_length_s": parakeet_chunk_length_s,
                    "parakeet_batch_size": parakeet_batch_size,
                    "experimental_overlap": experimental_overlap,
                    "parakeet_overlap_stride_s": (parakeet_chunk_length_s / 2.0) if experimental_overlap else None,
                },
                notes=run_notes,
                started_at=started_at,
            )
            console.print(f"Run metadata: {run_path.as_posix()}")

    @app.command("transcribe-plan")
    def transcribe_plan_command(
        session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
        project_id: str = typer.Option(..., "--project-id", help="Google Cloud project id."),
        instance_name: str = typer.Option(..., "--instance-name", help="Compute Engine instance name."),
        local_session_dir: Path = typer.Option(..., "--local-session-dir", exists=True, file_okay=False, dir_okay=True),
        backend: str = typer.Option("whisper", "--backend", help="Transcription backend (`whisper` or `parakeet`)."),
        worker_user: str = typer.Option(getpass.getuser(), "--worker-user", help="Linux username on the worker VM."),
        zone: str = typer.Option("us-central1-a", "--zone", help="Google Cloud zone."),
        machine_type: str = typer.Option("e2-standard-8", "--machine-type", help="Compute Engine machine type."),
        gpu_enabled: bool = typer.Option(False, "--gpu-enabled/--cpu-only", help="Include GPU accelerator flags."),
        gpu_type: str = typer.Option("nvidia-l4", "--gpu-type", help="GPU accelerator type."),
        model_name: str | None = typer.Option(None, "--model", help="Override model name."),
        local_output_dir: str = typer.Option("./outputs", "--local-output-dir", help="Local destination for downloaded outputs."),
        local_participants_file: str = typer.Option(
            "inputs/global/participants.yaml",
            "--local-participants-file",
            help="Local participants file to upload.",
        ),
    ) -> None:
        """Render a Stage 1 cloud command plan."""
        config = GcpStage1Config(
            project_id=project_id,
            instance_name=instance_name,
            session_id=session_id,
            backend=backend,
            zone=zone,
            machine_type=machine_type,
            gpu_enabled=gpu_enabled,
            gpu_type=gpu_type,
            model_name=model_name or ("large-v3-turbo" if backend == "whisper" else DEFAULT_STAGE1_PARAKEET_MODEL),
            local_output_dir=local_output_dir,
            local_participants_file=local_participants_file,
        )
        plan = build_gcp_stage1_plan(
            config=config,
            local_session_dir=local_session_dir,
            worker_user=worker_user,
        )
        typer.echo(json.dumps(plan.asdict(), indent=2))

    @app.command("transcribe-command")
    def transcribe_command_command(
        step: str = typer.Argument(..., help="Stage 1 step id."),
        session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
        project_id: str = typer.Option(..., "--project-id", help="Google Cloud project id."),
        instance_name: str = typer.Option(..., "--instance-name", help="Compute Engine instance name."),
        local_session_dir: Path = typer.Option(..., "--local-session-dir", exists=True, file_okay=False, dir_okay=True),
        backend: str = typer.Option("whisper", "--backend", help="Transcription backend (`whisper` or `parakeet`)."),
        worker_user: str = typer.Option(getpass.getuser(), "--worker-user", help="Linux username on the worker VM."),
        zone: str = typer.Option("us-central1-a", "--zone", help="Google Cloud zone."),
        machine_type: str = typer.Option("e2-standard-8", "--machine-type", help="Compute Engine machine type."),
        gpu_enabled: bool = typer.Option(False, "--gpu-enabled/--cpu-only", help="Include GPU accelerator flags."),
        gpu_type: str = typer.Option("nvidia-l4", "--gpu-type", help="GPU accelerator type."),
        model_name: str | None = typer.Option(None, "--model", help="Override model name."),
        local_output_dir: str = typer.Option("./outputs", "--local-output-dir", help="Local destination for downloaded outputs."),
        local_participants_file: str = typer.Option(
            "inputs/global/participants.yaml",
            "--local-participants-file",
            help="Local participants file to upload.",
        ),
    ) -> None:
        """Render one shell-safe Stage 1 orchestration command."""
        config = GcpStage1Config(
            project_id=project_id,
            instance_name=instance_name,
            session_id=session_id,
            backend=backend,
            zone=zone,
            machine_type=machine_type,
            gpu_enabled=gpu_enabled,
            gpu_type=gpu_type,
            model_name=model_name or ("large-v3-turbo" if backend == "whisper" else DEFAULT_STAGE1_PARAKEET_MODEL),
            local_output_dir=local_output_dir,
            local_participants_file=local_participants_file,
        )
        plan = build_gcp_stage1_plan(
            config=config,
            local_session_dir=local_session_dir,
            worker_user=worker_user,
        )
        try:
            typer.echo(plan.shell_command(step))
        except KeyError as exc:
            raise typer.BadParameter(str(exc)) from exc
