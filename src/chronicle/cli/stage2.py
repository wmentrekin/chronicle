"""Stage 2 CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel

from datetime import datetime, timezone

from ..exceptions import SessionValidationError, StageExecutionError
from getpass import getuser
import re

from ..paths import DEFAULT_PARTICIPANTS_FILE, OUTPUTS_ROOT, ensure_output_dirs, input_session_dir, repo_relative, session_manifest_path
from ..session import require_valid_session
from ..stage2 import GcpStage2Config, build_gcp_stage2_plan, run_gcp_stage2_plan
from ..stage2.service import (
    DEFAULT_STAGE2_SPEECHBRAIN_PYTHON,
    execute_stage2,
)
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
        project_id: str | None = typer.Option(
            None,
            "--project-id",
            help="Google Cloud project id for the Stage 2 worker.",
        ),
        instance_name: str | None = typer.Option(
            None,
            "--instance-name",
            help="Compute Engine instance name. Defaults to a session-derived name.",
        ),
        zone: str = typer.Option(
            "us-central1-a",
            "--zone",
            help="Google Cloud zone for the Stage 2 worker.",
        ),
        machine_type: str = typer.Option(
            "e2-standard-8",
            "--machine-type",
            help="Compute Engine machine type for the Stage 2 worker.",
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
            getuser(),
            "--worker-user",
            help="Linux username on the worker VM.",
        ),
        keep_instance: bool = typer.Option(
            False,
            "--keep-instance/--teardown",
            help="Keep the worker instance running after Stage 2 completes.",
        ),
        local_worker: bool = typer.Option(False, "--local-worker", hidden=True),
        num_speakers: int | None = typer.Option(
            None,
            "--num-speakers",
            help="Exact expected speaker count for anonymous diarization.",
        ),
        min_speakers: int | None = typer.Option(
            None,
            "--min-speakers",
            help="Minimum speaker count hint for anonymous diarization.",
        ),
        max_speakers: int | None = typer.Option(
            None,
            "--max-speakers",
            help="Maximum speaker count hint for anonymous diarization.",
        ),
        device: str = typer.Option("cpu", "--device", help="Execution device for Stage 2."),
        stage2_python: Path = typer.Option(
            DEFAULT_STAGE2_SPEECHBRAIN_PYTHON,
            "--stage2-python",
            file_okay=True,
            dir_okay=False,
            help="Python executable for the separate Chronicle-managed Stage 2 runtime.",
        ),
        force: bool = typer.Option(False, "--force", help="Overwrite existing stage 2 artifacts."),
    ) -> None:
        """Run Stage 2 anonymous audio diarization over the current session audio."""
        try:
            manifest = require_valid_session(session_id, console, participants_file)
        except SessionValidationError as exc:
            raise typer.Exit(code=1) from exc

        directories = ensure_output_dirs(manifest.session_id)
        render_stage_plan(manifest, "stage2", directories)
        stage2_output_paths = [
            directories["stage2"] / "diarization.json",
            directories["stage2"] / "diarization.md",
            directories["stage2"] / "diarization.partial.json",
            directories["stage2"] / "diarization.partial.md",
        ]
        force = confirm_overwrite_if_needed(
            stage_label="Stage 2",
            force=force,
            output_paths=stage2_output_paths,
        )

        if not local_worker and project_id:
            default_instance_name = re.sub(
                r"[^a-z0-9-]+",
                "-",
                f"chronicle-stage2-{manifest.session_id}".lower().replace("_", "-"),
            ).strip("-")[:60]
            cloud_config = GcpStage2Config(
                project_id=project_id,
                instance_name=instance_name or default_instance_name,
                session_id=manifest.session_id,
                backend="speechbrain",
                zone=zone,
                machine_type=machine_type,
                gpu_enabled=gpu_enabled,
                gpu_type=gpu_type,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                local_output_dir=OUTPUTS_ROOT.as_posix(),
                local_participants_file=participants_file.as_posix(),
            )
            plan = build_gcp_stage2_plan(
                config=cloud_config,
                local_session_dir=input_session_dir(manifest.session_id),
                worker_user=worker_user,
            )
            try:
                run_gcp_stage2_plan(plan, console=console, keep_instance=keep_instance)
                console.print(
                    Panel(
                        f"Stage 2 completed for session `{manifest.session_id}`.",
                        title="Stage 2 Complete",
                    )
                )
            except StageExecutionError as exc:
                console.print(Panel(str(exc), title="Stage 2 Failed", style="red"))
                raise typer.Exit(code=1) from exc
            return

        started_at = datetime.now(timezone.utc)
        run_notes: list[str] = []
        output_paths: list[str] = []
        status = "failed"
        try:
            output_paths, skipped_paths, stage_notes = execute_stage2(
                manifest=manifest,
                stage2_dir=directories["stage2"],
                force=force,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                device=device,
                stage2_python=stage2_python,
                vad_model_name="speechbrain/vad-crdnn-libriparty",
                embedding_model_name="speechbrain/spkrec-ecapa-voxceleb",
                progress_callback=lambda message: console.print(f"[cyan]{message}[/cyan]"),
            )
            run_notes.extend(stage_notes)
            if skipped_paths and not output_paths:
                status = "partial"
                console.print(
                    Panel(
                        "Stage 2 artifacts already exist. Keeping the existing outputs unchanged.",
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
            partial_paths = [
                directories["stage2"] / "diarization.partial.json",
                directories["stage2"] / "diarization.partial.md",
            ]
            existing_partial_paths = [repo_relative(path) for path in partial_paths if path.exists()]
            if existing_partial_paths:
                run_notes.append(f"Partial Stage 2 artifacts exist: {', '.join(existing_partial_paths)}")
            console.print(Panel(str(exc), title="Stage 2 Failed", style="red"))
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            run_notes.append(f"Unhandled stage 2 error: {exc}")
            partial_paths = [
                directories["stage2"] / "diarization.partial.json",
                directories["stage2"] / "diarization.partial.md",
            ]
            existing_partial_paths = [repo_relative(path) for path in partial_paths if path.exists()]
            if existing_partial_paths:
                run_notes.append(f"Partial Stage 2 artifacts exist: {', '.join(existing_partial_paths)}")
            console.print(Panel(str(exc), title="Stage 2 Failed", style="red"))
            raise typer.Exit(code=1) from exc
        finally:
            if not output_paths:
                partial_paths = [
                    directories["stage2"] / "diarization.partial.json",
                    directories["stage2"] / "diarization.partial.md",
                ]
                output_paths = [repo_relative(path) for path in partial_paths if path.exists()]
            run_path = write_run_metadata(
                runs_dir=directories["runs"],
                stage_name="stage2",
                status=status,
                input_paths=[repo_relative(Path(manifest.manifest_path or session_manifest_path(manifest.session_id)))],
                output_paths=output_paths,
                config={
                    "force": force,
                    "participants_file": repo_relative(participants_file),
                    "num_speakers": num_speakers,
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                    "device": device,
                    "stage2_python": repo_relative(stage2_python),
                    "backend": "speechbrain",
                },
                notes=run_notes,
                started_at=started_at,
            )
            console.print(f"Run metadata: {run_path.as_posix()}")
