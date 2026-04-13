"""Chronicle command-line interface."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .exceptions import SessionValidationError, StageExecutionError
from .paths import (
    DEFAULT_PARAKEET_MODEL_DIR,
    DEFAULT_PARTICIPANTS_FILE,
    DEFAULT_STAGE1_CACHE_DIR,
    ensure_output_dirs,
    repo_relative,
    session_manifest_path,
)
from .session import (
    SessionManifest,
    render_validation_report,
    require_valid_session,
    resolve_audio_path,
    resolve_context_path,
    validate_session,
)
from .stage1.service import (
    DEFAULT_STAGE1_BACKEND,
    DEFAULT_STAGE1_PARAKEET_MODEL,
    benchmark_parakeet_concurrency,
    benchmark_parakeet_chunk_sizes,
    ensure_local_parakeet_model,
    execute_stage1,
)
from .stage2.service import execute_stage2
from .stage3.service import planned_stage3_artifacts
from .utils import write_json, write_run_metadata


console = Console()
app = typer.Typer(
    add_completion=False,
    help="Chronicle family oral-history pipeline CLI.",
    no_args_is_help=True,
)
models_app = typer.Typer(add_completion=False, help="Manage local model files.")
app.add_typer(models_app, name="models")


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
            (directories["stage2"] / "diarized_conversation.json").as_posix(),
        )
    elif stage_name == "stage3":
        table.add_row(
            "Planned artifacts",
            ", ".join(path.name for path in planned_stage3_artifacts(manifest, directories["stage3"])),
        )
    console.print(table)


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
    report = validate_session(session_id, participants_file)
    render_validation_report(report, console, as_json=as_json)
    if not report.ok or (strict and report.warnings):
        raise typer.Exit(code=1)


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
        help="Stage 1 backend: auto, faster-whisper, or parakeet.",
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
                "experimental_overlap": experimental_overlap,
                "parakeet_overlap_stride_s": (parakeet_chunk_length_s / 2.0) if experimental_overlap else None,
            },
            notes=run_notes,
            started_at=started_at,
        )
        console.print(f"Run metadata: {run_path.as_posix()}")


@models_app.command("fetch")
def fetch_model_command(
    model_key: str = typer.Argument(..., help="Model key to fetch. Currently supported: parakeet."),
    model_name: str = typer.Option(
        DEFAULT_STAGE1_PARAKEET_MODEL,
        "--model",
        help="Remote Hugging Face model id to fetch.",
    ),
    model_dir: Path = typer.Option(
        DEFAULT_PARAKEET_MODEL_DIR,
        "--model-dir",
        file_okay=False,
        dir_okay=True,
        help="Local destination for the downloaded model files.",
    ),
) -> None:
    """Download a managed local model for Chronicle."""
    if model_key.strip().lower() != "parakeet":
        console.print(Panel("Only `parakeet` is currently supported.", title="Unsupported Model", style="red"))
        raise typer.Exit(code=1)
    try:
        resolved = ensure_local_parakeet_model(
            model_name=model_name,
            model_dir=model_dir,
            allow_download=True,
        )
    except StageExecutionError as exc:
        console.print(Panel(str(exc), title="Model Fetch Failed", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(
        Panel(
            f"Parakeet model files are available at `{repo_relative(resolved)}`.",
            title="Model Ready",
        )
    )


@app.command("benchmark-stage1")
def benchmark_stage1_command(
    session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
    participants_file: Path = typer.Option(
        DEFAULT_PARTICIPANTS_FILE,
        "--participants-file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    model_name: str = typer.Option(
        DEFAULT_STAGE1_PARAKEET_MODEL,
        "--model",
        help="Parakeet model id used for benchmarking.",
    ),
    parakeet_model_dir: Path = typer.Option(
        DEFAULT_PARAKEET_MODEL_DIR,
        "--parakeet-model-dir",
        file_okay=False,
        dir_okay=True,
        help="Chronicle-managed local directory for Parakeet model files.",
    ),
    device: str = typer.Option("cpu", "--device", help="Execution device for benchmarking."),
    allow_download: bool = typer.Option(
        False,
        "--allow-download/--local-only",
        help="Allow Chronicle to download the model into the managed model directory if it is missing.",
    ),
    sample_seconds: int = typer.Option(
        120,
        "--sample-seconds",
        help="Length in seconds for each evenly spaced benchmark sample window.",
    ),
    sample_count: int = typer.Option(
        4,
        "--sample-count",
        help="Number of evenly spaced samples to benchmark across the full session.",
    ),
    chunk_sizes: str = typer.Option(
        "3,5,10,15,20",
        "--chunk-sizes",
        help="Comma-separated Parakeet chunk sizes in seconds.",
    ),
) -> None:
    """Benchmark Stage 1 Parakeet chunk sizes on evenly spaced session subsamples."""
    try:
        manifest = require_valid_session(session_id, console, participants_file)
    except SessionValidationError as exc:
        raise typer.Exit(code=1) from exc

    directories = ensure_output_dirs(manifest.session_id)
    try:
        parsed_chunk_sizes = sorted(
            {
                int(value.strip())
                for value in chunk_sizes.split(",")
                if value.strip()
            }
        )
    except ValueError as exc:
        console.print(Panel("Chunk sizes must be integers.", title="Invalid Benchmark Config", style="red"))
        raise typer.Exit(code=1) from exc

    started_at = datetime.now(timezone.utc)
    try:
        with console.status("Benchmarking Stage 1 chunk sizes..."):
            result = benchmark_parakeet_chunk_sizes(
                manifest=manifest,
                chunk_sizes=parsed_chunk_sizes,
                sample_seconds=sample_seconds,
                sample_count=sample_count,
                model_name=model_name,
                model_dir=parakeet_model_dir,
                device=device,
                allow_download=allow_download,
            )
    except StageExecutionError as exc:
        console.print(Panel(str(exc), title="Benchmark Failed", style="red"))
        raise typer.Exit(code=1) from exc

    benchmark_table = Table(title="Stage 1 Benchmark")
    benchmark_table.add_column("Chunk (s)", style="cyan")
    benchmark_table.add_column("Samples", style="white")
    benchmark_table.add_column("Wall Time", style="white")
    benchmark_table.add_column("Throughput", style="white")
    benchmark_table.add_column("Chunks", style="white")
    benchmark_table.add_column("Empty", style="white")
    benchmark_table.add_column("<unk>", style="white")
    for row in result["results"]:
        benchmark_table.add_row(
            str(row["chunk_length_s"]),
            str(row["sample_count"]),
            f"{row['elapsed_seconds']}s",
            f"{row['throughput_audio_minutes_per_wall_minute']}x",
            str(row["chunk_count"]),
            str(row["empty_chunk_count"]),
            str(row["unk_chunk_count"]),
        )
    console.print(benchmark_table)

    benchmark_path = directories["runs"] / f"stage1-benchmark.{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json(benchmark_path, result)
    console.print(f"Benchmark results: {benchmark_path.as_posix()}")


@app.command("benchmark-stage1-concurrency")
def benchmark_stage1_concurrency_command(
    session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
    participants_file: Path = typer.Option(
        DEFAULT_PARTICIPANTS_FILE,
        "--participants-file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    model_name: str = typer.Option(
        DEFAULT_STAGE1_PARAKEET_MODEL,
        "--model",
        help="Parakeet model id used for benchmarking.",
    ),
    parakeet_model_dir: Path = typer.Option(
        DEFAULT_PARAKEET_MODEL_DIR,
        "--parakeet-model-dir",
        file_okay=False,
        dir_okay=True,
        help="Chronicle-managed local directory for Parakeet model files.",
    ),
    device: str = typer.Option("cpu", "--device", help="Execution device for benchmarking."),
    allow_download: bool = typer.Option(
        False,
        "--allow-download/--local-only",
        help="Allow Chronicle to download the model into the managed model directory if it is missing.",
    ),
    sample_seconds: int = typer.Option(
        120,
        "--sample-seconds",
        help="Length in seconds for each evenly spaced benchmark sample window.",
    ),
    sample_count: int = typer.Option(
        4,
        "--sample-count",
        help="Number of evenly spaced samples to benchmark across the full session.",
    ),
    chunk_length_s: int = typer.Option(
        15,
        "--chunk-length-s",
        help="Parakeet chunk size to hold constant during the concurrency benchmark.",
    ),
    worker_counts: str = typer.Option(
        "1,2,3",
        "--workers",
        help="Comma-separated worker counts to benchmark.",
    ),
    partition_seconds: int = typer.Option(
        600,
        "--partition-seconds",
        help="Partition size in seconds for each worker task.",
    ),
) -> None:
    """Benchmark Stage 1 Parakeet concurrency using partitioned worker processes."""
    try:
        manifest = require_valid_session(session_id, console, participants_file)
    except SessionValidationError as exc:
        raise typer.Exit(code=1) from exc

    directories = ensure_output_dirs(manifest.session_id)
    try:
        parsed_worker_counts = sorted(
            {
                int(value.strip())
                for value in worker_counts.split(",")
                if value.strip()
            }
        )
    except ValueError as exc:
        console.print(Panel("Worker counts must be integers.", title="Invalid Benchmark Config", style="red"))
        raise typer.Exit(code=1) from exc

    started_at = datetime.now(timezone.utc)
    try:
        with console.status("Benchmarking Stage 1 concurrency..."):
            result = benchmark_parakeet_concurrency(
                manifest=manifest,
                worker_counts=parsed_worker_counts,
                partition_seconds=partition_seconds,
                sample_seconds=sample_seconds,
                sample_count=sample_count,
                model_name=model_name,
                model_dir=parakeet_model_dir,
                device=device,
                allow_download=allow_download,
                chunk_length_s=chunk_length_s,
            )
    except StageExecutionError as exc:
        console.print(Panel(str(exc), title="Concurrency Benchmark Failed", style="red"))
        raise typer.Exit(code=1) from exc

    benchmark_table = Table(title="Stage 1 Concurrency Benchmark")
    benchmark_table.add_column("Workers", style="cyan")
    benchmark_table.add_column("Partition (s)", style="white")
    benchmark_table.add_column("Wall Time", style="white")
    benchmark_table.add_column("Throughput", style="white")
    benchmark_table.add_column("Partitions", style="white")
    benchmark_table.add_column("Empty", style="white")
    benchmark_table.add_column("<unk>", style="white")
    for row in result["results"]:
        benchmark_table.add_row(
            str(row["worker_count"]),
            str(row["partition_seconds"]),
            f"{row['elapsed_seconds']}s",
            f"{row['throughput_audio_minutes_per_wall_minute']}x",
            str(row["partition_count"]),
            str(row["empty_chunk_count"]),
            str(row["unk_chunk_count"]),
        )
    console.print(benchmark_table)

    benchmark_path = directories["runs"] / f"stage1-concurrency-benchmark.{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json(benchmark_path, result)
    console.print(f"Concurrency benchmark results: {benchmark_path.as_posix()}")


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


def main() -> None:
    app()
