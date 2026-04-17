"""Benchmark CLI commands."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from ..exceptions import SessionValidationError, StageExecutionError
from ..paths import DEFAULT_PARAKEET_MODEL_DIR, DEFAULT_PARTICIPANTS_FILE, REPO_ROOT, ensure_output_dirs, repo_relative
from ..session import require_valid_session
from ..stage1.service import (
    DEFAULT_STAGE1_PARAKEET_MODEL,
    benchmark_parakeet_concurrency,
    benchmark_parakeet_chunk_sizes,
)
from ..stage2.benchmark import (
    DEFAULT_STAGE2_BACKEND,
    DEFAULT_STAGE2_PYANNOTE_MODEL,
    DEFAULT_STAGE2_SPEECHBRAIN_EMBEDDING_MODEL,
    DEFAULT_STAGE2_SPEECHBRAIN_VAD_MODEL,
    DEFAULT_STAGE2_VENV_PYTHON,
    run_stage2_pyannote_spike,
    run_stage2_speechbrain_spike,
)
from ..utils import write_json
from .common import console


def register(app: typer.Typer) -> None:
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

    @app.command("benchmark-stage2")
    def benchmark_stage2_command(
        session_id: str = typer.Argument(..., help="Session folder name under inputs/sessions/."),
        participants_file: Path = typer.Option(
            DEFAULT_PARTICIPANTS_FILE,
            "--participants-file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
        audio_file: str | None = typer.Option(
            None,
            "--audio-file",
            help="Specific audio file entry from session.yaml to sample. Defaults to the first session audio file.",
        ),
        sample_seconds: int = typer.Option(
            120,
            "--sample-seconds",
            help="Length in seconds for the local diarization sample window.",
        ),
        sample_start_seconds: float = typer.Option(
            0.0,
            "--sample-start-seconds",
            help="Start offset in seconds for the sample window.",
        ),
        backend: str = typer.Option(
            DEFAULT_STAGE2_BACKEND,
            "--backend",
            help="Stage 2 spike backend to run (`pyannote` or `speechbrain`).",
        ),
        num_speakers: int | None = typer.Option(
            None,
            "--num-speakers",
            help="Exact expected speaker count for the diarization run.",
        ),
        min_speakers: int | None = typer.Option(
            None,
            "--min-speakers",
            help="Minimum speaker count hint for the diarization run.",
        ),
        max_speakers: int | None = typer.Option(
            None,
            "--max-speakers",
            help="Maximum speaker count hint for the diarization run.",
        ),
        device: str = typer.Option("cpu", "--device", help="Execution device for the diarization spike."),
        stage2_python: Path = typer.Option(
            DEFAULT_STAGE2_VENV_PYTHON,
            "--stage2-python",
            file_okay=True,
            dir_okay=False,
            help="Python executable for the separate Chronicle-managed Stage 2 runtime.",
        ),
        model_name: str = typer.Option(
            DEFAULT_STAGE2_PYANNOTE_MODEL,
            "--model",
            help="Pyannote model id used when `--backend pyannote`.",
        ),
        vad_model_name: str = typer.Option(
            DEFAULT_STAGE2_SPEECHBRAIN_VAD_MODEL,
            "--vad-model",
            help="SpeechBrain VAD model id used when `--backend speechbrain`.",
        ),
        embedding_model_name: str = typer.Option(
            DEFAULT_STAGE2_SPEECHBRAIN_EMBEDDING_MODEL,
            "--embedding-model",
            help="SpeechBrain embedding model id used when `--backend speechbrain`.",
        ),
        force: bool = typer.Option(False, "--force", help="Overwrite existing Stage 2 spike artifacts."),
    ) -> None:
        """Benchmark a Stage 2 diarization backend on one local sample window."""
        try:
            manifest = require_valid_session(session_id, console, participants_file)
        except SessionValidationError as exc:
            raise typer.Exit(code=1) from exc

        directories = ensure_output_dirs(manifest.session_id)
        try:
            with console.status(f"Running Stage 2 {backend} spike..."):
                if backend == "pyannote":
                    result = run_stage2_pyannote_spike(
                        manifest=manifest,
                        audio_file=audio_file,
                        sample_seconds=sample_seconds,
                        sample_start_seconds=sample_start_seconds,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        device=device,
                        stage2_python=stage2_python,
                        model_name=model_name,
                        spike_dir=directories["stage2_spike"],
                        force=force,
                    )
                elif backend == "speechbrain":
                    result = run_stage2_speechbrain_spike(
                        manifest=manifest,
                        audio_file=audio_file,
                        sample_seconds=sample_seconds,
                        sample_start_seconds=sample_start_seconds,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        device=device,
                        stage2_python=stage2_python,
                        vad_model_name=vad_model_name,
                        embedding_model_name=embedding_model_name,
                        spike_dir=directories["stage2_spike"],
                        force=force,
                    )
                else:
                    raise StageExecutionError(
                        f"Unsupported Stage 2 benchmark backend `{backend}`. Use `pyannote` or `speechbrain`."
                    )
        except StageExecutionError as exc:
            console.print(Panel(str(exc), title="Stage 2 Spike Failed", style="red"))
            raise typer.Exit(code=1) from exc

        runner = result["runner"]
        speakers = ", ".join(runner.get("speakers", [])) or "none"
        table = Table(title="Stage 2 Spike")
        table.add_column("Item", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Sample audio", str(result["sample_audio"]))
        table.add_row("Sample start", f"{result['sample_start_seconds']}s")
        table.add_row("Sample length", f"{result['sample_seconds']}s")
        table.add_row("Backend", str(result["backend"]))
        table.add_row("Load time", f"{runner['load_seconds']}s")
        table.add_row("Run time", f"{runner['run_seconds']}s")
        table.add_row("Wall time", f"{result['wall_seconds']}s")
        table.add_row("Speakers", speakers)
        table.add_row("Turns", str(runner["turn_count"]))
        console.print(table)
        console.print(
            Panel(
                (
                    f"Stage 2 spike artifact written to "
                    f"`{repo_relative(directories['stage2_spike'])}`."
                ),
                title="Stage 2 Spike Complete",
            )
        )
