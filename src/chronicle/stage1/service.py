"""Stage 1 transcription logic."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel

from ..exceptions import StageExecutionError
from ..paths import DEFAULT_PARAKEET_MODEL_DIR, repo_relative
from ..session import SessionManifest, resolve_audio_path
from .artifacts import (
    build_session_stage1_artifact,
    build_stage1_audio_summary,
    format_duration_summary,
    legacy_stage1_output_paths,
    load_existing_audio_artifact,
    parse_timestamp_seconds,
    session_stage1_output_paths,
    write_stage1_markdown,
)
from .audio import (
    STAGE1_TRANSCRIPT_SAMPLE_RATE,
    decode_audio_to_mono_16k,
    probe_audio_file,
)
from .parakeet import (
    build_parakeet_pipeline,
    ensure_local_parakeet_model,
    transcribe_with_parakeet_pipeline,
)
from ..utils import (
    write_json,
)

from .whisper import transcribe_with_faster_whisper

DEFAULT_STAGE1_PARAKEET_MODEL = "nvidia/parakeet-ctc-0.6b"
DEFAULT_STAGE1_WHISPER_MODEL = "large-v3-turbo"


def execute_stage1(
    manifest: SessionManifest,
    stage_dir: Path,
    participants_file: Path,
    force: bool,
    model_name: Optional[str],
    device: str,
    backend: str = "parakeet",
    whisper_compute_type: str = "int8",
    whisper_vad_filter: bool = True,
    parakeet_chunk_length_s: int = 15,
    parakeet_batch_size: int = 1,
    parakeet_overlap_stride_s: Optional[float] = None,
    parakeet_model_dir: Path = DEFAULT_PARAKEET_MODEL_DIR,
    local_files_only: bool = True,
    console: Optional[Console] = None,
) -> tuple[list[str], list[str], list[str]]:
    if backend not in ("whisper", "parakeet"):
        raise StageExecutionError(f"Unsupported Stage 1 backend `{backend}`. Expected `whisper` or `parakeet`.")

    if backend == "whisper":
        resolved_model_name = model_name or DEFAULT_STAGE1_WHISPER_MODEL
    else:
        resolved_model_name = model_name or DEFAULT_STAGE1_PARAKEET_MODEL

    generated_paths: list[str] = []
    skipped_paths: list[str] = []
    notes: list[str] = []
    session_json_path, session_markdown_path = session_stage1_output_paths(stage_dir)

    if not force and session_json_path.exists() and session_markdown_path.exists():
        skipped_paths.extend([repo_relative(session_json_path), repo_relative(session_markdown_path)])
        if console is not None:
            console.print(
                Panel(
                    "Skipping existing session-level Stage 1 artifacts. Use `--force` to overwrite.",
                    title="Stage 1 Skip",
                )
            )
        return [], skipped_paths, notes

    audio_artifacts: list[dict[str, Any]] = []
    session_model_info: Optional[dict[str, Any]] = None
    audio_probes = [probe_audio_file(resolve_audio_path(manifest, audio_file)) for audio_file in manifest.audio_files]
    total_session_duration = sum(probe.duration_seconds for probe in audio_probes)

    if console is not None:
        console.print(Panel(build_stage1_audio_summary(audio_probes), title="Stage 1 Audio Summary"))

    shared_parakeet_pipeline = None
    if backend == "parakeet":
        parakeet_local_model_dir = ensure_local_parakeet_model(
            model_name=resolved_model_name,
            model_dir=parakeet_model_dir,
            allow_download=not local_files_only,
        )
        notes.append(f"Using local Parakeet model files from {repo_relative(parakeet_local_model_dir)}.")

        if console is not None:
            with console.status(f"Loading Parakeet:{resolved_model_name} once for this session..."):
                _, shared_parakeet_pipeline = build_parakeet_pipeline(
                    model_dir=parakeet_local_model_dir,
                    device=device,
                )
        else:
            _, shared_parakeet_pipeline = build_parakeet_pipeline(
                model_dir=parakeet_local_model_dir,
                device=device,
            )

    processed_session_seconds = 0.0
    stage_started_at = time.perf_counter()
    progress_context = contextlib.nullcontext()
    overall_progress = None
    progress_task_id = None
    if console is not None:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        overall_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("{task.fields[current_file]}"),
            TextColumn("{task.fields[throughput]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        progress_context = overall_progress

    with progress_context:
        if overall_progress is not None:
            progress_task_id = overall_progress.add_task(
                "Stage 1",
                total=total_session_duration,
                completed=0.0,
                current_file="starting",
                throughput="warming up",
            )

        for audio_file in manifest.audio_files:
            legacy_json_path, legacy_markdown_path = legacy_stage1_output_paths(stage_dir, audio_file)
            audio_path = resolve_audio_path(manifest, audio_file)
            audio_label = repo_relative(audio_path)
            if not force and legacy_json_path.exists():
                payload = load_existing_audio_artifact(legacy_json_path)
                payload["source_audio"] = audio_label
                payload["session_id"] = manifest.session_id
                audio_artifacts.append(payload)
                session_model_info = session_model_info or dict(payload.get("model") or {})
                if console is not None:
                    console.print(
                        Panel(
                            f"Reusing existing transcript artifact for `{audio_file}` to build the session transcript.",
                            title="Stage 1 Reuse",
                        )
                    )
                continue

            if not force and legacy_markdown_path.exists():
                skipped_paths.append(repo_relative(legacy_markdown_path))
                if console is not None:
                    console.print(
                        Panel(
                            f"Legacy markdown exists for `{audio_file}`, but the JSON artifact was missing and the file will be retranscribed.",
                            title="Stage 1 Note",
                        )
                    )

            if backend == "parakeet":
                if console is not None:
                    with console.status(f"Decoding {audio_label} to mono 16 kHz audio..."):
                        audio = decode_audio_to_mono_16k(audio_path)
                else:
                    audio = decode_audio_to_mono_16k(audio_path)
                audio_duration_seconds = len(audio) / STAGE1_TRANSCRIPT_SAMPLE_RATE

            def report_progress(completed_chunks: int, total_chunks: int, processed_audio_seconds: float) -> None:
                if overall_progress is None or progress_task_id is None:
                    return
                completed_session_seconds = min(
                    total_session_duration,
                    processed_session_seconds + min(processed_audio_seconds, total_session_duration),
                )
                elapsed = max(time.perf_counter() - stage_started_at, 0.001)
                throughput = completed_session_seconds / elapsed
                remaining = max(total_session_duration - completed_session_seconds, 0.0)
                eta_seconds = remaining / throughput if throughput > 0 else None
                throughput_text = (
                    f"{throughput / 60.0:.2f} audio-min/wall-min | "
                    f"ETA {format_duration_summary(eta_seconds or 0.0)}"
                    if eta_seconds is not None
                    else "transcribing..."
                )
                overall_progress.update(
                    progress_task_id,
                    completed=completed_session_seconds,
                    current_file=audio_path.name,
                    throughput=throughput_text,
                )

            if backend == "whisper":
                model_info, segments, word_timestamps = transcribe_with_faster_whisper(
                    audio_path=audio_path,
                    language=manifest.language,
                    model_name=resolved_model_name,
                    device=device,
                    compute_type=whisper_compute_type,
                    vad_filter=whisper_vad_filter,
                    progress_callback=report_progress,
                )
                audio_duration_seconds = max(
                    (parse_timestamp_seconds(segments[-1]["end"]) if segments and segments[-1].get("end") else 0.0) or 0.0,
                    1.0,
                )
            else:
                model_info, segments, word_timestamps = transcribe_with_parakeet_pipeline(
                    audio=audio,
                    language=manifest.language,
                    model_name=resolved_model_name,
                    model_dir=parakeet_local_model_dir,
                    device=device,
                    chunk_length_s=parakeet_chunk_length_s,
                    asr_pipeline=shared_parakeet_pipeline,
                    batch_size=parakeet_batch_size,
                    overlap_stride_s=parakeet_overlap_stride_s,
                    progress_callback=report_progress,
                )

            transcript_text = "\n".join(
                segment["text"] for segment in segments if segment.get("text")
            ).strip()
            artifact_notes = list(notes)
            artifact_notes.extend(model_info.get("notes", []))

            audio_artifact_payload = {
                "stage": "stage1_transcription",
                "session_id": manifest.session_id,
                "source_audio": audio_label,
                "normalized_audio": None,
                "model": model_info,
                "language": manifest.language,
                "transcript_text": transcript_text,
                "segments": segments,
                "word_timestamps": word_timestamps,
                "notes": artifact_notes,
            }
            audio_artifacts.append(audio_artifact_payload)
            session_model_info = session_model_info or model_info
            processed_session_seconds += audio_duration_seconds

    if not audio_artifacts:
        raise StageExecutionError("Stage 1 produced no usable transcript artifacts for the session.")

    session_artifact_notes = list(notes)
    if any(
        artifact.get("session_id") != manifest.session_id
        for artifact in audio_artifacts
        if isinstance(artifact, dict)
    ):
        session_artifact_notes.append(
            "Session-level artifact was built from legacy per-audio artifacts created before the current session-id layout."
        )

    session_artifact = build_session_stage1_artifact(
        manifest=manifest,
        audio_artifacts=audio_artifacts,
        model_info=session_model_info
        or {
            "family": "parakeet",
            "name": resolved_model_name,
            "runtime": "parakeet",
            "device": device,
        },
        notes=session_artifact_notes,
    )
    write_json(session_json_path, session_artifact)
    write_stage1_markdown(session_markdown_path, manifest, session_artifact)
    generated_paths.extend([repo_relative(session_json_path), repo_relative(session_markdown_path)])

    return generated_paths, skipped_paths, notes
