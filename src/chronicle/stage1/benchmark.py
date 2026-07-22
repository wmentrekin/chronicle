"""Benchmark and concurrency helpers for Stage 1."""

from __future__ import annotations

import concurrent.futures
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

from ..exceptions import StageExecutionError
from ..paths import repo_relative
from ..session import SessionManifest, resolve_audio_path
from ..utils import normalize_text
from .audio import SampleWindow, STAGE1_TRANSCRIPT_SAMPLE_RATE, decode_audio_to_mono_16k, probe_audio_file
from .parakeet import build_parakeet_pipeline, ensure_local_parakeet_model


_PARAKEET_WORKER_PIPELINE: Any = None
_PARAKEET_WORKER_MODEL_DIR: Optional[str] = None
_PARAKEET_WORKER_DEVICE: Optional[str] = None


def build_evenly_spaced_sample_windows(
    manifest: SessionManifest,
    sample_seconds: int,
    sample_count: int,
) -> list[SampleWindow]:
    if sample_seconds <= 0:
        raise StageExecutionError("Benchmark sample seconds must be positive.")
    if sample_count <= 0:
        raise StageExecutionError("Benchmark sample count must be positive.")

    audio_paths = [resolve_audio_path(manifest, audio_file) for audio_file in manifest.audio_files]
    audio_probes = [probe_audio_file(audio_path) for audio_path in audio_paths]
    total_duration = sum(probe.duration_seconds for probe in audio_probes)
    if total_duration <= 0:
        raise StageExecutionError("Session has no measurable audio duration for benchmarking.")

    sample_windows: list[SampleWindow] = []
    cumulative_start = 0.0
    boundaries: list[tuple[Path, str, float, float]] = []
    for audio_path, probe in zip(audio_paths, audio_probes, strict=False):
        boundaries.append((audio_path, probe.source_audio, cumulative_start, cumulative_start + probe.duration_seconds))
        cumulative_start += probe.duration_seconds

    spacing = total_duration / (sample_count + 1)
    centers = [spacing * (index + 1) for index in range(sample_count)]
    for center in centers:
        for audio_path, source_audio, global_start, global_end in boundaries:
            if global_start <= center <= global_end:
                local_center = center - global_start
                audio_duration = global_end - global_start
                local_start = max(0.0, min(local_center - sample_seconds / 2.0, audio_duration - sample_seconds))
                sample_windows.append(
                    SampleWindow(
                        audio_path=audio_path,
                        source_audio=source_audio,
                        start_seconds=local_start,
                        duration_seconds=sample_seconds,
                    )
                )
                break
    return sample_windows


def benchmark_parakeet_chunk_sizes(
    manifest: SessionManifest,
    chunk_sizes: list[int],
    sample_seconds: int,
    sample_count: int,
    model_name: str,
    model_dir: Path,
    device: str,
    allow_download: bool,
) -> dict[str, Any]:
    if not chunk_sizes:
        raise StageExecutionError("Benchmark requires at least one chunk size.")

    local_model_dir = ensure_local_parakeet_model(
        model_name=model_name,
        model_dir=model_dir,
        allow_download=allow_download,
    )
    _, asr_pipeline = build_parakeet_pipeline(local_model_dir, device=device)
    sample_windows = build_evenly_spaced_sample_windows(
        manifest=manifest,
        sample_seconds=sample_seconds,
        sample_count=sample_count,
    )

    benchmark_runs: list[dict[str, Any]] = []
    for chunk_size in chunk_sizes:
        started_at = time.perf_counter()
        total_chunks = 0
        empty_chunks = 0
        unk_chunks = 0
        transcript_chars = 0

        for sample_window in sample_windows:
            audio = decode_audio_to_mono_16k(
                sample_window.audio_path,
                start_seconds=sample_window.start_seconds,
                duration_seconds=sample_window.duration_seconds,
            )
            chunk_samples = chunk_size * STAGE1_TRANSCRIPT_SAMPLE_RATE
            chunk_count = max(1, math.ceil(len(audio) / chunk_samples))
            total_chunks += chunk_count
            for index in range(chunk_count):
                start_sample = index * chunk_samples
                end_sample = min(len(audio), start_sample + chunk_samples)
                chunk_audio = audio[start_sample:end_sample]
                if chunk_audio.size == 0:
                    continue
                result = asr_pipeline(chunk_audio)
                text = normalize_text(str(result.get("text", "")).strip())
                transcript_chars += len(text)
                if not text:
                    empty_chunks += 1
                elif text == "<unk>":
                    unk_chunks += 1

        elapsed_seconds = max(time.perf_counter() - started_at, 0.001)
        total_sample_audio_seconds = sample_seconds * len(sample_windows)
        benchmark_runs.append(
            {
                "chunk_length_s": chunk_size,
                "sample_count": len(sample_windows),
                "sample_seconds": sample_seconds,
                "sample_audio_seconds": total_sample_audio_seconds,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "throughput_audio_minutes_per_wall_minute": round(
                    (total_sample_audio_seconds / 60.0) / (elapsed_seconds / 60.0),
                    3,
                ),
                "chunk_count": total_chunks,
                "empty_chunk_count": empty_chunks,
                "unk_chunk_count": unk_chunks,
                "transcript_char_count": transcript_chars,
            }
        )

    return {
        "stage": "stage1_benchmark",
        "session_id": manifest.session_id,
        "model_name": model_name,
        "model_dir": repo_relative(local_model_dir),
        "device": device,
        "sample_count": len(sample_windows),
        "sample_seconds": sample_seconds,
        "chunk_sizes": chunk_sizes,
        "samples": [
            {
                "source_audio": sample.source_audio,
                "start_seconds": round(sample.start_seconds, 3),
                "duration_seconds": sample.duration_seconds,
            }
            for sample in sample_windows
        ],
        "results": benchmark_runs,
    }


def build_partition_ranges(duration_seconds: float, partition_seconds: int) -> list[tuple[float, float]]:
    if partition_seconds <= 0:
        raise StageExecutionError("Partition seconds must be positive.")
    ranges: list[tuple[float, float]] = []
    start_seconds = 0.0
    while start_seconds < duration_seconds:
        end_seconds = min(duration_seconds, start_seconds + partition_seconds)
        ranges.append((start_seconds, end_seconds))
        if end_seconds >= duration_seconds:
            break
        start_seconds = end_seconds
    return ranges


def _init_parakeet_worker(model_dir: str, device: str) -> None:
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"
    global _PARAKEET_WORKER_PIPELINE
    global _PARAKEET_WORKER_MODEL_DIR
    global _PARAKEET_WORKER_DEVICE
    _PARAKEET_WORKER_MODEL_DIR = model_dir
    _PARAKEET_WORKER_DEVICE = device
    _, _PARAKEET_WORKER_PIPELINE = build_parakeet_pipeline(Path(model_dir), device=device)


def _parakeet_partition_batch_worker(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if _PARAKEET_WORKER_PIPELINE is None or _PARAKEET_WORKER_MODEL_DIR is None or _PARAKEET_WORKER_DEVICE is None:
        raise StageExecutionError("Parakeet worker was not initialized before processing tasks.")

    total_segments = 0
    total_unk = 0
    total_empty = 0
    total_chars = 0
    partition_count = 0

    for task in batch:
        audio = decode_audio_to_mono_16k(
            Path(task["audio_path"]),
            start_seconds=task["sample_start_seconds"] + task["partition_start_seconds"],
            duration_seconds=task["partition_duration_seconds"],
        )
        chunk_samples = task["chunk_length_s"] * STAGE1_TRANSCRIPT_SAMPLE_RATE
        chunk_count = max(1, math.ceil(len(audio) / chunk_samples))
        start_sample = 0
        segments: list[dict[str, Any]] = []
        for _ in range(chunk_count):
            end_sample = min(len(audio), start_sample + chunk_samples)
            chunk_audio = audio[start_sample:end_sample]
            if chunk_audio.size == 0:
                continue
            result = _PARAKEET_WORKER_PIPELINE(chunk_audio)
            raw_text = str(result.get("text", "")).strip()
            text = normalize_text(raw_text)
            decode_status = "ok"
            if not text:
                decode_status = "empty"
                total_empty += 1
            elif text == "<unk>":
                decode_status = "unk"
                total_unk += 1
            segments.append({"text": text, "decode_status": decode_status})
            if end_sample >= len(audio):
                break
            start_sample += chunk_samples

        total_segments += len(segments)
        total_chars += sum(len(str(segment.get("text") or "")) for segment in segments)
        partition_count += 1

    return {
        "partition_count": partition_count,
        "segment_count": total_segments,
        "unk_chunk_count": total_unk,
        "empty_chunk_count": total_empty,
        "transcript_char_count": total_chars,
    }


def benchmark_parakeet_concurrency(
    manifest: SessionManifest,
    worker_counts: list[int],
    partition_seconds: int,
    sample_seconds: int,
    sample_count: int,
    model_name: str,
    model_dir: Path,
    device: str,
    allow_download: bool,
    chunk_length_s: int,
) -> dict[str, Any]:
    if not worker_counts:
        raise StageExecutionError("Concurrency benchmark requires at least one worker count.")
    if chunk_length_s <= 0:
        raise StageExecutionError("Concurrency benchmark chunk length must be positive.")

    local_model_dir = ensure_local_parakeet_model(
        model_name=model_name,
        model_dir=model_dir,
        allow_download=allow_download,
    )
    sample_windows = build_evenly_spaced_sample_windows(
        manifest=manifest,
        sample_seconds=sample_seconds,
        sample_count=sample_count,
    )

    results: list[dict[str, Any]] = []
    for worker_count in worker_counts:
        if worker_count <= 0:
            raise StageExecutionError("Worker counts must be positive integers.")

        tasks: list[dict[str, Any]] = []
        for sample_index, sample_window in enumerate(sample_windows):
            partition_ranges = build_partition_ranges(sample_window.duration_seconds, partition_seconds)
            for partition_index, (start_seconds, end_seconds) in enumerate(partition_ranges):
                tasks.append(
                    {
                        "sample_index": sample_index,
                        "partition_index": partition_index,
                        "audio_path": sample_window.audio_path.as_posix(),
                        "sample_start_seconds": sample_window.start_seconds,
                        "partition_start_seconds": start_seconds,
                        "partition_duration_seconds": end_seconds - start_seconds,
                        "language": manifest.language,
                        "model_name": model_name,
                        "model_dir": local_model_dir.as_posix(),
                        "device": device,
                        "chunk_length_s": chunk_length_s,
                    }
                )

        started_at = time.perf_counter()
        total_segments = 0
        total_unk = 0
        total_empty = 0
        total_chars = 0
        task_batches: list[list[dict[str, Any]]] = [[] for _ in range(min(worker_count, max(1, len(tasks))))]
        for index, task in enumerate(tasks):
            task_batches[index % len(task_batches)].append(task)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_parakeet_worker,
            initargs=(local_model_dir.as_posix(), device),
        ) as executor:
            futures = [executor.submit(_parakeet_partition_batch_worker, batch) for batch in task_batches if batch]
            for future in concurrent.futures.as_completed(futures):
                partition_result = future.result()
                total_segments += partition_result["segment_count"]
                total_unk += partition_result["unk_chunk_count"]
                total_empty += partition_result["empty_chunk_count"]
                total_chars += partition_result["transcript_char_count"]

        elapsed_seconds = max(time.perf_counter() - started_at, 0.001)
        total_sample_audio_seconds = sum(sample.duration_seconds for sample in sample_windows)
        results.append(
            {
                "worker_count": worker_count,
                "partition_seconds": partition_seconds,
                "chunk_length_s": chunk_length_s,
                "sample_count": len(sample_windows),
                "sample_audio_seconds": total_sample_audio_seconds,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "throughput_audio_minutes_per_wall_minute": round(
                    (total_sample_audio_seconds / 60.0) / (elapsed_seconds / 60.0),
                    3,
                ),
                "partition_count": len(tasks),
                "segment_count": total_segments,
                "empty_chunk_count": total_empty,
                "unk_chunk_count": total_unk,
                "transcript_char_count": total_chars,
            }
        )

    return {
        "stage": "stage1_concurrency_benchmark",
        "session_id": manifest.session_id,
        "model_name": model_name,
        "model_dir": repo_relative(local_model_dir),
        "device": device,
        "chunk_length_s": chunk_length_s,
        "partition_seconds": partition_seconds,
        "worker_counts": worker_counts,
        "sample_count": len(sample_windows),
        "sample_seconds": sample_seconds,
        "samples": [
            {
                "source_audio": sample.source_audio,
                "start_seconds": round(sample.start_seconds, 3),
                "duration_seconds": sample.duration_seconds,
            }
            for sample in sample_windows
        ],
        "results": results,
    }


def benchmark_faster_whisper(
    manifest: SessionManifest,
    sample_seconds: int,
    sample_count: int,
    model_name: str = "large-v3-turbo",
    device: str = "cpu",
    compute_type: str = "int8",
    vad_filter: bool = True,
) -> dict[str, Any]:
    from .whisper import transcribe_with_faster_whisper

    sample_windows = build_evenly_spaced_sample_windows(
        manifest=manifest,
        sample_seconds=sample_seconds,
        sample_count=sample_count,
    )

    started_at = time.perf_counter()
    total_segments = 0
    empty_segments = 0
    transcript_chars = 0
    total_words = 0

    for sample_window in sample_windows:
        model_info, segments, word_timestamps = transcribe_with_faster_whisper(
            audio_path=sample_window.audio_path,
            language=manifest.language,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            vad_filter=vad_filter,
        )
        total_segments += len(segments)
        total_words += len(word_timestamps)
        for segment in segments:
            text = segment.get("text", "")
            transcript_chars += len(text)
            if segment.get("decode_status") == "empty":
                empty_segments += 1

    elapsed_seconds = max(time.perf_counter() - started_at, 0.001)
    total_sample_audio_seconds = sample_seconds * len(sample_windows)

    result_entry = {
        "backend": "faster-whisper",
        "model_name": model_name,
        "device": device,
        "compute_type": compute_type,
        "vad_filter": vad_filter,
        "sample_count": len(sample_windows),
        "sample_seconds": sample_seconds,
        "sample_audio_seconds": total_sample_audio_seconds,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_audio_minutes_per_wall_minute": round(
            (total_sample_audio_seconds / 60.0) / (elapsed_seconds / 60.0),
            3,
        ),
        "segment_count": total_segments,
        "word_count": total_words,
        "empty_segment_count": empty_segments,
        "transcript_char_count": transcript_chars,
    }

    return {
        "stage": "stage1_whisper_benchmark",
        "session_id": manifest.session_id,
        "sample_count": len(sample_windows),
        "sample_seconds": sample_seconds,
        "samples": [
            {
                "source_audio": sample.source_audio,
                "start_seconds": round(sample.start_seconds, 3),
                "duration_seconds": sample.duration_seconds,
            }
            for sample in sample_windows
        ],
        "results": [result_entry],
    }

