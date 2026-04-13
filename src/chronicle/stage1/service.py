"""Stage 1 transcription logic."""

from __future__ import annotations

import contextlib
import concurrent.futures
import io
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel

from ..exceptions import StageExecutionError
from ..paths import DEFAULT_PARAKEET_MODEL_DIR, DEFAULT_STAGE1_CACHE_DIR, repo_relative
from ..session import SessionManifest, resolve_audio_path
from ..utils import (
    format_timestamp,
    normalize_text,
    package_version,
    write_json,
)


DEFAULT_STAGE1_BACKEND = "auto"
DEFAULT_STAGE1_FASTER_WHISPER_MODEL = "distil-large-v3"
DEFAULT_STAGE1_PARAKEET_MODEL = "nvidia/parakeet-ctc-0.6b"
STAGE1_TRANSCRIPT_SAMPLE_RATE = 16000
_PARAKEET_WORKER_PIPELINE: Any = None
_PARAKEET_WORKER_MODEL_DIR: Optional[str] = None
_PARAKEET_WORKER_DEVICE: Optional[str] = None


@dataclass(frozen=True)
class AudioProbe:
    source_audio: str
    duration_seconds: float
    file_size_bytes: int


@dataclass(frozen=True)
class SampleWindow:
    audio_path: Path
    source_audio: str
    start_seconds: float
    duration_seconds: float


def parse_ffmpeg_duration(stderr_text: str) -> Optional[float]:
    match = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2})\.(\d{2})", stderr_text)
    if not match:
        return None
    hours, minutes, seconds, centiseconds = match.groups()
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(centiseconds) / 100.0
    )


def probe_audio_file(audio_path: Path) -> AudioProbe:
    try:
        import imageio_ffmpeg
    except Exception as exc:
        raise StageExecutionError(
            "Audio probing requires the `imageio-ffmpeg` package. Install a Stage 1 dependency group first."
        ) from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg_exe, "-i", audio_path.as_posix(), "-f", "null", "-"]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    duration_seconds = parse_ffmpeg_duration(stderr_text)
    if duration_seconds is None:
        raise StageExecutionError(
            f"Failed to determine duration for {repo_relative(audio_path)} from ffmpeg probe output."
        )
    return AudioProbe(
        source_audio=repo_relative(audio_path),
        duration_seconds=duration_seconds,
        file_size_bytes=audio_path.stat().st_size,
    )


def decode_audio_to_mono_16k(
    audio_path: Path,
    start_seconds: Optional[float] = None,
    duration_seconds: Optional[float] = None,
) -> np.ndarray:
    try:
        import imageio_ffmpeg
    except Exception as exc:
        raise StageExecutionError(
            "Audio decoding requires the `imageio-ffmpeg` package. Install a Stage 1 dependency group first, "
            "for example `uv sync --group stage1-faster-whisper` or `uv sync --group stage1-parakeet`."
        ) from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-v",
        "error",
        "-i",
        audio_path.as_posix(),
    ]
    if start_seconds is not None:
        command.extend(["-ss", f"{max(0.0, start_seconds):.3f}"])
    if duration_seconds is not None:
        command.extend(["-t", f"{max(0.0, duration_seconds):.3f}"])
    command.extend(
        [
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(STAGE1_TRANSCRIPT_SAMPLE_RATE),
        "-",
        ]
    )
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        message = stderr or "ffmpeg returned a non-zero exit code while decoding audio."
        raise StageExecutionError(f"Failed to decode audio from {repo_relative(audio_path)}: {message}")

    raw_audio = np.frombuffer(result.stdout, dtype=np.int16)
    if not raw_audio.size:
        raise StageExecutionError(f"Decoded no audio samples from {repo_relative(audio_path)}")

    audio = raw_audio.astype(np.float32) / np.iinfo(np.int16).max
    return np.clip(audio, -1.0, 1.0)


def parakeet_runtime_available() -> bool:
    try:
        __import__("torch")
        __import__("transformers")
        return True
    except Exception:
        return False


def parakeet_model_available(model_dir: Path) -> bool:
    return (model_dir / "config.json").exists() and (model_dir / "preprocessor_config.json").exists()


def ensure_local_parakeet_model(
    model_name: str,
    model_dir: Path,
    allow_download: bool,
) -> Path:
    if parakeet_model_available(model_dir):
        return model_dir

    if not allow_download:
        raise StageExecutionError(
            "Parakeet model files are not available locally. "
            f"Expected them under {repo_relative(model_dir)}. "
            "Run `uv run chronicle models fetch parakeet` or rerun with `--allow-download`."
        )

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise StageExecutionError(
            "Downloading the Parakeet model requires the `huggingface-hub` package in the current environment."
        ) from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=model_dir.as_posix(),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    if not parakeet_model_available(model_dir):
        raise StageExecutionError(
            f"Parakeet model download completed, but the expected files were still missing under {repo_relative(model_dir)}."
        )
    return model_dir


def build_parakeet_pipeline(
    model_dir: Path,
    device: str,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCTC, AutoProcessor, pipeline

    suppressed_stdout = io.StringIO()
    with contextlib.redirect_stdout(suppressed_stdout):
        processor = AutoProcessor.from_pretrained(
            model_dir.as_posix(),
            local_files_only=True,
        )
        model = AutoModelForCTC.from_pretrained(
            model_dir.as_posix(),
            local_files_only=True,
        )
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )

    processor_sample_rate = int(processor.feature_extractor.sampling_rate)
    if processor_sample_rate != STAGE1_TRANSCRIPT_SAMPLE_RATE:
        raise StageExecutionError(
            "Parakeet processor sampling rate mismatch: "
            f"expected {STAGE1_TRANSCRIPT_SAMPLE_RATE}, got {processor_sample_rate}."
        )
    return processor, asr_pipeline


def resolve_stage1_backend(
    backend: str,
    manifest: SessionManifest,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    requested_preference = (manifest.stage1_model_preference or "").strip().lower()
    if backend != "auto":
        return backend, notes

    if requested_preference.startswith("parakeet"):
        if parakeet_runtime_available():
            return "parakeet", notes
        notes.append(
            "Session manifest prefers Parakeet, but the local runtime does not have the required "
            "Parakeet dependencies installed; falling back to faster-whisper."
        )
    return "faster-whisper", notes


def transcribe_with_faster_whisper(
    audio: np.ndarray,
    language: str,
    model_name: str,
    device: str,
    compute_type: str,
    cpu_threads: int,
    local_files_only: bool,
    download_root: Path,
    vad_filter: bool,
    beam_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        download_root=download_root.as_posix(),
        local_files_only=local_files_only,
    )
    segments_iter, info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=True,
    )

    segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments_iter, 1):
        segments.append(
            {
                "segment_id": index,
                "start": format_timestamp(float(segment.start)),
                "end": format_timestamp(float(segment.end)),
                "text": normalize_text(segment.text),
                "avg_logprob": getattr(segment, "avg_logprob", None),
                "no_speech_prob": getattr(segment, "no_speech_prob", None),
                "compression_ratio": getattr(segment, "compression_ratio", None),
            }
        )

    model_info = {
        "family": "faster-whisper",
        "name": model_name,
        "runtime": "faster-whisper",
        "device": device,
        "version": package_version("faster-whisper"),
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "local_files_only": local_files_only,
        "download_root": download_root.as_posix(),
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "language_probability": getattr(info, "language_probability", None),
        "duration_seconds": getattr(info, "duration", None),
        "duration_after_vad_seconds": getattr(info, "duration_after_vad", None),
    }
    return model_info, segments, []


def transcribe_with_parakeet(
    audio: np.ndarray,
    language: str,
    model_name: str,
    model_dir: Path,
    device: str,
    chunk_length_s: int,
    overlap_stride_s: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not parakeet_runtime_available():
        raise StageExecutionError(
            "Parakeet backend requested, but `torch` and `transformers` are not installed in the current environment."
        )

    if chunk_length_s <= 0:
        raise StageExecutionError("Parakeet chunk length must be a positive integer.")
    if overlap_stride_s is not None and overlap_stride_s <= 0:
        raise StageExecutionError("Parakeet overlap stride must be positive when provided.")
    if overlap_stride_s is not None and overlap_stride_s >= chunk_length_s:
        raise StageExecutionError("Parakeet overlap stride must be smaller than the chunk length.")

    _, asr_pipeline = build_parakeet_pipeline(model_dir=model_dir, device=device)

    chunk_samples = chunk_length_s * STAGE1_TRANSCRIPT_SAMPLE_RATE
    stride_samples = int((overlap_stride_s or chunk_length_s) * STAGE1_TRANSCRIPT_SAMPLE_RATE)
    if stride_samples <= 0:
        raise StageExecutionError("Parakeet stride configuration produced a non-positive sample count.")
    if len(audio) <= chunk_samples:
        chunk_count = 1
    else:
        chunk_count = 1 + math.ceil((len(audio) - chunk_samples) / stride_samples)

    segments: list[dict[str, Any]] = []
    word_timestamps: list[dict[str, Any]] = []
    unk_chunk_count = 0
    empty_chunk_count = 0
    start_sample = 0
    for index in range(chunk_count):
        end_sample = min(len(audio), start_sample + chunk_samples)
        chunk_audio = audio[start_sample:end_sample]
        if chunk_audio.size == 0:
            continue

        result = asr_pipeline(chunk_audio)
        raw_text = str(result.get("text", "")).strip()
        text = normalize_text(raw_text)
        decode_status = "ok"
        if not text:
            decode_status = "empty"
            empty_chunk_count += 1
        elif text == "<unk>":
            decode_status = "unk"
            unk_chunk_count += 1

        segments.append(
            {
                "segment_id": len(segments) + 1,
                "start": format_timestamp(start_sample / STAGE1_TRANSCRIPT_SAMPLE_RATE),
                "end": format_timestamp(end_sample / STAGE1_TRANSCRIPT_SAMPLE_RATE),
                "text": text,
                "decode_status": decode_status,
            }
        )
        if progress_callback is not None:
            progress_callback(len(segments), chunk_count, end_sample / STAGE1_TRANSCRIPT_SAMPLE_RATE)
        if end_sample >= len(audio):
            break
        start_sample += stride_samples

    notes = [
        "Parakeet transcription is using fixed-size chunk windows with chunk-level timestamps.",
        "This backend is using the Transformers automatic-speech-recognition pipeline per chunk because that was more reliable than direct long-window generation during local validation.",
    ]
    if overlap_stride_s is not None:
        notes.append(
            f"Experimental overlap mode is enabled with {chunk_length_s}s windows and {overlap_stride_s}s stride."
        )
    if unk_chunk_count:
        notes.append(
            f"Parakeet returned `<unk>` for {unk_chunk_count} chunk(s); those segments were preserved for manual review."
        )
    if empty_chunk_count:
        notes.append(
            f"Parakeet returned empty text for {empty_chunk_count} chunk(s); those segments were preserved for manual review."
        )

    model_info = {
        "family": "parakeet",
        "name": model_name,
        "model_dir": repo_relative(model_dir),
        "runtime": "transformers-asr-pipeline",
        "device": device,
        "version": package_version("transformers"),
        "language": language,
        "chunk_length_s": chunk_length_s,
        "overlap_stride_s": overlap_stride_s,
        "timestamp_mode": "chunk",
        "notes": notes,
    }
    return model_info, segments, word_timestamps


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    return f"{value:.1f}{units[unit_index]}"


def format_duration_summary(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def build_stage1_audio_summary(audio_probes: list[AudioProbe]) -> str:
    total_duration = sum(probe.duration_seconds for probe in audio_probes)
    total_size = sum(probe.file_size_bytes for probe in audio_probes)
    lines = [
        f"- Audio files: {len(audio_probes)}",
        f"- Total duration: {format_duration_summary(total_duration)}",
        f"- Total size: {format_bytes(total_size)}",
    ]
    for probe in audio_probes:
        lines.append(
            f"- {probe.source_audio}: {format_duration_summary(probe.duration_seconds)} | {format_bytes(probe.file_size_bytes)}"
        )
    return "\n".join(lines)


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


def parse_timestamp_seconds(value: Optional[str]) -> Optional[float]:
    if not value or not isinstance(value, str):
        return None
    try:
        hours_text, minutes_text, seconds_text = value.split(":")
        seconds_part, millis_part = seconds_text.split(".")
        return (
            int(hours_text) * 3600
            + int(minutes_text) * 60
            + int(seconds_part)
            + int(millis_part) / 1000.0
        )
    except (ValueError, AttributeError):
        return None


def write_stage1_markdown(
    path: Path,
    manifest: SessionManifest,
    artifact: dict[str, Any],
) -> None:
    source_audio_files = artifact.get("source_audio_files") or []
    lines = [
        "# Raw Transcript",
        "",
        f"- **Session ID:** {manifest.session_id}",
        f"- **Source audio files:** {len(source_audio_files)}",
        f"- **Language:** {artifact['language']}",
        f"- **Model family:** {artifact['model']['family']}",
        f"- **Model name:** {artifact['model']['name']}",
        f"- **Runtime:** {artifact['model']['runtime']}",
        f"- **Device:** {artifact['model']['device']}",
        "",
        "## Transcript",
        "",
    ]

    current_audio: Optional[str] = None
    for segment in artifact["segments"]:
        source_audio = segment.get("source_audio")
        if source_audio != current_audio:
            current_audio = source_audio
            if current_audio:
                lines.append(f"### {current_audio}")
                lines.append("")
        start = segment.get("start")
        decode_status = segment.get("decode_status")
        text = segment.get("text") or "[inaudible]"
        if decode_status == "unk":
            text = "`<unk>` [needs review]"
        elif decode_status == "empty":
            text = "[needs review: empty transcription chunk]"
        if start:
            lines.append(f"[{start}] {text}")
        else:
            lines.append(text)
        lines.append("")

    lines.extend(["## Notes", ""])
    lines.append("- Raw transcription only. No speaker assignment has been performed.")
    if artifact.get("notes"):
        for note in artifact["notes"]:
            lines.append(f"- {note}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def session_stage1_output_paths(stage_dir: Path) -> tuple[Path, Path]:
    return (
        stage_dir / "raw_transcript.json",
        stage_dir / "raw_transcript.md",
    )


def legacy_stage1_output_paths(stage_dir: Path, audio_file: str) -> tuple[Path, Path]:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(audio_file).stem).strip("-") or "audio"
    return (
        stage_dir / f"{stem}.raw_transcript.json",
        stage_dir / f"{stem}.raw_transcript.md",
    )


def build_session_stage1_artifact(
    manifest: SessionManifest,
    audio_artifacts: list[dict[str, Any]],
    model_info: dict[str, Any],
    notes: list[str],
) -> dict[str, Any]:
    combined_segments: list[dict[str, Any]] = []
    combined_word_timestamps: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    audio_summaries: list[dict[str, Any]] = []
    offset_seconds = 0.0

    for audio_artifact in audio_artifacts:
        source_audio = str(audio_artifact.get("source_audio", "")).strip()
        segments = audio_artifact.get("segments") or []
        transcript_text = str(audio_artifact.get("transcript_text", "")).strip()
        if transcript_text:
            transcript_parts.append(transcript_text)

        max_local_end = 0.0
        first_global_start: Optional[str] = None
        last_global_end: Optional[str] = None
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            local_start_seconds = parse_timestamp_seconds(segment.get("start"))
            local_end_seconds = parse_timestamp_seconds(segment.get("end"))
            global_start = (
                format_timestamp(offset_seconds + local_start_seconds)
                if local_start_seconds is not None
                else None
            )
            global_end = (
                format_timestamp(offset_seconds + local_end_seconds)
                if local_end_seconds is not None
                else None
            )
            if first_global_start is None and global_start is not None:
                first_global_start = global_start
            if global_end is not None:
                last_global_end = global_end
            if local_end_seconds is not None:
                max_local_end = max(max_local_end, local_end_seconds)
            elif local_start_seconds is not None:
                max_local_end = max(max_local_end, local_start_seconds)

            combined_segments.append(
                {
                    "segment_id": len(combined_segments) + 1,
                    "start": global_start,
                    "end": global_end,
                    "text": segment.get("text"),
                    "decode_status": segment.get("decode_status"),
                    "avg_logprob": segment.get("avg_logprob"),
                    "no_speech_prob": segment.get("no_speech_prob"),
                    "compression_ratio": segment.get("compression_ratio"),
                    "source_audio": source_audio,
                    "source_segment_id": segment.get("segment_id"),
                    "source_start": segment.get("start"),
                    "source_end": segment.get("end"),
                }
            )

        audio_summaries.append(
            {
                "source_audio": source_audio,
                "segment_count": len([segment for segment in segments if isinstance(segment, dict)]),
                "offset_start": format_timestamp(offset_seconds),
                "offset_end": format_timestamp(offset_seconds + max_local_end),
                "transcript_char_count": len(transcript_text),
                "global_first_segment_start": first_global_start,
                "global_last_segment_end": last_global_end,
            }
        )
        offset_seconds += max_local_end

        for word_timestamp in audio_artifact.get("word_timestamps") or []:
            if not isinstance(word_timestamp, dict):
                continue
            combined_word_timestamps.append(word_timestamp)

    artifact_notes = list(notes)
    artifact_notes.append(
        "Session-level timestamps are sequential across source audio files and are offset using each file's local segment timings."
    )

    return {
        "stage": "stage1_transcription",
        "session_id": manifest.session_id,
        "source_audio_files": [summary["source_audio"] for summary in audio_summaries],
        "audio_files": audio_summaries,
        "normalized_audio": None,
        "model": model_info,
        "language": manifest.language,
        "transcript_text": "\n".join(part for part in transcript_parts if part).strip(),
        "segments": combined_segments,
        "word_timestamps": combined_word_timestamps,
        "notes": artifact_notes,
    }


def load_existing_audio_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StageExecutionError(f"Invalid Stage 1 artifact format: {repo_relative(path)}")
    return payload


def execute_stage1(
    manifest: SessionManifest,
    stage_dir: Path,
    participants_file: Path,
    force: bool,
    backend: str,
    model_name: Optional[str],
    device: str,
    compute_type: str,
    cpu_threads: int,
    local_files_only: bool,
    download_root: Path = DEFAULT_STAGE1_CACHE_DIR,
    beam_size: int = 5,
    vad_filter: bool = True,
    parakeet_chunk_length_s: int = 15,
    parakeet_overlap_stride_s: Optional[float] = None,
    parakeet_model_dir: Path = DEFAULT_PARAKEET_MODEL_DIR,
    console: Optional[Console] = None,
) -> tuple[list[str], list[str], list[str]]:
    resolved_backend, backend_notes = resolve_stage1_backend(backend, manifest)
    if backend_notes and console is not None:
        console.print(Panel("\n".join(f"- {note}" for note in backend_notes), title="Stage 1 Notes"))

    if resolved_backend == "faster-whisper":
        resolved_model_name = model_name or DEFAULT_STAGE1_FASTER_WHISPER_MODEL
    elif resolved_backend == "parakeet":
        resolved_model_name = model_name or DEFAULT_STAGE1_PARAKEET_MODEL
    else:
        raise StageExecutionError(
            f"Unsupported stage 1 backend: {resolved_backend}. Expected one of: auto, faster-whisper, parakeet."
        )

    generated_paths: list[str] = []
    skipped_paths: list[str] = []
    notes: list[str] = list(backend_notes)
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

    parakeet_local_model_dir: Optional[Path] = None
    if resolved_backend == "parakeet":
        parakeet_local_model_dir = ensure_local_parakeet_model(
            model_name=resolved_model_name,
            model_dir=parakeet_model_dir,
            allow_download=not local_files_only,
        )
        notes.append(f"Using local Parakeet model files from {repo_relative(parakeet_local_model_dir)}.")

    processed_session_seconds = 0.0
    stage_started_at = time.perf_counter()
    progress_context = contextlib.nullcontext()
    overall_progress = None
    progress_task_id = None
    if resolved_backend == "parakeet" and console is not None:
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

            if console is not None:
                with console.status(f"Decoding {audio_label} to mono 16 kHz audio..."):
                    audio = decode_audio_to_mono_16k(audio_path)
            else:
                audio = decode_audio_to_mono_16k(audio_path)
            audio_duration_seconds = len(audio) / STAGE1_TRANSCRIPT_SAMPLE_RATE

            if resolved_backend == "faster-whisper":
                if console is not None:
                    with console.status(f"Transcribing {audio_label} with {resolved_backend}:{resolved_model_name}..."):
                        model_info, segments, word_timestamps = transcribe_with_faster_whisper(
                            audio=audio,
                            language=manifest.language,
                            model_name=resolved_model_name,
                            device=device,
                            compute_type=compute_type,
                            cpu_threads=cpu_threads,
                            local_files_only=local_files_only,
                            download_root=download_root,
                            vad_filter=vad_filter,
                            beam_size=beam_size,
                        )
                else:
                    model_info, segments, word_timestamps = transcribe_with_faster_whisper(
                        audio=audio,
                        language=manifest.language,
                        model_name=resolved_model_name,
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=cpu_threads,
                        local_files_only=local_files_only,
                        download_root=download_root,
                        vad_filter=vad_filter,
                        beam_size=beam_size,
                    )
            else:
                def report_progress(completed_chunks: int, total_chunks: int, processed_audio_seconds: float) -> None:
                    if overall_progress is None or progress_task_id is None:
                        return
                    completed_session_seconds = min(
                        total_session_duration,
                        processed_session_seconds + min(processed_audio_seconds, audio_duration_seconds),
                    )
                    elapsed = max(time.perf_counter() - stage_started_at, 0.001)
                    throughput = completed_session_seconds / elapsed
                    remaining = max(total_session_duration - completed_session_seconds, 0.0)
                    eta_seconds = remaining / throughput if throughput > 0 else None
                    throughput_text = (
                        f"{throughput / 60.0:.2f} audio-min/wall-min | "
                        f"{completed_chunks}/{total_chunks} chunks | "
                        f"ETA {format_duration_summary(eta_seconds or 0.0)}"
                        if eta_seconds is not None
                        else f"{completed_chunks}/{total_chunks} chunks"
                    )
                    overall_progress.update(
                        progress_task_id,
                        completed=completed_session_seconds,
                        current_file=audio_path.name,
                        throughput=throughput_text,
                    )

                model_info, segments, word_timestamps = transcribe_with_parakeet(
                    audio=audio,
                    language=manifest.language,
                    model_name=resolved_model_name,
                    model_dir=parakeet_local_model_dir or DEFAULT_PARAKEET_MODEL_DIR,
                    device=device,
                    chunk_length_s=parakeet_chunk_length_s,
                    overlap_stride_s=parakeet_overlap_stride_s,
                    progress_callback=report_progress,
                )

            transcript_text = "\n".join(
                segment["text"] for segment in segments if segment.get("text")
            ).strip()
            artifact_notes = list(notes)
            artifact_notes.extend(model_info.get("notes", []))
            if resolved_backend != "parakeet" and (manifest.stage1_model_preference or "").lower().startswith("parakeet"):
                artifact_notes.append(
                    "This run used the faster-whisper compatibility backend because Parakeet is not installed in the current environment."
                )

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
        model_info=session_model_info or {
            "family": resolved_backend,
            "name": resolved_model_name,
            "runtime": resolved_backend,
            "device": device,
        },
        notes=session_artifact_notes,
    )
    write_json(session_json_path, session_artifact)
    write_stage1_markdown(session_markdown_path, manifest, session_artifact)
    generated_paths.extend([repo_relative(session_json_path), repo_relative(session_markdown_path)])

    return generated_paths, skipped_paths, notes


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
