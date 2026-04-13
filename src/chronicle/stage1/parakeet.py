"""Parakeet Stage 1 backend."""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..exceptions import StageExecutionError
from ..paths import repo_relative
from ..utils import format_timestamp, normalize_text, package_version
from .audio import STAGE1_TRANSCRIPT_SAMPLE_RATE


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


def transcribe_with_parakeet(
    audio: np.ndarray,
    language: str,
    model_name: str,
    model_dir: Path,
    device: str,
    chunk_length_s: int,
    batch_size: int = 1,
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
    return transcribe_with_parakeet_pipeline(
        audio=audio,
        language=language,
        model_name=model_name,
        model_dir=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        overlap_stride_s=overlap_stride_s,
        asr_pipeline=asr_pipeline,
        progress_callback=progress_callback,
    )


def transcribe_with_parakeet_pipeline(
    audio: np.ndarray,
    language: str,
    model_name: str,
    model_dir: Path,
    device: str,
    chunk_length_s: int,
    asr_pipeline: Any,
    batch_size: int = 1,
    overlap_stride_s: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if chunk_length_s <= 0:
        raise StageExecutionError("Parakeet chunk length must be a positive integer.")
    if batch_size <= 0:
        raise StageExecutionError("Parakeet batch size must be a positive integer.")
    if overlap_stride_s is not None and overlap_stride_s <= 0:
        raise StageExecutionError("Parakeet overlap stride must be positive when provided.")
    if overlap_stride_s is not None and overlap_stride_s >= chunk_length_s:
        raise StageExecutionError("Parakeet overlap stride must be smaller than the chunk length.")

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
    pending_batch: list[tuple[int, int, np.ndarray]] = []
    start_sample = 0
    while True:
        end_sample = min(len(audio), start_sample + chunk_samples)
        chunk_audio = audio[start_sample:end_sample]
        if chunk_audio.size:
            pending_batch.append((start_sample, end_sample, chunk_audio))

        should_flush = len(pending_batch) >= batch_size or end_sample >= len(audio)
        if should_flush and pending_batch:
            batch_audio = [chunk_audio for _, _, chunk_audio in pending_batch]
            batch_results = asr_pipeline(batch_audio, batch_size=batch_size)
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            for (chunk_start, chunk_end, _), result in zip(pending_batch, batch_results, strict=False):
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
                        "start": format_timestamp(chunk_start / STAGE1_TRANSCRIPT_SAMPLE_RATE),
                        "end": format_timestamp(chunk_end / STAGE1_TRANSCRIPT_SAMPLE_RATE),
                        "text": text,
                        "decode_status": decode_status,
                    }
                )
                if progress_callback is not None:
                    progress_callback(len(segments), chunk_count, chunk_end / STAGE1_TRANSCRIPT_SAMPLE_RATE)
            pending_batch = []

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
        "batch_size": batch_size,
        "overlap_stride_s": overlap_stride_s,
        "timestamp_mode": "chunk",
        "notes": notes,
    }
    return model_info, segments, word_timestamps
