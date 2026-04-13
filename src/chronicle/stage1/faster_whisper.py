"""faster-whisper Stage 1 backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..utils import format_timestamp, normalize_text, package_version


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
