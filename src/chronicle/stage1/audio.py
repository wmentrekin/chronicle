"""Audio probing and decoding utilities for Stage 1."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..exceptions import StageExecutionError
from ..paths import repo_relative


STAGE1_TRANSCRIPT_SAMPLE_RATE = 16000


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
            "Audio decoding requires the `imageio-ffmpeg` package. Install the Stage 1 Parakeet dependency group first, "
            "for example `uv sync --group stage1-parakeet`."
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
