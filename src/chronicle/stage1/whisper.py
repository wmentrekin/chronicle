"""faster-whisper Stage 1 backend using CTranslate2 and Silero VAD."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from ..exceptions import StageExecutionError
from ..utils import format_timestamp, normalize_text, package_version


def faster_whisper_runtime_available() -> bool:
    try:
        __import__("faster_whisper")
        return True
    except Exception:
        return False


def transcribe_with_faster_whisper(
    audio_path: Path,
    language: str = "en",
    model_name: str = "large-v3-turbo",
    device: str = "cpu",
    compute_type: str = "int8",
    vad_filter: bool = True,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not faster_whisper_runtime_available():
        raise StageExecutionError(
            "faster-whisper backend requested, but `faster_whisper` package is not installed in the current environment. "
            "Run `uv sync --group stage1-whisper` to install it."
        )

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(
            model_size_or_path=model_name,
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        raise StageExecutionError(
            f"Failed to load faster-whisper model `{model_name}` on device `{device}` with compute_type `{compute_type}`: {exc}"
        ) from exc

    segments_gen, info = model.transcribe(
        audio_path.as_posix(),
        language=language if language and language != "auto" else None,
        vad_filter=vad_filter,
        word_timestamps=True,
    )

    segments: list[dict[str, Any]] = []
    word_timestamps: list[dict[str, Any]] = []
    empty_segment_count = 0

    for raw_segment in segments_gen:
        text = normalize_text(raw_segment.text)
        decode_status = "ok"
        if not text:
            decode_status = "empty"
            empty_segment_count += 1

        segment_dict = {
            "segment_id": len(segments) + 1,
            "start": format_timestamp(raw_segment.start),
            "end": format_timestamp(raw_segment.end),
            "text": text,
            "decode_status": decode_status,
            "avg_logprob": round(float(raw_segment.avg_logprob), 4) if hasattr(raw_segment, "avg_logprob") else None,
            "no_speech_prob": round(float(raw_segment.no_speech_prob), 4) if hasattr(raw_segment, "no_speech_prob") else None,
            "compression_ratio": round(float(raw_segment.compression_ratio), 4) if hasattr(raw_segment, "compression_ratio") else None,
        }
        segments.append(segment_dict)

        if raw_segment.words:
            for word_obj in raw_segment.words:
                word_timestamps.append(
                    {
                        "word": word_obj.word.strip(),
                        "start": format_timestamp(word_obj.start),
                        "end": format_timestamp(word_obj.end),
                        "probability": round(float(word_obj.probability), 4),
                        "segment_id": len(segments),
                    }
                )

        if progress_callback is not None:
            progress_callback(len(segments), 0, raw_segment.end)

    notes = [
        f"faster-whisper transcription using model `{model_name}` ({compute_type}).",
        f"Silero VAD filtering enabled: {vad_filter}.",
        f"Detected language: {info.language} (probability: {info.language_probability:.2f}).",
    ]
    if empty_segment_count:
        notes.append(
            f"faster-whisper returned empty text for {empty_segment_count} segment(s)."
        )

    model_info = {
        "family": "whisper",
        "name": model_name,
        "runtime": "faster-whisper (CTranslate2)",
        "device": device,
        "compute_type": compute_type,
        "version": package_version("faster-whisper"),
        "language": info.language or language,
        "vad_filter": vad_filter,
        "timestamp_mode": "segment_and_word",
        "notes": notes,
    }

    return model_info, segments, word_timestamps
