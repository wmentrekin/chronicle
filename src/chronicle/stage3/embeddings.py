"""Embedding extraction and cache helpers for Stage 3."""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..exceptions import StageExecutionError
from ..paths import OUTPUTS_ROOT, repo_relative
from ..stage1.audio import STAGE1_TRANSCRIPT_SAMPLE_RATE, decode_audio_to_mono_16k
from .enrollment import AudioSliceSpec

DEFAULT_STAGE3_SPEECHBRAIN_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

AudioDecoder = Callable[[Path, float | None, float | None], np.ndarray]
EmbeddingExtractor = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class CachedEmbeddingPaths:
    prepared_audio_path: Path
    embedding_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class CachedEmbeddingResult:
    cached: bool
    embedding: np.ndarray
    paths: CachedEmbeddingPaths
    cache_key: str


class SpeechBrainECAPAExtractor:
    """Lazy SpeechBrain ECAPA embedding extractor."""

    def __init__(self, *, model_name: str = DEFAULT_STAGE3_SPEECHBRAIN_MODEL, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._torch: Any | None = None
        self._encoder: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._torch is not None and self._encoder is not None:
            return self._torch, self._encoder

        try:
            import torch
        except Exception as exc:
            raise StageExecutionError(
                "Stage 3 SpeechBrain embeddings require `torch` in the current environment."
            ) from exc
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            try:
                from speechbrain.pretrained import EncoderClassifier
            except Exception as exc:
                raise StageExecutionError(
                    "Stage 3 SpeechBrain embeddings require the optional `stage2-speechbrain` dependency group."
                ) from exc

        self._torch = torch
        self._encoder = EncoderClassifier.from_hparams(
            source=self.model_name,
            run_opts={"device": self.device},
        )
        return self._torch, self._encoder

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != STAGE1_TRANSCRIPT_SAMPLE_RATE:
            raise StageExecutionError(
                f"Stage 3 SpeechBrain embeddings expect {STAGE1_TRANSCRIPT_SAMPLE_RATE} Hz mono audio."
            )
        torch, encoder = self._ensure_loaded()
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise StageExecutionError("Stage 3 embedding extraction requires non-empty mono audio samples.")
        batch = torch.from_numpy(waveform).unsqueeze(0)
        with torch.no_grad():
            embedding = encoder.encode_batch(batch).squeeze().detach().cpu().numpy()
        return np.asarray(embedding, dtype=np.float32)


def build_embedding_cache_key(*, model_name: str, slices: Sequence[AudioSliceSpec]) -> str:
    descriptors = sorted((item.as_cache_fragment() for item in slices), key=lambda value: json.dumps(value, sort_keys=True))
    payload = {"model_name": model_name, "slices": descriptors}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest[:24]


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    normalized = np.asarray(vector, dtype=np.float32)
    if normalized.ndim != 1 or normalized.size == 0:
        raise StageExecutionError("Stage 3 embedding extractor returned an invalid embedding vector.")
    norm = np.linalg.norm(normalized)
    if norm == 0:
        raise StageExecutionError("Stage 3 embedding extractor returned a zero-length embedding vector.")
    return normalized / norm


def _ensure_cache_root(cache_root: Path) -> None:
    try:
        cache_root.resolve().relative_to(OUTPUTS_ROOT.resolve())
    except ValueError as exc:
        raise StageExecutionError(
            f"Stage 3 embedding cache must live under outputs/. Received: {repo_relative(cache_root)}"
        ) from exc


def _write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int = STAGE1_TRANSCRIPT_SAMPLE_RATE) -> None:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = (clipped * np.iinfo(np.int16).max).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path.as_posix(), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


def _load_cached_embedding(embedding_path: Path) -> np.ndarray | None:
    if not embedding_path.exists():
        return None
    payload = json.loads(embedding_path.read_text(encoding="utf-8"))
    embedding = payload.get("embedding")
    if not isinstance(embedding, list):
        raise StageExecutionError(f"Invalid Stage 3 embedding cache payload: {repo_relative(embedding_path)}")
    return np.asarray(embedding, dtype=np.float32)


def prepare_cached_embedding(
    *,
    slices: Sequence[AudioSliceSpec],
    cache_root: Path,
    model_name: str,
    extractor: EmbeddingExtractor | None = None,
    audio_decoder: AudioDecoder = decode_audio_to_mono_16k,
) -> CachedEmbeddingResult:
    if not slices:
        raise StageExecutionError("Stage 3 embedding preparation requires at least one audio slice.")

    _ensure_cache_root(cache_root)
    cache_key = build_embedding_cache_key(model_name=model_name, slices=slices)
    paths = CachedEmbeddingPaths(
        prepared_audio_path=cache_root / f"{cache_key}.prepared.wav",
        embedding_path=cache_root / f"{cache_key}.embedding.json",
        metadata_path=cache_root / f"{cache_key}.metadata.json",
    )

    cached_embedding = _load_cached_embedding(paths.embedding_path)
    if cached_embedding is not None:
        return CachedEmbeddingResult(cached=True, embedding=cached_embedding, paths=paths, cache_key=cache_key)

    embedding_extractor = extractor or SpeechBrainECAPAExtractor(model_name=model_name)
    decoded_audio = [
        np.asarray(
            audio_decoder(item.audio_path, item.start_seconds, item.duration_seconds),
            dtype=np.float32,
        )
        for item in slices
    ]
    prepared_audio = np.concatenate(decoded_audio)
    _write_pcm16_wav(paths.prepared_audio_path, prepared_audio)

    embedding = _normalize_embedding(embedding_extractor(prepared_audio, STAGE1_TRANSCRIPT_SAMPLE_RATE))
    paths.embedding_path.parent.mkdir(parents=True, exist_ok=True)
    paths.embedding_path.write_text(
        json.dumps({"model_name": model_name, "cache_key": cache_key, "embedding": embedding.tolist()}, indent=2) + "\n",
        encoding="utf-8",
    )
    paths.metadata_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "model_name": model_name,
                "sample_rate": STAGE1_TRANSCRIPT_SAMPLE_RATE,
                "slices": [item.as_cache_fragment() for item in slices],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return CachedEmbeddingResult(cached=False, embedding=embedding, paths=paths, cache_key=cache_key)
