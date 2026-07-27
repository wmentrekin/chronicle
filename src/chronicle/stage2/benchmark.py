"""Stage 2 spike helpers."""

from __future__ import annotations

import json
import subprocess
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from ..exceptions import StageExecutionError
from ..paths import REPO_ROOT, repo_relative
from ..session import SessionManifest, resolve_audio_path
from ..stage1.audio import STAGE1_TRANSCRIPT_SAMPLE_RATE, decode_audio_to_mono_16k, probe_audio_file
from ..utils import slugify_stem


DEFAULT_STAGE2_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_STAGE2_BACKEND = "pyannote"
DEFAULT_STAGE2_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_STAGE2_SPEECHBRAIN_VAD_MODEL = "speechbrain/vad-crdnn-libriparty"
DEFAULT_STAGE2_SPEECHBRAIN_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def resolve_stage2_audio_file(manifest: SessionManifest, audio_file: Optional[str]) -> tuple[str, Path]:
    if audio_file is None:
        selected = manifest.audio_files[0]
    else:
        selected = audio_file

    if selected in manifest.audio_files:
        return selected, resolve_audio_path(manifest, selected)

    candidate = resolve_audio_path(manifest, selected)
    if candidate.exists():
        return selected, candidate

    raise StageExecutionError(
        f"Audio file `{selected}` is not part of session `{manifest.session_id}` and could not be resolved."
    )


def write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int = STAGE1_TRANSCRIPT_SAMPLE_RATE) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * np.iinfo(np.int16).max).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path.as_posix(), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


def stage2_spike_output_paths(
    spike_dir: Path,
    backend: str,
    audio_file: str,
    sample_start_seconds: float,
    sample_seconds: int,
) -> dict[str, Path]:
    slug = slugify_stem(Path(audio_file))
    sample_tag = f"{int(sample_start_seconds)}s-{int(sample_seconds)}s"
    base = f"{slug}.{sample_tag}"
    return {
        "sample_wav": spike_dir / f"{base}.sample.wav",
        "result_json": spike_dir / f"{base}.{backend}.json",
    }


def _run_stage2_runner(
    *,
    manifest: SessionManifest,
    backend: str,
    audio_file: Optional[str],
    sample_seconds: int,
    sample_start_seconds: float,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    device: str,
    stage2_python: Path,
    runner_path: Path,
    command_tail: list[str],
    spike_dir: Path,
    force: bool,
) -> dict[str, object]:
    selected_audio, audio_path = resolve_stage2_audio_file(manifest, audio_file)
    output_paths = stage2_spike_output_paths(spike_dir, backend, selected_audio, sample_start_seconds, sample_seconds)
    sample_wav_path = output_paths["sample_wav"]
    result_json_path = output_paths["result_json"]

    if not force and result_json_path.exists():
        existing = json.loads(result_json_path.read_text(encoding="utf-8"))
        existing["cached"] = True
        return existing

    if not stage2_python.exists():
        raise StageExecutionError(
            f"Stage 2 Python runtime not found at {stage2_python}. Create it before running the Stage 2 spike."
        )

    probe = probe_audio_file(audio_path)
    if sample_start_seconds < 0:
        raise StageExecutionError("`sample_start_seconds` must be non-negative.")
    if sample_start_seconds >= probe.duration_seconds:
        raise StageExecutionError(
            f"Sample start {sample_start_seconds}s is beyond the audio duration {probe.duration_seconds:.2f}s."
        )

    bounded_sample_seconds = min(sample_seconds, max(1, int(probe.duration_seconds - sample_start_seconds)))
    audio = decode_audio_to_mono_16k(
        audio_path,
        start_seconds=sample_start_seconds,
        duration_seconds=bounded_sample_seconds,
    )
    write_pcm16_wav(sample_wav_path, audio)

    command = [
        stage2_python.as_posix(),
        runner_path.as_posix(),
        "--audio",
        sample_wav_path.as_posix(),
        "--device",
        device,
    ] + command_tail

    started = time.perf_counter()
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        cwd=REPO_ROOT.as_posix(),
    )
    wall_seconds = round(time.perf_counter() - started, 2)

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "Stage 2 pyannote spike failed."
        raise StageExecutionError(stderr)

    try:
        runner_payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise StageExecutionError(f"Stage 2 {backend} runner returned invalid JSON.") from exc

    artifact = {
        "stage": "stage2_audio_diarization_spike",
        "backend": backend,
        "session_id": manifest.session_id,
        "source_audio": repo_relative(audio_path),
        "sample_audio": repo_relative(sample_wav_path),
        "sample_start_seconds": sample_start_seconds,
        "sample_seconds": bounded_sample_seconds,
        "runtime": {
            "device": device,
            "runtime_python": repo_relative(stage2_python),
        },
        "constraints": {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        },
        "wall_seconds": wall_seconds,
        "runner": runner_payload,
    }
    result_json_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def run_stage2_pyannote_spike(
    *,
    manifest: SessionManifest,
    audio_file: Optional[str],
    sample_seconds: int,
    sample_start_seconds: float,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    device: str,
    stage2_python: Path,
    model_name: str,
    spike_dir: Path,
    force: bool,
) -> dict[str, object]:
    command_tail = [
        "--model",
        model_name,
    ]
    if num_speakers is not None:
        command_tail.extend(["--num-speakers", str(num_speakers)])
    if min_speakers is not None:
        command_tail.extend(["--min-speakers", str(min_speakers)])
    if max_speakers is not None:
        command_tail.extend(["--max-speakers", str(max_speakers)])

    return _run_stage2_runner(
        manifest=manifest,
        backend="pyannote",
        audio_file=audio_file,
        sample_seconds=sample_seconds,
        sample_start_seconds=sample_start_seconds,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device,
        stage2_python=stage2_python,
        runner_path=REPO_ROOT / "src" / "chronicle" / "stage2" / "pyannote_spike_runner.py",
        command_tail=command_tail,
        spike_dir=spike_dir,
        force=force,
    )


def run_stage2_speechbrain_spike(
    *,
    manifest: SessionManifest,
    audio_file: Optional[str],
    sample_seconds: int,
    sample_start_seconds: float,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    device: str,
    stage2_python: Path,
    vad_model_name: str,
    embedding_model_name: str,
    spike_dir: Path,
    force: bool,
) -> dict[str, object]:
    command_tail = [
        "--vad-model",
        vad_model_name,
        "--embedding-model",
        embedding_model_name,
    ]
    if num_speakers is not None:
        command_tail.extend(["--num-speakers", str(num_speakers)])
    if min_speakers is not None:
        command_tail.extend(["--min-speakers", str(min_speakers)])
    if max_speakers is not None:
        command_tail.extend(["--max-speakers", str(max_speakers)])

    artifact = _run_stage2_runner(
        manifest=manifest,
        backend="speechbrain",
        audio_file=audio_file,
        sample_seconds=sample_seconds,
        sample_start_seconds=sample_start_seconds,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device,
        stage2_python=stage2_python,
        runner_path=REPO_ROOT / "src" / "chronicle" / "stage2" / "speechbrain_spike_runner.py",
        command_tail=command_tail,
        spike_dir=spike_dir,
        force=force,
    )
    artifact["models"] = {
        "vad": vad_model_name,
        "embedding": embedding_model_name,
    }
    output_paths = stage2_spike_output_paths(
        spike_dir,
        "speechbrain",
        resolve_stage2_audio_file(manifest, audio_file)[0],
        sample_start_seconds,
        sample_seconds,
    )
    output_paths["result_json"].write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact
