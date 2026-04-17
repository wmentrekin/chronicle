"""Stage 2 anonymous audio diarization orchestration."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from ..exceptions import StageExecutionError
from ..paths import REPO_ROOT, repo_relative
from ..session import SessionManifest, resolve_audio_path
from ..stage1.audio import decode_audio_to_mono_16k, probe_audio_file
from .artifacts import stage2_output_paths, write_stage2_artifacts
from .benchmark import (
    DEFAULT_STAGE2_SPEECHBRAIN_EMBEDDING_MODEL,
    DEFAULT_STAGE2_SPEECHBRAIN_VAD_MODEL,
    write_pcm16_wav,
)


DEFAULT_STAGE2_BACKEND = "speechbrain"
DEFAULT_STAGE2_SPEECHBRAIN_PYTHON = REPO_ROOT / ".venv-stage2-speechbrain" / "bin" / "python"


def execute_stage2(
    *,
    manifest: SessionManifest,
    stage2_dir: Path,
    force: bool,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    device: str,
    stage2_python: Path,
    vad_model_name: str,
    embedding_model_name: str,
) -> tuple[list[str], list[str], list[str]]:
    json_path, markdown_path = stage2_output_paths(stage2_dir)
    existing_paths = [path for path in (json_path, markdown_path) if path.exists()]
    if existing_paths and not force:
        return [], [repo_relative(path) for path in existing_paths], []

    if not stage2_python.exists():
        raise StageExecutionError(
            f"Stage 2 SpeechBrain runtime not found at {stage2_python}. "
            "Create it first or pass `--stage2-python`."
        )

    audio_artifacts: list[dict[str, Any]] = []
    combined_turns: list[dict[str, Any]] = []
    notes: list[str] = []
    session_offset_seconds = 0.0
    total_wall_seconds = 0.0
    total_run_seconds = 0.0
    total_load_seconds = 0.0

    for audio_index, audio_file in enumerate(manifest.audio_files, start=1):
        audio_path = resolve_audio_path(manifest, audio_file)
        probe = probe_audio_file(audio_path)
        runner_payload, sample_audio_path, wall_seconds = run_stage2_speechbrain_audio(
            audio_path=audio_path,
            duration_seconds=probe.duration_seconds,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            device=device,
            stage2_python=stage2_python,
            vad_model_name=vad_model_name,
            embedding_model_name=embedding_model_name,
        )

        audio_artifacts.append(
            {
                "audio_index": audio_index,
                "source_audio": repo_relative(audio_path),
                "duration_seconds": round(probe.duration_seconds, 3),
                "wall_seconds": wall_seconds,
                "load_seconds": runner_payload["load_seconds"],
                "run_seconds": runner_payload["run_seconds"],
                "speaker_labels": runner_payload["speakers"],
                "turn_count": runner_payload["turn_count"],
            }
        )

        total_wall_seconds += wall_seconds
        total_load_seconds += float(runner_payload["load_seconds"])
        total_run_seconds += float(runner_payload["run_seconds"])

        for turn in runner_payload["turns"]:
            combined_turns.append(
                {
                    "turn_id": len(combined_turns) + 1,
                    "speaker_label": turn["speaker_label"],
                    "source_audio": repo_relative(audio_path),
                    "source_start_seconds": round(float(turn["start"]), 3),
                    "source_end_seconds": round(float(turn["end"]), 3),
                    "session_start_seconds": round(session_offset_seconds + float(turn["start"]), 3),
                    "session_end_seconds": round(session_offset_seconds + float(turn["end"]), 3),
                }
            )

        session_offset_seconds += probe.duration_seconds
        try:
            sample_audio_path.unlink(missing_ok=True)
        except Exception:
            notes.append(f"Could not remove temporary Stage 2 audio sample `{repo_relative(sample_audio_path)}`.")

    speaker_labels = sorted({turn["speaker_label"] for turn in combined_turns})
    artifact = {
        "stage": "stage2_audio_diarization",
        "backend": DEFAULT_STAGE2_BACKEND,
        "session_id": manifest.session_id,
        "audio_files": audio_artifacts,
        "speaker_labels": speaker_labels,
        "turns": combined_turns,
        "constraints": {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        },
        "models": {
            "vad": vad_model_name,
            "embedding": embedding_model_name,
        },
        "runtime": {
            "device": device,
            "runtime_python": repo_relative(stage2_python),
            "wall_seconds": round(total_wall_seconds, 2),
            "load_seconds": round(total_load_seconds, 2),
            "run_seconds": round(total_run_seconds, 2),
        },
        "notes": notes,
    }
    written_paths = write_stage2_artifacts(stage_dir=stage2_dir, artifact=artifact)
    return [repo_relative(path) for path in written_paths], [], notes


def run_stage2_speechbrain_audio(
    *,
    audio_path: Path,
    duration_seconds: float,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    device: str,
    stage2_python: Path,
    vad_model_name: str,
    embedding_model_name: str,
) -> tuple[dict[str, Any], Path, float]:
    audio = decode_audio_to_mono_16k(audio_path, start_seconds=0.0, duration_seconds=duration_seconds)
    tmp_root = REPO_ROOT / "tmp" / "stage2"
    tmp_root.mkdir(parents=True, exist_ok=True)
    temp_handle = tempfile.NamedTemporaryFile(prefix=f"{audio_path.stem}.", suffix=".wav", dir=tmp_root, delete=False)
    sample_audio_path = Path(temp_handle.name)
    temp_handle.close()
    write_pcm16_wav(sample_audio_path, audio)

    runner_path = REPO_ROOT / "src" / "chronicle" / "stage2" / "speechbrain_spike_runner.py"
    command = [
        stage2_python.as_posix(),
        runner_path.as_posix(),
        "--audio",
        sample_audio_path.as_posix(),
        "--device",
        device,
        "--vad-model",
        vad_model_name,
        "--embedding-model",
        embedding_model_name,
    ]
    if num_speakers is not None:
        command.extend(["--num-speakers", str(num_speakers)])
    if min_speakers is not None:
        command.extend(["--min-speakers", str(min_speakers)])
    if max_speakers is not None:
        command.extend(["--max-speakers", str(max_speakers)])

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
        stderr = result.stderr.strip() or result.stdout.strip() or "Stage 2 SpeechBrain run failed."
        raise StageExecutionError(stderr)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise StageExecutionError("Stage 2 SpeechBrain runner returned invalid JSON.") from exc
    return payload, sample_audio_path, wall_seconds
