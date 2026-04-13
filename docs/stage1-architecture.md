# Stage 1 Architecture

This document explains how Stage 1 transcription is organized in code and what actually happens when `chronicle transcribe <session_id>` runs.

## Stage 1 goal

Stage 1 takes one session manifest plus one or more raw audio files and produces:

- a session-level raw transcript JSON artifact
- a session-level raw transcript Markdown artifact
- stage run metadata

Stage 1 does not identify speakers.
It is a speech-to-text stage only.

## Stage 1 module layout

### `src/chronicle/stage1/service.py`

Purpose:
- orchestrate the full Stage 1 run
- resolve the backend
- coordinate model loading, audio decoding, transcription calls, progress reporting, and artifact writing

This is the control-flow module for Stage 1.
It should stay much smaller than the backend/helper modules.

### `src/chronicle/stage1/audio.py`

Purpose:
- ffmpeg-based audio probing
- ffmpeg-based decode to mono 16 kHz float arrays
- sample-window dataclasses used by benchmarking

This module owns all low-level audio preprocessing needed by Stage 1.

Main responsibilities:
- `probe_audio_file(...)`
- `decode_audio_to_mono_16k(...)`
- `AudioProbe`
- `SampleWindow`

### `src/chronicle/stage1/parakeet.py`

Purpose:
- Parakeet runtime availability checks
- local model availability checks
- managed local model download
- Parakeet pipeline construction
- Parakeet transcription execution

Main responsibilities:
- `parakeet_runtime_available(...)`
- `ensure_local_parakeet_model(...)`
- `build_parakeet_pipeline(...)`
- `transcribe_with_parakeet(...)`
- `transcribe_with_parakeet_pipeline(...)`

This module is the Stage 1 default backend.

### `src/chronicle/stage1/faster_whisper.py`

Purpose:
- compatibility transcription backend using `faster-whisper`

This is intentionally isolated so the fallback backend does not clutter the default Parakeet path.

### `src/chronicle/stage1/artifacts.py`

Purpose:
- output path helpers
- human-readable summary formatting
- transcript markdown writing
- session-level artifact assembly from one or more audio-level transcription results
- loading legacy per-audio Stage 1 artifacts

Main responsibilities:
- `session_stage1_output_paths(...)`
- `legacy_stage1_output_paths(...)`
- `write_stage1_markdown(...)`
- `build_session_stage1_artifact(...)`
- `load_existing_audio_artifact(...)`

### `src/chronicle/stage1/benchmark.py`

Purpose:
- benchmarking and concurrency experiments

This module is intentionally separate from the production Stage 1 execution path.

Main responsibilities:
- sample-window planning
- chunk-size benchmark
- concurrency benchmark
- worker initialization for experimental multiprocessing

## End-to-end Stage 1 execution flow

When someone runs:

```bash
chronicle transcribe <session_id>
```

the actual Stage 1 flow is:

1. `src/chronicle/cli/stage1.py` validates the session and calls `execute_stage1(...)`.
2. `execute_stage1(...)` resolves the backend.
3. Stage 1 computes the canonical output paths:
   - `outputs/<session_id>/stage1/raw_transcript.json`
   - `outputs/<session_id>/stage1/raw_transcript.md`
4. If outputs already exist and overwrite is not approved, the run exits conservatively.
5. Stage 1 probes the session audio files using `audio.py` so it can:
   - show total duration
   - show total file size
   - configure progress reporting
6. If the backend is Parakeet:
   - Stage 1 ensures the local Chronicle-managed model exists
   - Stage 1 loads the Parakeet pipeline once for the session
7. For each audio file in the session:
   - decode to mono 16 kHz
   - transcribe with the selected backend
   - capture segments and model metadata
   - accumulate them into the session transcript
8. After all audio files are processed, Stage 1 builds one session-level artifact.
9. Stage 1 writes JSON and Markdown outputs.
10. The CLI writes separate run metadata into `outputs/<session_id>/runs/`.

## What data flows through Stage 1

### Inputs

Required:
- `inputs/sessions/<session_id>/session.yaml`
- `inputs/sessions/<session_id>/context.md`
- one or more audio files under `inputs/sessions/<session_id>/audio/`
- `inputs/global/participants.yaml`

Important note:
- Stage 1 does not use participant/context metadata to change transcript wording
- session validation still happens before Stage 1 runs

### Internal working data

Stage 1 transforms audio through these main forms:

1. probed file metadata
   - duration
   - file size

2. decoded waveform arrays
   - mono
   - 16 kHz
   - floating-point samples

3. backend transcript segments
   - start
   - end
   - text
   - decode status

4. session-level merged artifact
   - all transcript segments across all source audio files
   - sequential session timestamps

### Outputs

Session-level outputs:
- `raw_transcript.json`
- `raw_transcript.md`

Run metadata:
- `outputs/<session_id>/runs/stage1.<timestamp>.json`

## Current Parakeet execution model

Current defaults:
- backend: `parakeet`
- chunk length: `15s`
- batch size: `4`

Current runtime model:
- decode audio locally
- run Parakeet through the Transformers ASR pipeline
- use fixed chunk windows with chunk-level timestamps
- reuse one loaded Parakeet pipeline across all audio files in the session

Current design choice:
- keep Stage 1 serial on this machine
- do not make multiprocessing the default path

That choice came from benchmark results showing the current serial path is more efficient than the current CPU multiprocessing design.

## Why Stage 1 is split this way

The file split is intentional:

- `audio.py` isolates ffmpeg and waveform handling
- `parakeet.py` isolates the default backend
- `faster_whisper.py` isolates the compatibility backend
- `artifacts.py` isolates output formatting and assembly
- `benchmark.py` isolates experimental code from production orchestration
- `service.py` stays readable as the actual Stage 1 control flow

This is the code-organization model Chronicle prefers across the repo.
