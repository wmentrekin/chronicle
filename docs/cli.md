# CLI

Status: current implementation

Chronicle is a filesystem-based CLI. Each command reads session inputs from `inputs/` and writes derived artifacts under `outputs/<session_id>/`.

## Command surface

- `chronicle init`
- `chronicle validate <session_id>`
- `chronicle transcribe <session_id>`
- `chronicle diarize <session_id>`
- `chronicle identify <session_id>`
- `chronicle organize <session_id>`
- `chronicle run <session_id>`
- `chronicle benchmark-stage1 <session_id>`
- `chronicle benchmark-stage1-concurrency <session_id>`
- `chronicle benchmark-stage2 <session_id>`
- `chronicle benchmark-stage3 <session_id>`
- `chronicle models fetch parakeet`

## What each command does

- `init` prepares local runtime dependencies. It can fetch Chronicle-managed Parakeet model files, validate or pull the default Stage 3 Ollama model, and install a stable `chronicle` symlink.
- `validate` checks a session manifest, its inputs, and the canonical participant list.
- `transcribe` runs Stage 1 transcription and writes `stage1/raw_transcript.json` plus `stage1/raw_transcript.md`.
- `diarize` runs Stage 2 anonymous diarization and writes `stage2/diarization.json` plus `stage2/diarization.md`.
- `identify` runs Stage 3 speaker identification or alignment and writes either `identified_conversation.*` or `aligned_transcript.*` depending on mode.
- `organize` currently only validates the session and prepares the Stage 4 output directory. It is scaffold-only.
- `run` validates the session and prints pipeline status for all stages.
- `benchmark-stage1` and `benchmark-stage1-concurrency` compare Stage 1 Parakeet settings.
- `benchmark-stage2` compares Stage 2 diarization backends.
- `benchmark-stage3` compares Stage 3 automatic identification backends against an explicit truth speaker map and writes JSON and Markdown reports under `outputs/<session_id>/runs/`.
- `models fetch parakeet` downloads the Chronicle-managed local Parakeet model directory.

## Current defaults and constraints

- Stage 1 prefers local Parakeet and can fall back to `faster-whisper` when required.
- Stage 2 uses the local SpeechBrain-backed diarization path in production.
- Stage 3 automatic identification still defaults to the local `ollama_decomposed` backend in `--mode llm`, with `qwen3:8b` as the default Ollama model where Ollama is used.
- `manual` and `align-only` modes do not use automatic backends.
- Stage 3 benchmark output is recommendation-only and does not switch the production default backend.
- Commands operate on one session at a time and keep all sensitive processing local by default.

## Important options

- `transcribe` accepts backend selection and Parakeet runtime/model options.
- `diarize` accepts speaker-count constraints, device selection, and the separate Stage 2 Python runtime path.
- `identify` accepts `--mode llm|manual|align-only`, `--backend`, `--model`, `--speaker-map`, and `--participants-file`.
- `identify --backend` supports `ollama_decomposed`, `speechbrain_refmatch`, and `speechbrain_hybrid` when automatic Stage 3 assignment is requested.
- `benchmark-stage3` requires `--truth-file` and accepts `--backends`, `--model`, `--cpu-note`, and `--participants-file`.
- `init` accepts model-management flags for Parakeet and Stage 3 Ollama setup.
