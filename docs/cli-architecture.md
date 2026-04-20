# CLI Architecture

This document explains how the `chronicle` command is structured in code and how the CLI modules relate to the stage implementations.

## Entry point

The Python package entrypoint is declared in `pyproject.toml`:

- `chronicle = "chronicle.cli:main"`

That resolves to:

- `src/chronicle/cli/__init__.py`
- `src/chronicle/cli/app.py`

`app.py` constructs the top-level Typer application and registers command modules.

## CLI module layout

### `src/chronicle/cli/app.py`

Purpose:
- create the top-level Typer app
- register each command group module

This file should stay thin. It is the assembly point, not the implementation point.

### `src/chronicle/cli/common.py`

Purpose:
- shared `rich` console
- overwrite confirmation prompt
- chronicle symlink install helper
- shared stage plan rendering used by multiple commands

This is the small shared surface used across command modules.

### `src/chronicle/cli/validate.py`

Purpose:
- implement `chronicle validate <session_id>`

This command delegates to session validation logic in `src/chronicle/session.py`.

### `src/chronicle/cli/init.py`

Purpose:
- implement `chronicle init`
- implement `chronicle models fetch parakeet`

This module is the operator-facing bootstrap layer.
It calls Stage 1 Parakeet model-management code but does not perform transcription itself.

### `src/chronicle/cli/stage1.py`

Purpose:
- implement `chronicle transcribe <session_id>`

This module:
- validates the session
- renders the Stage 1 plan
- handles overwrite prompting
- calls `execute_stage1(...)`
- writes run metadata

It is command orchestration only. The actual transcription logic lives under `src/chronicle/stage1/`.

### `src/chronicle/cli/benchmark.py`

Purpose:
- implement `chronicle benchmark-stage1`
- implement `chronicle benchmark-stage1-concurrency`
- implement `chronicle benchmark-stage2`

This is the operator-facing wrapper around Stage 1 and Stage 2 benchmark functions.

Current Stage 2 benchmark role:
- expose backend comparison through `--backend`
- currently support:
  - `pyannote`
  - `speechbrain`
- orchestrate separate Stage 2 runtimes during backend evaluation

### `src/chronicle/cli/stage2.py`

Purpose:
- currently implement `chronicle diarize <session_id>`

Current state:
- validate
- render plan
- handle overwrite prompting
- call the Stage 2 service
- write run metadata and completion status

Migration direction:
- this module should keep owning `chronicle diarize <session_id>`
- and it now calls the current SpeechBrain-backed anonymous audio-diarization implementation

### `src/chronicle/cli/stage3.py`

Purpose:
- currently implement `chronicle identify <session_id>`
- currently implement `chronicle run <session_id>`

Current state:
- this module now owns Stage 3 speaker identification
- it requires Stage 1 and Stage 2 artifacts
- it exposes `--mode llm`, `--mode manual`, and `--mode align-only`
- it accepts `--model` for Ollama model selection in `llm` mode
- it accepts `--speaker-map` for manual overrides or complete manual assignment
- it also exposes the `run` status command

Stage 3 CLI behavior:
- default `llm` mode requires local Ollama and sends compact evidence/context metadata to the local Ollama API
- `manual` mode runs locally when a complete valid speaker map is supplied
- `align-only` mode runs locally and writes anonymous alignment artifacts rather than final identity artifacts
- raw/private Stage 3 evidence stays on the user's machine

### `src/chronicle/cli/stage4.py`

Purpose:
- implement `chronicle organize <session_id>`

Current state:
- this module is lightweight because Stage 4 is scaffold-only
- it validates the session and prepares Stage 4 output locations

## How the CLI relates to the rest of the package

The CLI layer should do four things only:

1. parse command-line arguments
2. validate/prep the operator-facing workflow
3. call stage/service functions
4. render status and write run metadata

The CLI layer should not own core transformation logic.

That logic belongs in:
- `src/chronicle/session.py`
- `src/chronicle/stage1/`
- `src/chronicle/stage2/`
- later `src/chronicle/stage3/`
- later `src/chronicle/stage4/`

## What happens when someone runs `chronicle transcribe`

High-level flow:

1. The shell resolves the `chronicle` executable.
2. The CLI entrypoint calls `chronicle.cli.main()`.
3. `src/chronicle/cli/app.py` routes the command to `src/chronicle/cli/stage1.py`.
4. `stage1.py` validates the session using `require_valid_session(...)`.
5. `stage1.py` prepares output directories with `ensure_output_dirs(...)`.
6. `stage1.py` renders the Stage 1 plan in the terminal.
7. `stage1.py` checks whether outputs already exist and prompts if overwrite is needed.
8. `stage1.py` calls `src/chronicle/stage1/service.py:execute_stage1(...)`.
9. `execute_stage1(...)` performs the actual Stage 1 orchestration.
10. Control returns to `stage1.py`, which writes run metadata and prints completion status.

The CLI command does not directly decode audio, load Parakeet, or build transcript artifacts.
It delegates those responsibilities to the Stage 1 package.

## Current mismatch worth knowing

Chronicle is still in a transition period, but the command surface is now aligned with the 4-stage model:
- `chronicle diarize` is now implemented with the current local SpeechBrain-backed Stage 2 backend
- `chronicle benchmark-stage2` is the real current entrypoint for Stage 2 backend evaluation
- `chronicle identify` is current Stage 3 and reconciles Stage 1 transcript artifacts with Stage 2 diarization artifacts
- `chronicle organize` is Stage 4 naming and is scaffold-only

The next implementation step is real-session evaluation of Stage 3 outputs, then Stage 4 organization.
