# Chronicle

Chronicle is a local-first, multi-stage agentic audio-processing workflow for turning interview recordings into reviewable source material through a CLI.

Current implemented stages:
- Stage 1 transcription
- Stage 2 semantic diarization and conservative speaker assignment
- Stage 3 chronology is still a scaffold

## Repository shape

```text
inputs/
  global/
    participants.yaml
  sessions/
    <session_id>/
      audio/
      session.yaml
      context.md
outputs/
  <session_id>/
    stage1/
    stage2/
    stage3/
    runs/
src/chronicle/
docs/
models/
examples/
```

`inputs/` and `outputs/` are intentionally gitignored because they can contain private source material.
`models/` is also gitignored by default because Chronicle can manage local model files there.
Sanitized scaffolds for a public setup live under `examples/inputs/`.

## Quick Start

From a fresh clone:

```bash
./bin/bootstrap
```

If you want a stable `chronicle` command without sourcing `.envrc`, use:

```bash
./bin/bootstrap --install-link
```

If you prefer the manual setup path:

```bash
uv sync
source .envrc
chronicle init
```

## Main Workflow

Create a local `inputs/` tree from the templates under `examples/inputs/`, then run:

```bash
chronicle init
chronicle validate <session_id>
chronicle transcribe <session_id>
chronicle diarize <session_id>
chronicle chronology <session_id>
chronicle run <session_id>
```

## Commands

Initialize the local model/runtime:

```bash
chronicle init
```

Validate a session bundle:

```bash
chronicle validate <session_id>
```

Run Stage 1 transcription:

```bash
chronicle transcribe <session_id>
```

Run Stage 2:

```bash
chronicle diarize <session_id>
```

Prepare Stage 3:

```bash
chronicle chronology <session_id>
```

Show current session status:

```bash
chronicle run <session_id>
```

## Notes

- Raw audio is expected under `inputs/sessions/<session_id>/audio/`.
- Session-specific metadata lives beside it in `session.yaml` and `context.md`.
- Canonical people metadata lives in `inputs/global/participants.yaml`.
- Public users should start from the sanitized scaffolds under `examples/inputs/` and create their own local `inputs/` tree from those templates.
- Stage 1 writes one session-level transcript artifact per session, even when the session contains multiple audio files.
- Stage 2 is transcript-driven semantic diarization, not audio-based speaker embedding or acoustic diarization.
- Detailed implementation walkthroughs live under `docs/`.
