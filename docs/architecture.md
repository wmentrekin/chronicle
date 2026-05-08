# Chronicle Architecture

Status: current implementation

Chronicle is a local-first, multi-stage, filesystem-based pipeline for turning interview audio into reviewable source material.

## Core rules

- Inputs live under `inputs/`.
- Derived artifacts live under `outputs/<session_id>/`.
- Each stage only depends on its own inputs plus earlier stage outputs.
- Processing stays local by default.
- Stage outputs are deterministic file contracts rather than in-memory handoffs.

## Current pipeline

1. Stage 1 transcription
2. Stage 2 anonymous diarization
3. Stage 3 speaker identification
4. Stage 4 organization scaffold

## Repository layout

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
    stage4/
    runs/
models/
  parakeet-ctc-0.6b/
docs/
  architecture.md
  cli.md
  artifacts.md
  stage-1/
    transcription.md
    cloud-requirements.yaml
    cloud-research.md
    cloud-runbook.md
  stage-2/
    stage2-diarization.md
  stage-3/
    stage3-identification.md
  stage-4/
    stage4-organization.md
```

`inputs/`, `outputs/`, and `models/` are private working directories and are not committed.

## Current stage summary

- Stage 1 transcribes audio through the default cloud orchestration path that provisions a Parakeet worker.
- Stage 2 writes anonymous diarization artifacts with the local SpeechBrain-backed production path.
- Stage 3 reconciles Stage 1 and Stage 2, then maps anonymous speakers to canonical participants with `llm`, `manual`, or `align-only` modes.
- Stage 4 is scaffold-only today.

## Doc map

- `docs/cli.md`
- `docs/artifacts.md`
- `docs/stage-1/transcription.md`
- `docs/stage-2/stage2-diarization.md`
- `docs/stage-3/stage3-identification.md`
- `docs/stage-4/stage4-organization.md`

## Known limitations

- Stage 3 `llm` mode depends on local Ollama and can be slow on constrained CPU-only systems.
- Stage 4 is present in the CLI but not yet implemented as a production organizer.
