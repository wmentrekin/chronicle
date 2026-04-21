# Stage 3 Identification

Status: current implementation

Stage 3 reconciles Stage 1 transcript segments with Stage 2 anonymous diarization and maps anonymous speakers to canonical people when justified.

## Inputs

- `outputs/<session_id>/stage1/raw_transcript.json`
- `outputs/<session_id>/stage2/diarization.json`
- `inputs/global/participants.yaml`
- `inputs/sessions/<session_id>/session.yaml`
- `inputs/sessions/<session_id>/context.md`
- optional `inputs/sessions/<session_id>/speaker-map.yaml`

## Outputs

- `outputs/<session_id>/stage3/identified_conversation.json`
- `outputs/<session_id>/stage3/identified_conversation.md`
- `outputs/<session_id>/stage3/aligned_transcript.json`
- `outputs/<session_id>/stage3/aligned_transcript.md`

## Current modes

- `llm` uses local Ollama and the default model is `qwen3:8b`.
- `manual` applies a complete canonical speaker map locally.
- `align-only` keeps speakers anonymous and writes alignment artifacts only.

## Current behavior

- Stage 3 aligns transcript blocks to anonymous diarization turns using source audio and timing overlap.
- It preserves Stage 1 wording instead of rewriting transcript content.
- It preserves Stage 2 anonymous provenance.
- `manual` speaker maps must use canonical participant names only.
- Stage 3 writes run metadata under `outputs/<session_id>/runs/`.

## What Stage 3 does not do

- It does not perform acoustic diarization.
- It does not transcribe audio.
- It does not require remote model services.

## Current limitations

- `llm` mode depends on local Ollama being available and responsive.
- On constrained CPU-only machines, the default model can be slow enough that `align-only` or `manual` mode is the safer choice.

