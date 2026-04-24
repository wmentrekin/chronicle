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
- optional participant `voice_references` entries in `inputs/global/participants.yaml` for enrollment-dependent local backends

## Outputs

- `outputs/<session_id>/stage3/identified_conversation.json`
- `outputs/<session_id>/stage3/identified_conversation.md`
- `outputs/<session_id>/stage3/aligned_transcript.json`
- `outputs/<session_id>/stage3/aligned_transcript.md`
- benchmark reports under `outputs/<session_id>/runs/stage3-benchmark.<timestamp>.json` and `.md` when `chronicle benchmark-stage3` is used

## Current modes

- `llm` is the automatic identification mode. It accepts `--backend` and defaults to the local `ollama_decomposed` backend.
- `manual` applies a complete canonical speaker map locally.
- `align-only` keeps speakers anonymous and writes alignment artifacts only.

## Automatic backends

- `ollama_decomposed` is the current default automatic backend. It uses local Ollama and can run without participant voice-reference clips.
- `speechbrain_refmatch` is a local enrollment-dependent backend. It compares anonymous speaker audio against participant `voice_references` and fails closed on missing enrollment, ambiguous matches, or low-confidence matches.
- `speechbrain_hybrid` starts from the same local SpeechBrain enrollment flow and uses local Ollama only for unresolved low-margin conflicts.

`voice_references` is optional participant metadata. When present, it must be a list of repo-relative or participants-file-relative audio clip paths. These clips are used only by the SpeechBrain-based backends.

## Current behavior

- Stage 3 aligns transcript blocks to anonymous diarization turns using source audio and timing overlap.
- It preserves Stage 1 wording instead of rewriting transcript content.
- It preserves Stage 2 anonymous provenance.
- `manual` speaker maps must use canonical participant names only.
- Automatic runs write backend metadata into the Stage 3 JSON artifact, including `backend` and `backend_usage`. `llm_usage` is present only when the selected backend actually calls Ollama.
- Stage 3 writes run metadata under `outputs/<session_id>/runs/`.

## Benchmark workflow

Use `chronicle benchmark-stage3 <session_id> --truth-file <speaker-map.yaml>` to compare automatic backends on one labeled session or slice.

- `--truth-file` is required and must provide a complete Stage 2 speaker-label to canonical-person mapping using the existing top-level `speaker_map` YAML shape.
- `--backends` accepts a comma-separated subset of `ollama_decomposed,speechbrain_refmatch,speechbrain_hybrid`.
- The benchmark runs each backend sequentially and writes a JSON report plus a Markdown summary under `outputs/<session_id>/runs/`.
- Reports include per-backend status, exact speaker-label assignment accuracy, runtime, backend diagnostics, enrollment coverage, CPU-feasibility notes, and failures.
- Recommendation logic is fixed: choose the highest-accuracy backend; if successful backends land within 2 percentage points of each other, prefer the faster or lighter backend.
- This benchmark phase does not change the production default backend. It produces a recommendation artifact only.

## What Stage 3 does not do

- It does not perform acoustic diarization.
- It does not transcribe audio.
- It does not require remote model services.

## Current limitations

- `ollama_decomposed` and the hybrid tie-break path depend on local Ollama being available and responsive.
- `speechbrain_refmatch` and `speechbrain_hybrid` require usable local participant enrollment clips for the participants being assigned.
- On constrained CPU-only machines, `align-only` or `manual` mode remains the safest production path until benchmark evidence justifies any backend-default change.
