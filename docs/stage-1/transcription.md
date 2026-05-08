# Stage 1 Transcription

Status: current implementation

Stage 1 converts raw session audio into a session-level transcript. It does not assign speakers.

## Inputs

- `inputs/sessions/<session_id>/session.yaml`
- `inputs/sessions/<session_id>/context.md`
- one or more audio files under `inputs/sessions/<session_id>/audio/`

## Outputs

- `outputs/<session_id>/stage1/raw_transcript.json`
- `outputs/<session_id>/stage1/raw_transcript.md`

## Current behavior

- `chronicle transcribe` is the default cloud Stage 1 command.
- The current cloud implementation provisions a worker that executes the Parakeet transcription path.
- Audio is decoded on the worker before transcription.
- The session transcript is assembled from the source audio files in order.
- The transcript keeps source audio provenance and sequential session-relative timestamps.
- Stage 1 writes run metadata under `outputs/<session_id>/runs/`.
- `stage1_model_preference` remains in the session manifest schema for compatibility, but Stage 1 ignores it.
- GCP operator wrappers exist under `scripts/stage1/gcp/`.

## What Stage 1 does not do

- It does not identify speakers.
- It does not use participant metadata to change wording.
- It does not depend on Stage 2 or Stage 3 outputs.

## Current limitations

- The current cloud implementation still depends on GCP worker availability and operator-side `gcloud` setup.
- Cloud execution is partially permanentized in `src`, but still relies on shell/operator steps outside the main CLI for full lifecycle control.
- The Parakeet path depends on model availability and the current Python environment on the worker.
