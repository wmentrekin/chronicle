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

- Default backend is local Parakeet.
- `faster-whisper` remains available as a compatibility fallback.
- Audio is decoded locally before transcription.
- The session transcript is assembled from the source audio files in order.
- The transcript keeps source audio provenance and sequential session-relative timestamps.
- Stage 1 writes run metadata under `outputs/<session_id>/runs/`.

## What Stage 1 does not do

- It does not identify speakers.
- It does not use participant metadata to change wording.
- It does not depend on Stage 2 or Stage 3 outputs.

## Current limitations

- Stage 1 is optimized for local execution, not remote services.
- The Parakeet path depends on local model availability and the current Python environment.

