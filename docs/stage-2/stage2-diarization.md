# Stage 2 Diarization

Status: current implementation

Stage 2 performs anonymous audio diarization. It answers "who spoke when" using anonymous labels such as `SPEAKER_00`.

## Inputs

- raw session audio from `inputs/sessions/<session_id>/audio/`
- Stage 1 is not required as an input to the diarizer itself

## Outputs

- `outputs/<session_id>/stage2/diarization.json`
- `outputs/<session_id>/stage2/diarization.md`

## Current behavior

- Production diarization uses the local SpeechBrain-backed path.
- `chronicle benchmark-stage2` keeps pyannote and a SpeechBrain-style benchmark runner available for comparison.
- Turns are written with anonymous speaker labels, per-file source audio, and source/session-relative timestamps.
- The stage writes partial checkpoint artifacts during processing and replaces them with final artifacts on success.
- Stage 2 records run metadata under `outputs/<session_id>/runs/`.

## What Stage 2 does not do

- It does not map speakers to canonical people.
- It does not rewrite transcript wording.
- It does not use biography-heavy context to infer identity.

## Historical note

Earlier transcript-driven heuristic "semantic diarization" is no longer the current Stage 2 implementation. The current stage is acoustic, anonymous, and local-first.

## Current limitations

- Output quality still depends on the speaker-count constraints and the chosen backend settings.
- Long-file visibility is still more useful when watching a live run than when reading the final artifacts alone.

