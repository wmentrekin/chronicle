# Stage 2 Architecture

This document defines the intended architecture for Stage 2 after the stage-model reset.

## Stage 2 goal

Stage 2 should perform anonymous audio diarization.

It should take raw session audio and produce:
- anonymous speaker-turn artifacts
- timestamps for those turns
- overlap or ambiguity markers where supported

Stage 2 should not decide whether `SPEAKER_00` is Bill or Pat.
That belongs in Stage 3.

## Why Stage 2 is changing

The earlier Stage 2 implementation was transcript-driven semantic diarization.
That approach has a ceiling when multiple people discuss the same family context.

The new design separates evidence types more cleanly:
- Stage 1 answers: what words were spoken?
- Stage 2 answers: which anonymous voice spoke when?
- Stage 3 answers: which known person is each anonymous speaker most likely to be?

## Target Stage 2 inputs

Required:
- raw session audio

Optional:
- expected speaker count
- minimum speaker count
- maximum speaker count
- other diarization constraints that are general enough to remain reusable

Stage 2 should remain broadly reusable.
It should not depend on biography-heavy family context, because that belongs in Stage 3.

## Target Stage 2 outputs

Machine artifact should include:
- `speaker_label` such as `SPEAKER_00`
- `start_time`
- `end_time`
- `source_audio`
- `overlap` or similar flag if available
- diarization notes or confidence metadata if exposed by the chosen stack

Markdown companion should be reviewable by a human and preserve the anonymous labels.

## Intended module layout

The exact stack is not chosen yet, but the preferred structure is:

- `src/chronicle/stage2/service.py`
  - Stage 2 orchestration only
- `src/chronicle/stage2/audio.py`
  - any Stage 2-specific audio preparation
- `src/chronicle/stage2/diarizer.py`
  - chosen diarization backend wrapper
- `src/chronicle/stage2/artifacts.py`
  - Stage 2 artifact writing and formatting

If the chosen stack requires backend-specific modules, keep them separated rather than growing one large file.

## What `chronicle diarize <session_id>` should eventually do

Target flow:
1. validate the session
2. resolve the session audio files
3. load diarization configuration or speaker-count hints if provided
4. run anonymous speaker diarization over the audio
5. write Stage 2 JSON and markdown artifacts with anonymous speaker labels
6. record run metadata

## Speaker-count guidance

Stage 2 should support `n` speakers up to a reasonable local-machine limit.

Useful options to support:
- exact speaker count when known
- min/max speaker count when exact count is not known

That keeps Stage 2 useful across small interviews without hard-coding a two-speaker assumption.

## Relationship to Stage 1

Stage 1 and Stage 2 should both consume raw audio, but they answer different questions:
- Stage 1 produces words and timestamps
- Stage 2 produces anonymous speaker turns

These are parallel evidence layers that Stage 3 will reconcile.

## Current code-state note

The current `src/chronicle/stage2/` package does not match this target architecture.
It still contains the older text-first heuristic speaker-assignment logic.

That code should be migrated into future Stage 3 ownership before the new Stage 2 is implemented.
