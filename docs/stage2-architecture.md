# Stage 2 Architecture

This document defines the intended architecture for Stage 2 after the stage-model reset.

Current status:
- `chronicle diarize <session_id>` now uses the current SpeechBrain-backed production backend
- `chronicle benchmark-stage2 <session_id>` remains the evaluation entrypoint
- Chronicle currently supports two benchmark backends:
  - `pyannote`
  - a custom SpeechBrain-style VAD + embedding + clustering spike

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

## Current module layout

- `src/chronicle/stage2/service.py`
  - Stage 2 orchestration, per-file processing, progress callbacks
- `src/chronicle/stage2/artifacts.py`
  - Stage 2 artifact writing and formatting
- `src/chronicle/stage2/benchmark.py`
  - Stage 2 sample extraction and separate-runtime orchestration
- `src/chronicle/stage2/pyannote_spike_runner.py`
  - pyannote benchmark backend
- `src/chronicle/stage2/speechbrain_spike_runner.py`
  - SpeechBrain-style benchmark backend and current production backend runner

## What `chronicle diarize <session_id>` does now

Current flow:
1. validate the session
2. resolve the session audio files
3. process each source audio file through the SpeechBrain runner in a separate runtime
4. offset anonymous turns into one session timeline
5. write Stage 2 JSON and markdown artifacts with anonymous speaker labels
6. write partial checkpoints after each completed source file
7. record run metadata

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

## Current benchmark findings

Current backend roles:
- `pyannote` is the reference-quality baseline
- the refined SpeechBrain-style path is the current local-performance baseline

On this machine, the early directional result is:
- pyannote can produce cleaner fine-grained turns
- SpeechBrain currently scales better on longer `30s` and `60s` CPU windows

That means the next Stage 2 decision should be based on:
- output quality review on representative windows
- not runtime alone

## Current code-state note

The old text-first heuristic speaker-assignment logic has already been moved out of Stage 2 and into Stage 3 ownership.

What Stage 2 now has:
- a chosen current production backend for local use
- stable Stage 2 artifact writers for `chronicle diarize`
- benchmark backends for quality/performance comparison

What still needs hardening:
- cleanup of partial checkpoint files on successful completion
- investigation of the lingering terminal-session behavior after successful runs
- better long-file visibility while a single source file is still being diarized
