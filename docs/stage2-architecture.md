# Stage 2 Architecture

This document explains how Stage 2 semantic diarization is organized in code and how it consumes Stage 1 output.

## Stage 2 goal

Stage 2 takes the Stage 1 transcript plus session metadata and produces:

- a speaker-attributed conversation JSON artifact
- a speaker-attributed conversation Markdown artifact
- per-block confidence labels and notes

Stage 2 is not audio diarization.
It is transcript-driven semantic diarization and conservative speaker assignment.

## Stage 2 module layout

### `src/chronicle/stage2/service.py`

Purpose:
- orchestrate the Stage 2 run
- load inputs
- build candidate blocks
- assign speakers
- write artifacts

This should remain the control-flow layer only.

### `src/chronicle/stage2/inputs.py`

Purpose:
- load participant metadata
- load Stage 1 transcript segments
- load session context text
- build aliases and context clues

This module owns the Stage 2 input-preparation layer.

### `src/chronicle/stage2/segmentation.py`

Purpose:
- segment-type classification
- transcript tokenization for matching
- merge Stage 1 segments into candidate conversation blocks

This module turns Stage 1 chunks into larger candidate blocks such as:
- question
- response
- acknowledgments folded into neighboring blocks where appropriate

### `src/chronicle/stage2/assignment.py`

Purpose:
- score likely interviewee targets
- score likely response speakers
- assign speakers conservatively
- reconcile uncertain question targeting after seeing the following response

This module contains the actual attribution heuristics.

### `src/chronicle/stage2/artifacts.py`

Purpose:
- output path helpers
- Markdown rendering for the diarized conversation artifact

This isolates output formatting from attribution logic.

## What happens when `chronicle diarize <session_id>` runs

1. `src/chronicle/cli/stage2.py` validates the session and calls `execute_stage2(...)`.
2. `execute_stage2(...)` computes Stage 2 output paths.
3. It loads participant metadata from `inputs/global/participants.yaml`.
4. It loads the session context document.
5. It loads Stage 1 transcript segments from:
   - session-level Stage 1 artifact if present
   - legacy per-audio artifacts only as fallback
6. It passes those segments into `build_stage2_candidate_blocks(...)`.
7. It passes the candidate blocks plus metadata into `assign_stage2_blocks(...)`.
8. It runs a reconciliation pass to soften overconfident question targeting if the following response suggests otherwise.
9. It writes:
   - `diarized_conversation.json`
   - `diarized_conversation.md`

## What Stage 2 actually uses

Stage 2 uses:
- Stage 1 transcript text
- Stage 1 timestamps
- session context text
- participant metadata
- known interviewee lists
- heuristic text matching

Stage 2 does not currently use:
- speaker embeddings
- waveform-based speaker separation
- cloud LLM calls
- audio diarization models

That is an intentional current design choice.

## Why Stage 2 is split this way

The split mirrors the actual mental model of the stage:

1. load inputs
2. segment transcript text into plausible conversation blocks
3. score candidate speakers
4. assign conservatively
5. write artifacts

That is easier to maintain than one large file mixing:
- metadata loading
- regex heuristics
- scoring logic
- output formatting

## Relationship to Stage 1

Stage 2 depends on Stage 1, but the boundary is explicit:

- Stage 1 owns speech-to-text
- Stage 2 owns speaker inference over transcript text

That separation is important.
It keeps transcript wording and speaker attribution as distinct evidence layers.
