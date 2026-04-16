# Stage 3 Architecture

This document defines the intended architecture for Stage 3 after the stage-model reset.

## Stage 3 goal

Stage 3 should identify real people from the combination of:
- Stage 1 transcript text
- Stage 2 anonymous speaker turns
- session context
- participant metadata

Stage 3 should answer:
- is `SPEAKER_00` Bill, Pat, Wyatt, or still unknown?

It should not do raw acoustic diarization.

## Why Stage 3 is changing

The previous heuristic "semantic diarization" logic belongs more naturally here.
It was trying to solve identity attribution from transcript text and metadata.

That is a speaker-identification problem, not a raw diarization problem.

## Target Stage 3 inputs

Required:
- Stage 1 transcript artifact
- Stage 2 anonymous diarization artifact
- `inputs/global/participants.yaml`
- session `context.md`

Likely useful:
- structured per-speaker background cues
- people likely discussed
- likely ambiguities

## Target Stage 3 outputs

Machine artifact should include:
- canonical speaker name when justified
- anonymous speaker label provenance
- confidence label
- notes
- references back to Stage 1 segment ids and Stage 2 turn ids

Markdown companion should read like a speaker-attributed conversation while preserving uncertainty.

## Expected implementation direction

Stage 3 is a good candidate for an LLM-assisted reconciliation step because it needs:
- contextual reasoning
- conservative mapping from anonymous voices to known people
- explicit handling of ambiguity

If an LLM is used, it should be structured and auditable:
- small windows or grouped turns
- explicit candidate participants
- explicit uncertainty rules
- machine-readable outputs

## Preferred module layout

Target structure:
- `src/chronicle/stage3/service.py`
  - orchestration only
- `src/chronicle/stage3/inputs.py`
  - load Stage 1, Stage 2, and metadata/context
- `src/chronicle/stage3/reconcile.py`
  - identity assignment logic
- `src/chronicle/stage3/artifacts.py`
  - Stage 3 artifact writing
- optionally `src/chronicle/stage3/llm.py` and `prompts.py`
  - if the stage becomes LLM-assisted

## Migration note

The current `src/chronicle/stage2/` package is effectively the prototype for this stage.

So the first code migration should be:
- move current heuristic loading/segmentation/assignment/artifact code out of `stage2/`
- rehouse it under `stage3/`
- update CLI naming and output contracts to match

That migration should happen before building the new anonymous audio-diarization Stage 2.

## Relationship to Stage 2

Stage 2 should stay general and anonymous.
Stage 3 should absorb the biography-heavy context and identity reasoning.

That separation keeps Stage 2 reusable for `n` speakers while keeping Stage 3 responsible for real-person mapping.
