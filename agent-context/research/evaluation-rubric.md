# Evaluation Rubric

Status: draft

## Purpose

This rubric is for deciding whether a pipeline stage is good enough to keep in the Chronicle workflow.

## Evaluation Principles

- prefer reviewability over flashy output
- penalize invented certainty heavily
- value correct names, dates, and relationships highly
- preserve verbatim language
- record both quality and operational cost

## Stage 1

Check transcription quality for:

- word accuracy
- proper noun accuracy
- timestamp usefulness
- duplicate or skipped content
- runtime practicality

## Stage 2

Check diarization quality for:

- turn-boundary accuracy
- speaker-cluster stability
- confidence-label honesty
- readability
- usefulness of notes

## Stage 3

Check speaker-identification quality for:

- speaker-name accuracy
- confidence-label honesty
- provenance clarity
- ambiguity handling
- usefulness for Stage 4

## Stage 4

Check organization quality for:

- verbatim fidelity
- chronology usefulness
- theme grouping usefulness
- provenance clarity
- ambiguity handling

## End-to-End

Check the full pipeline for:

- setup effort
- rerun reliability
- terminal UX
- cost control
- usefulness for manual narrative drafting

## Review Checklist

- raw files were preserved unchanged
- canonical names were resolved through participant metadata
- uncertain names or dates were flagged
- outputs were written to stage-specific locations
- model/runtime were recorded
- the result is better than the previous baseline

## Benchmark Log

```md
| Variant | Session | Runtime | Cost | Stage 1 score | Stage 2 score | Stage 3 score | Stage 4 score | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
```

## Recommendation

Do not promote a stage implementation to default status until it passes a real session review and its failure modes are documented.
