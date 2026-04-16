# Stage 4 Architecture

This document defines the intended architecture for Stage 4 after the stage-model reset.

## Stage 4 goal

Stage 4 should organize identified speaker content into narrative-ready source documents without rewriting the speaker's words.

Stage 4 should take the identified Stage 3 conversation artifact and produce:
- one organized artifact per interviewee per session
- grouping by theme
- grouping by approximate chronology where justified
- explicit provenance and uncertainty

## Why Stage 4 exists separately

Organization is not the same thing as identification.

Stage 3 should decide who was speaking.
Stage 4 should decide how to group already-identified excerpts into something more usable for manual narrative drafting.

That keeps attribution, chronology, and organization from collapsing into one opaque step.

## Target Stage 4 inputs

Required:
- Stage 3 identified speaker artifact

Optional:
- configurable theme sets
- configurable chronology headings

## Target Stage 4 outputs

Per interviewee:
- machine-readable organization artifact
- markdown companion

Each organized entry should preserve:
- verbatim wording
- source transcript block ids
- timestamps where available
- confidence and review notes

## Expected implementation direction

This stage is likely LLM-assisted, but should remain auditable:
- no paraphrasing
- no invented chronology
- explicit uncertainty when ordering is weak

## Current code-state note

Current `src/chronicle/stage3/` is only a scaffold for the old chronology concept.
That scaffold should become the seed of a future `stage4/` package once Stage 3 is repurposed for speaker identification.
