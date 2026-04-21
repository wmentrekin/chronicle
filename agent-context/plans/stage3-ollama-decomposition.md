# Stage 3 Ollama Decomposition Plan

Status: active plan

## Problem

The current Stage 3 prompt is too large for the local 8 GB CPU-only environment. `qwen3:8b` and `gemma3:4b` timed out with the single-prompt design.

## Goal

Replace the one-shot Stage 3 speaker-identification prompt with a decomposed local Ollama workflow that fits the machine and stays reviewable.

## Proposed Shape

- first pass: compact evidence extraction per anonymous speaker
- second pass: constrained identity assignment over a much smaller prompt
- optional final pass: validation and conflict checks

## Design Priorities

- keep all processing local
- reduce prompt size aggressively
- preserve provenance and uncertainty
- avoid inventing names, dates, or relationships
- keep output deterministic enough for review

## Success Criteria

- the local Ollama path completes on this machine
- speaker assignments remain conservative
- prompts stay bounded and explainable
- the resulting output is easier to inspect than the failed single-prompt version

## Open Questions

- how much evidence to carry between passes
- whether the decomposition should be speaker-centric or block-centric first
- how to surface truncated evidence without losing reviewability
