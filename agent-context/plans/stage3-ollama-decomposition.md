# Stage 3 Ollama Decomposition Plan

Status: superseded by implementation-ready plan

## Current Plan

The refined implementation plan now lives at:

- `docs/plans/stage3-ollama-decomposition.md`

## Planning Outcome

The plan replaces the one-shot Stage 3 Ollama prompt with a speaker-centric decomposed workflow.

Key resolved decisions:

- Stage 3 `llm` and `manual` remain strict one-to-one mapping modes.
- Stage 3 fails before Ollama if Stage 2 speaker count and manifest participant count do not match.
- Decomposition uses one bounded assignment call per unresolved speaker.
- Exactly one repair pass is allowed.
- `llm_usage` records timings, token counts, statuses, attempts, and hashes, but not raw prompt text or response text.
- A local smoke benchmark should validate the decomposed path against target-machine budgets.

## Why This Plan Exists

The current single-prompt Stage 3 LLM design timed out with both `qwen3:8b` and `gemma3:4b` on the local 8 GB CPU-only machine.

The next implementation should optimize prompt shape before changing model strategy.
