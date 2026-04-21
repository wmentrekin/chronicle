# Next Session Handoff

Date captured: 2026-04-20

## Current State

- Stage 1 and Stage 2 are stable enough to treat as upstream inputs.
- Stage 3 is the current focus and now uses local Ollama rather than the stale OpenAI/GPT defaults that were previously written down.
- The current single-prompt Stage 3 design is too large for this 8 GB CPU-only machine.
- `qwen3:8b` and `gemma3:4b` both timed out under that design.

## Next Step

Implement prompt decomposition for Stage 3 so the local model can work on smaller, bounded tasks instead of one large assignment prompt.

## Follow-on Order

1. Split Stage 3 into smaller prompt stages.
2. Keep speaker attribution conservative and reviewable.
3. Re-run the local Ollama path after decomposition.
4. Move on to Stage 4 organization only after Stage 3 is usable.

## Guardrails

- Keep the work local-first.
- Do not add private transcript excerpts here.
- Do not reintroduce the old OpenAI/GPT defaults as current behavior.
