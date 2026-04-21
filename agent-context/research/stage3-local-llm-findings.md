# Stage 3 Local LLM Findings

Status: research notes

## Current State

- Stage 3 now uses local Ollama for the current implementation direction.
- The older OpenAI/GPT defaults are stale and should not be treated as current behavior.
- The current single-prompt design does not fit the local 8 GB CPU-only machine well enough.

## Model Spike Results

- `qwen3:8b` timed out.
- `gemma3:4b` timed out.

## Conclusion

- The next implementation step is prompt decomposition.
- Smaller bounded prompts are more likely to finish locally and stay reviewable.
- Any future local model choice should be judged against completion time, not just output quality.
