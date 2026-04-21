# Local-First Model Policy

Status: current decision

## Policy

- Prefer local models and local execution by default.
- Do not introduce remote dependencies without an explicit reason.
- Keep raw audio and transcript data local unless a future exception is deliberately approved.

## Current Stage 3 Implication

- Stage 3 should be designed around local Ollama for the current implementation path.
- The prompt shape must fit the available CPU-only machine.
- If a model does not finish locally, the prompt design should be simplified before widening scope.

## Safety Rule

- Never treat remote model defaults as the hidden fallback.
- Never leak sensitive session material into a remote service by accident.
