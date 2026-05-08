# Stage 1 Remote Research Notes

These notes capture discovery-time findings that informed the Stage 1 remote single-path requirements. They are intentionally time-sensitive and should be revalidated before implementation.

## Repository Findings

- Stage 1 currently exposes `--backend` with `parakeet`, `faster-whisper`, and `auto`, plus backend-specific tuning flags and manifest preference fallback logic.
- The current default Stage 1 backend is `parakeet`, with default model `nvidia/parakeet-ctc-0.6b`.
- Existing session and benchmark artifacts already lean toward Parakeet rather than faster-whisper.
- Local concurrency benchmarks did not clearly beat the simpler baseline in the exploratory run referenced in the requirements artifact.

## Provider Research

- Google Cloud currently appears to have the clearest fit for a first remote implementation because it combines trial credit, published GPU pricing, and accessible T4 availability.
- Azure and AWS are viable alternatives, but the public pricing path is less direct for this use case.
- Oracle Cloud has useful trial credit and always-free CPU resources, but no compelling permanent free GPU path.
- Hugging Face ZeroGPU and Colab are useful for experimentation, but they are not a strong fit for a private, repeatable Chronicle worker.

## Working Hypothesis

- The best initial remote shape is a small NVIDIA GPU VM under the user's own cloud account.
- The first implementation should preserve Chronicle's stage isolation and artifact contract rather than introducing a new product architecture.

