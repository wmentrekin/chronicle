# Stage 1 Parakeet Notes

Status: research notes

## Findings

- Parakeet remains the preferred Stage 1 family for transcript quality.
- The local machine does not match the ideal NVIDIA-first deployment path.
- The lighter CTC variant is the practical local baseline for now.
- The TDT family still looks better on paper for timestamps and punctuation, but it is not the current local target.

## Current Takeaway

- Keep the existing Stage 1 baseline.
- Do not spend more session time on Stage 1 architecture unless transcript quality or runtime becomes the bottleneck again.
- Preserve Stage 1 as a words-only stage with stable file outputs.
