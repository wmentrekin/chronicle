# Stage 2 Diarization Benchmarks

Status: research notes

## Findings

- The SpeechBrain-backed local path is the current performance baseline.
- Pyannote remains the quality/reference baseline.
- SpeechBrain scales materially better on the observed CPU windows.

## Observed Runtime Signal

- 30s window: pyannote `155.88s` wall, speechbrain `39.32s` wall
- 60s window: pyannote `213.09s` wall, speechbrain `70.76s` wall

## Operational Notes

- Full-file runs still need better checkpoint cleanup.
- Long files remain hard to inspect while running.
- The next decision point is artifact quality review, not more backend churn.
