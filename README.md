# Chronicle

Chronicle is a local-first oral-history pipeline for turning interview recordings into reviewable source material.

Current implemented stages:
- Stage 1 transcription
- Stage 2 semantic diarization and conservative speaker assignment
- Stage 3 chronology is still a scaffold

## Repository shape

```text
inputs/
  global/
    participants.yaml
  sessions/
    <session_id>/
      audio/
      session.yaml
      context.md
outputs/
  <session_id>/
    stage1/
    stage2/
    stage3/
    runs/
src/chronicle/
docs/
models/
examples/
```

`inputs/` and `outputs/` are intentionally gitignored because they contain private family material.
`models/` is also gitignored by default because Chronicle can manage local model files there.
Sanitized scaffolds for a public setup live under `examples/inputs/`.

## Environment

```bash
uv sync
```

Optional Stage 1 backends:

```bash
uv sync --group stage1-faster-whisper
uv sync --group stage1-parakeet
```

Fetch the local Parakeet model into Chronicle-managed storage:

```bash
uv run chronicle models fetch parakeet
```

## CLI

Validate a session:

```bash
uv run chronicle validate <session_id>
```

Run Stage 1:

```bash
uv run chronicle transcribe <session_id>
```

Run Stage 1 with experimental overlap mode:

```bash
uv run chronicle transcribe <session_id> --experimental-overlap
```

Benchmark Stage 1 chunk sizes across evenly spaced subsamples:

```bash
uv run chronicle benchmark-stage1 <session_id>
```

Benchmark experimental Stage 1 concurrency with partitioned worker processes:

```bash
uv run chronicle benchmark-stage1-concurrency <session_id>
```

Run Stage 2:

```bash
uv run chronicle diarize <session_id>
```

Prepare Stage 3:

```bash
uv run chronicle chronology <session_id>
```

Show current pipeline status for a session:

```bash
uv run chronicle run <session_id>
```

## Notes

- Raw audio is expected under `inputs/sessions/<session_id>/audio/`.
- Session-specific metadata lives beside it in `session.yaml` and `context.md`.
- Canonical people metadata lives in `inputs/global/participants.yaml`.
- Public users should start from the sanitized scaffolds under `examples/inputs/` and create their own local `inputs/` tree from those templates.
- Stage 1 writes one session-level transcript artifact per session, even when the session contains multiple audio files.
- Chronicle now prefers a managed local Parakeet model directory under `models/parakeet-ctc-0.6b/` rather than treating the Hugging Face cache as the primary runtime contract.
- Stage 1 Parakeet runs now expose a single session-level progress bar with chunk counts, throughput, and ETA.
- The current default Parakeet chunk length is `15s`.
- Chronicle also exposes an experimental overlap mode that uses `15s` windows with half-window stride. It is slower and currently emits overlapping chunks without a reconciliation pass.
- The current Stage 2 implementation is local heuristic diarization over the Stage 1 transcript. It does not do audio speaker embedding or acoustic diarization.
