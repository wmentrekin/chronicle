# Chronicle

Chronicle is a local-first, multi-stage agentic audio-processing workflow for turning interview recordings into reviewable source material through a CLI.

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

`inputs/` and `outputs/` are intentionally gitignored because they can contain private source material.
`models/` is also gitignored by default because Chronicle can manage local model files there.
Sanitized scaffolds for a public setup live under `examples/inputs/`.

## Environment

```bash
uv sync
```

Preferred first-time bootstrap:

```bash
./bin/bootstrap
```

That:
- runs `uv sync`
- initializes Chronicle's managed local Parakeet model
- leaves the repo ready for `chronicle ...` usage through either `direnv`, `.envrc`, or an optional user-level link

If you want a stable `chronicle` command without sourcing `.envrc`, use:

```bash
./bin/bootstrap --install-link
```

That installs `~/.local/bin/chronicle` pointing at this repo's active Chronicle executable.

You can also do that directly after setup:

```bash
chronicle init --install-link
```

To use repo-local PATH handling instead, load the repo shell settings manually:

```bash
source .envrc
```

That prepends `.venv/bin` to `PATH` and sets a repo-local `UV_CACHE_DIR`.

Plain `uv sync` installs Chronicle's default Stage 1 runtime, which is Parakeet.

Optional alternate Stage 1 backend:

```bash
uv sync --group stage1-faster-whisper
```

Initialize Chronicle's managed local Parakeet model:

```bash
chronicle init
```

Manual first-time bootstrap, from a fresh clone:

```bash
uv sync
source .envrc
chronicle init
```

## CLI

Validate a session:

```bash
chronicle validate <session_id>
```

Run Stage 1:

```bash
chronicle transcribe <session_id>
```

Run Stage 1 with experimental overlap mode:

```bash
chronicle transcribe <session_id> --experimental-overlap
```

Benchmark Stage 1 chunk sizes across evenly spaced subsamples:

```bash
chronicle benchmark-stage1 <session_id>
```

Benchmark experimental Stage 1 concurrency with partitioned worker processes:

```bash
chronicle benchmark-stage1-concurrency <session_id>
```

Run Stage 2:

```bash
chronicle diarize <session_id>
```

Prepare Stage 3:

```bash
chronicle chronology <session_id>
```

Show current pipeline status for a session:

```bash
chronicle run <session_id>
```

## Notes

- Raw audio is expected under `inputs/sessions/<session_id>/audio/`.
- Session-specific metadata lives beside it in `session.yaml` and `context.md`.
- Canonical people metadata lives in `inputs/global/participants.yaml`.
- Public users should start from the sanitized scaffolds under `examples/inputs/` and create their own local `inputs/` tree from those templates.
- Stage 1 writes one session-level transcript artifact per session, even when the session contains multiple audio files.
- Chronicle now prefers a managed local Parakeet model directory under `models/parakeet-ctc-0.6b/` rather than treating the Hugging Face cache as the primary runtime contract.
- The default Stage 1 backend is Parakeet when the runtime and local model are available.
- Stage 1 Parakeet runs now expose a single session-level progress bar with chunk counts, throughput, and ETA.
- The current default Parakeet chunk length is `15s`.
- The current default Parakeet batch size is `4`.
- Chronicle also exposes an experimental overlap mode that uses `15s` windows with half-window stride. It is slower and currently emits overlapping chunks without a reconciliation pass.
- The current Stage 2 implementation is local heuristic diarization over the Stage 1 transcript. It does not do audio speaker embedding or acoustic diarization.
