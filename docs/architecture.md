# Pipeline Architecture

Status: current

## Purpose

Chronicle is a local-first oral-history pipeline. It takes one session input bundle, runs staged processing, and writes reviewable outputs without losing provenance.

The current implemented path is:
1. validate a session bundle
2. run Stage 1 transcription
3. run Stage 2 semantic diarization
4. prepare for Stage 3 chronology extraction

Stage 3 is still a scaffold. Narrative synthesis is not implemented in this repository.

## Current repository structure

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
models/
  parakeet-ctc-0.6b/
src/
  chronicle/
    cli.py
    session.py
    paths.py
    utils.py
    stage1/
    stage2/
    stage3/
docs/
  architecture.md
```

`inputs/` and `outputs/` are intentionally treated as private working data and are gitignored.
`models/` is also gitignored and is the preferred home for Chronicle-managed local model files.

## Runtime model

Chronicle is a packaged Python CLI managed with `uv`.

Current operator flow:

```bash
uv sync
uv run chronicle validate <session_id>
uv run chronicle models fetch parakeet
uv run chronicle transcribe <session_id>
uv run chronicle benchmark-stage1 <session_id>
uv run chronicle diarize <session_id>
uv run chronicle chronology <session_id>
```

Optional Stage 1 backend groups:

```bash
uv sync --group stage1-faster-whisper
uv sync --group stage1-parakeet
```

## CLI architecture

The CLI entrypoint lives in `src/chronicle/cli.py`.

The code is split by responsibility:
- `src/chronicle/paths.py`
  - canonical repo paths and output directory creation
- `src/chronicle/session.py`
  - session manifest parsing and validation
- `src/chronicle/stage1/service.py`
  - audio decode and transcription backends
- `src/chronicle/stage2/service.py`
  - semantic diarization heuristics
- `src/chronicle/stage3/service.py`
  - Stage 3 planning helpers
- `src/chronicle/utils.py`
  - shared serialization and run metadata helpers

This split is intentional. Path changes, validation changes, and stage changes can now be made more locally.

## Input contract

For each session:
- audio lives under `inputs/sessions/<session_id>/audio/`
- `session.yaml` lives under `inputs/sessions/<session_id>/session.yaml`
- `context.md` lives under `inputs/sessions/<session_id>/context.md`
- canonical people metadata lives under `inputs/global/participants.yaml`

The current manifest contract is:
- `audio_files` are relative to the session folder
- `context_doc` is relative to the session folder
- names in `participants`, `primary_interviewees`, and `people_likely_discussed` must match canonical names in `participants.yaml`

## Output contract

Each session writes to `outputs/<session_id>/`.

Current layout:
- `outputs/<session_id>/stage1/`
  - session-level raw transcript JSON
  - session-level raw transcript markdown
- `outputs/<session_id>/stage2/`
  - diarized conversation JSON
  - diarized conversation markdown
- `outputs/<session_id>/stage3/`
  - reserved for chronology outputs
- `outputs/<session_id>/runs/`
  - stage run metadata JSON files

The rule is simple: each stage writes a new artifact. Stages do not silently mutate previous-stage semantics.

## Stage 1 architecture

Current implementation:
- decode session audio to mono 16 kHz using `imageio-ffmpeg`
- select backend: `auto`, `faster-whisper`, or `parakeet`
- prefer Chronicle-managed local Parakeet model files under `models/parakeet-ctc-0.6b/`
- write one combined transcript artifact per session, ordered sequentially across source audio files

Current backends:
- `faster-whisper`
- `parakeet`

Current local Parakeet path:
- local `transformers` ASR pipeline
- fixed-size chunking
- chunk-level timestamps
- explicit local model path management instead of treating the generic Hugging Face cache as the primary runtime contract
- one overall session progress bar with current file, chunk counts, throughput, elapsed time, and ETA
- current default chunk length: `15s`
- experimental overlap mode: `15s` windows with `7.5s` stride

Current Stage 1 benchmark path:
- `uv run chronicle benchmark-stage1 <session_id>`
- benchmarks evenly spaced subsamples across the full session
- compares multiple chunk sizes on the same sampled audio windows
- writes terminal output plus JSON results under `outputs/<session_id>/runs/`
- experimental concurrency benchmark:
  - `uv run chronicle benchmark-stage1-concurrency <session_id>`
  - holds chunk size constant and varies worker process count over partitioned sample windows
  - intended to validate speedup before changing the main transcribe path

Current benchmark findings on the sample session:
- `15s` is the best current default tradeoff on local CPU
- `10s` and `20s` were faster than smaller chunks in some tests but degraded into `<unk>` output on the sampled windows
- `15s` overlap mode with `7.5s` stride preserves alternate boundary readings but is slower and currently requires a future reconciliation/stitching pass to be useful as a primary artifact

Current optimization findings from profiling:
- the serial `15s` path is dominated by chunk inference time, not setup time
- representative 10-minute serial profile:
  - decode: about `6.6s`
  - model init: about `13.3s`
  - chunk inference: about `128s`
- the current multiprocessing concurrency experiments are slower than the serial baseline on this machine
- the main reason is not worker startup alone; it is that concurrent CPU inference with multiple full Parakeet worker processes becomes inefficient
- practical recommendation today:
  - keep Stage 1 serial
  - keep `15s` as the default chunk size
  - do not promote current multiprocessing concurrency into the main `transcribe` path
- better optimization directions:
  - reduce cost per inference call in the serial path
  - reduce the number of inference calls without triggering `<unk>` degradation
  - profile or replace the current `transformers` CPU inference runtime before revisiting worker-based concurrency

Important boundaries:
- Stage 1 does not assign speakers
- Stage 1 does not use family metadata to alter wording
- Stage 1 preserves decode uncertainty with explicit markers like empty chunks or `<unk>`
- Stage 1 preserves source-audio provenance inside the combined session artifact

## Stage 2 architecture

Current Stage 2 is semantic diarization, not acoustic diarization.

Inputs:
- Stage 1 transcript JSON
- `session.yaml`
- `context.md`
- `inputs/global/participants.yaml`

Current process:
1. load Stage 1 transcript segments
2. derive alias data from `participants.yaml`
3. extract clue tokens from the background section of `context.md`
4. classify transcript segments as likely question, acknowledgment, or response
5. merge adjacent segments into larger speaking blocks
6. score question targets and response candidates against session clues
7. assign speakers conservatively
8. write JSON and markdown outputs with confidence labels and notes

Outputs:
- `diarized_conversation.json`
- `diarized_conversation.md`

Important boundaries:
- no API calls are used in current Stage 2
- no LLM calls are used in current Stage 2
- no speaker embeddings or acoustic diarization are used
- uncertainty is preserved instead of forced into a single speaker label

## Stage 3 architecture

Stage 3 is not implemented yet.

Current state:
- the CLI validates the session
- it prepares `outputs/<session_id>/stage3/`
- it shows the planned artifact names for each primary interviewee

Target direction:
- one chronology artifact per interviewee per session
- verbatim excerpts grouped by theme and approximate life period
- references back to transcript provenance

## Current workflow

To add a new interview:
1. create `inputs/sessions/<session_id>/`
2. place the raw audio under `audio/`
3. add `session.yaml`
4. add `context.md`
5. update `inputs/global/participants.yaml` if needed
6. run `uv sync`
7. run `uv run chronicle validate <session_id>`
8. run Stage 1 and Stage 2 explicitly

## Current design tradeoffs

Why this structure is cleaner than the old one:
- inputs and outputs are separated
- session-local input context is separated from derived artifacts
- the CLI resolves sessions by `session_id` instead of file paths
- stage code is no longer trapped inside one large script
- future public release work is simpler because private data sits under gitignored roots

Known limits:
- Stage 1 Parakeet quality still needs review
- Stage 2 remains heuristic and transcript-driven
- Stage 3 is still pending

## Next architectural steps

Near-term:
1. tighten Stage 1 quality and local runtime behavior
2. refine Stage 2 heuristics and context extraction
3. implement Stage 3 chronology extraction
4. add tests around validation and stage artifact contracts

Later:
1. add sanitized examples for public sharing
2. decide whether any stage should use controlled LLM assistance
3. add a narrative drafting layer only after earlier artifacts are stable
