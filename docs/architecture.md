# Pipeline Architecture

Status: current design direction

## Purpose

Chronicle is a local-first, multi-stage agentic audio-processing workflow. It takes one session input bundle, runs staged CLI processing, and writes reviewable outputs without losing provenance.

The intended architecture is now:
1. Stage 1 transcription
2. Stage 2 audio diarization
3. Stage 3 speaker identification
4. Stage 4 verbatim organization

This is a change from the earlier 3-stage model. The old transcript-driven heuristic "semantic diarization" step is being reclassified as speaker identification logic and should move out of `stage2/` into a future `stage3/` implementation area.

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
    cli/
    session.py
    paths.py
    utils.py
    stage1/
    stage2/
    stage3/
    stage4/
docs/
  architecture.md
  cli-architecture.md
  stage1-architecture.md
  stage2-architecture.md
  stage3-architecture.md
  stage4-architecture.md
agent-context/
  ...
```

`inputs/` and `outputs/` are intentionally treated as private working data and are gitignored.
`models/` is also gitignored and is the preferred home for Chronicle-managed local model files.

## Runtime model

Chronicle is a packaged Python CLI managed with `uv`.

Current operator flow:

```bash
./bin/bootstrap
chronicle init
chronicle validate <session_id>
chronicle transcribe <session_id>
chronicle diarize <session_id>
chronicle identify <session_id>
chronicle organize <session_id>
```

Current code is mostly aligned with that command surface now. Today:
- `transcribe` is real Stage 1
- `diarize` uses the current local SpeechBrain-backed Stage 2 implementation
- `identify` runs the migrated heuristic text-first logic under Stage 3
- `organize` is scaffold-only

So the command surface is aligned, but Stage 2 still needs backend hardening and Stage 3 still needs redesign around true Stage 2 inputs.

## CLI architecture

The CLI entrypoint lives in the `src/chronicle/cli/` package, with command registration in `src/chronicle/cli/app.py`.

Chronicle intentionally prefers smaller modules split by responsibility over a few very large files.
From a code-organization standpoint, it is better for this repository to have more focused files than to let stage logic accumulate into monolithic `service.py` modules.
The goal is to keep behavior easier to inspect, change, test, and document locally.

Detailed walkthroughs:
- `docs/cli-architecture.md`
- `docs/stage1-architecture.md`
- `docs/stage2-architecture.md`
- `docs/stage3-architecture.md`
- `docs/stage4-architecture.md`

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

## Target output contract

Each session writes to `outputs/<session_id>/`.

Target layout:
- `outputs/<session_id>/stage1/`
  - session-level raw transcript JSON and markdown
- `outputs/<session_id>/stage2/`
  - anonymous diarization JSON and markdown
- `outputs/<session_id>/stage3/`
  - identified speaker transcript JSON and markdown
- `outputs/<session_id>/stage4/`
  - organized verbatim chronology/theme artifacts
- `outputs/<session_id>/runs/`
  - stage run metadata JSON files

Current code implements:
- `stage1/` transcription
- `stage2/` local anonymous diarization plus benchmark backends
- `stage3/` migrated heuristic speaker-identification logic
- `stage4/` scaffold-only organization planning helpers

## Stage boundaries

## Stage 1 architecture

Stage 1 is speech-to-text only.

Current implementation:
- decode session audio to mono 16 kHz using `imageio-ffmpeg`
- select backend: `auto`, `faster-whisper`, or `parakeet`
- prefer Chronicle-managed local Parakeet model files under `models/parakeet-ctc-0.6b/`
- write one combined transcript artifact per session, ordered sequentially across source audio files

Current local Parakeet path:
- local `transformers` ASR pipeline
- fixed-size chunking
- chunk-level timestamps
- one Parakeet pipeline load per Stage 1 run, reused across all source audio files in that session
- one overall session progress bar with current file, chunk counts, throughput, elapsed time, and ETA
- current default chunk length: `15s`
- current default batch size: `4`

Important boundaries:
- Stage 1 does not assign speakers
- Stage 1 does not use contextual metadata to alter wording
- Stage 1 preserves source-audio provenance inside the combined session artifact

## Stage 2 architecture

Stage 2 should become anonymous audio diarization.

Target purpose:
- separate waveform turns into anonymous speaker tracks such as `SPEAKER_00`, `SPEAKER_01`
- support `n` speakers up to a reasonable local-machine limit
- stay general and not depend on biography-heavy context

Target inputs:
- raw session audio
- optional min/max/expected speaker count
- optional session-level diarization constraints

Target outputs:
- speaker turns with anonymous labels
- start/end timestamps
- overlap or uncertainty flags where available

Important boundary:
- Stage 2 should answer "who spoke when" in anonymous speaker-cluster terms, not "which known person was that"

## Stage 3 architecture

Stage 3 should become speaker identification and transcript reconciliation.

Target purpose:
- combine Stage 1 transcript words with Stage 2 anonymous diarization turns
- use participants metadata and session context to map anonymous speakers to canonical people
- preserve uncertainty when mapping is weak

Target inputs:
- Stage 1 transcript artifact
- Stage 2 diarization artifact
- `inputs/global/participants.yaml`
- session `context.md`

Expected implementation direction:
- likely LLM-assisted
- structured outputs
- conservative identity assignment

Important boundary:
- Stage 3 should not do raw acoustic diarization
- Stage 3 should not rewrite the speaker's wording

## Stage 4 architecture

Stage 4 should become verbatim organization.

Target purpose:
- reorganize identified conversation content into reviewable, narrative-ready source documents
- preserve verbatim excerpts while grouping by theme and approximate chronology

Target inputs:
- Stage 3 identified transcript artifact

Target outputs:
- one organized artifact per primary interviewee per session
- chronology/theme groupings
- explicit uncertainty and provenance references

## Current code state vs target architecture

Current code now reflects the new numbering, and Stage 2 now has an initial production implementation:
- `src/chronicle/stage1/` is real transcription code
- `src/chronicle/stage2/` now contains the current SpeechBrain-backed implementation plus benchmark tooling and backend spikes
- `src/chronicle/stage3/` contains the migrated text-first heuristic speaker-identification logic
- `src/chronicle/stage4/` is scaffold-only organization planning

That means the current code-state mapping is:
- new Stage 2 is implemented as the current production `chronicle diarize` path using the SpeechBrain backend
- current Stage 3 is transitional until it can consume true Stage 2 diarization artifacts

## Implementation plan

### Step 1: migrate code and artifact naming to the new stage model

This step is now complete:
- current `src/chronicle/stage2/` logic was moved into `src/chronicle/stage3/`
- current CLI command surface was realigned to `transcribe -> diarize -> identify -> organize`
- old chronology scaffolding was moved forward into `src/chronicle/stage4/`
- output directory expectations now include `stage4/`

The next work is hardening the current Stage 2 backend and integrating it cleanly with Stage 3.

### Step 2: research and choose a local Stage 2 diarization stack

This step is now in progress.

Current findings:
- `pyannote` remains the quality/reference baseline
- a custom SpeechBrain-style spike path is now implemented for comparison
- both backends are exposed through `chronicle benchmark-stage2`
- Stage 2 now uses separate Chronicle-managed runtimes for backend experimentation rather than trying to force one shared env

Current benchmark direction on this machine:
- pyannote is workable but scales poorly on longer CPU samples
- the refined SpeechBrain-style path scales materially better on `30s` and `60s` windows
- SpeechBrain is the current local implementation baseline
- pyannote remains the comparison baseline until output quality is reviewed more carefully

### Step 3: define the Stage 2 artifact contract

Decide the machine and markdown outputs for:
- anonymous speaker labels
- timestamps
- overlap handling
- diarization confidence and notes

### Step 4: redesign current heuristic logic as Stage 3 identification

After Stage 2 exists:
- redesign the current text-first heuristic logic as identity reconciliation
- combine Stage 1 transcript text and Stage 2 anonymous speaker turns
- likely replace or augment heuristics with an LLM-assisted identifier

### Step 5: build Stage 4 organization on top of identified speakers

Only after Stage 3 is stable:
- implement organization by chronology and theme
- preserve verbatim language and provenance

## Current recommendation

Do not invest further in the current heuristic text-only Stage 2 as the long-term diarization solution.

The current best direction is:
1. keep Stage 1 largely as-is
2. continue Stage 2 backend evaluation with pyannote and SpeechBrain-style spikes
3. promote one backend into the real `chronicle diarize` path only after quality review
4. keep identity assignment in Stage 3
5. keep Stage 4 focused on organization, not attribution
