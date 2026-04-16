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

Current code is not fully aligned with that command surface yet. Today:
- `transcribe` is real Stage 1
- `diarize` still runs the old heuristic text-first logic
- `chronology` is still scaffold-only

So the public architecture target and the current code state are temporarily different. The first implementation step is to reconcile them.

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

Current code only implements `stage1/` and the older `stage2/` shape. The rest of the directory contract is part of the migration plan.

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

Current code still reflects the older 3-stage model:
- `src/chronicle/stage1/` is real transcription code
- `src/chronicle/stage2/` contains text-first heuristic speaker assignment logic
- `src/chronicle/stage3/` is only a chronology scaffold

That means the current code-state mapping is:
- old Stage 2 logic should become new Stage 3 logic
- old Stage 3 planning should become new Stage 4 planning
- new Stage 2 is not implemented yet

This mismatch is intentional to surface now rather than hide.

## Implementation plan

### Step 1: migrate code and artifact naming to the new stage model

Before building the new diarization stage, the repo should be realigned:
- migrate current `src/chronicle/stage2/` logic into a new `src/chronicle/stage3/`
- migrate current `src/chronicle/cli/stage2.py` behavior into a new Stage 3 command surface
- move current `src/chronicle/stage3/` scaffold forward into Stage 4 planning/code locations
- update stage directory expectations from the old `stage1/stage2/stage3` model to the new `stage1/stage2/stage3/stage4` model
- keep artifact contracts explicit during migration rather than silently reusing old names

This is the first implementation step because adding a new diarization Stage 2 on top of the old numbering would make the code and docs harder to reason about.

### Step 2: research and choose a local Stage 2 diarization stack

Once the numbering and code layout match the intended architecture:
- evaluate local diarization options
- compare local-machine feasibility
- decide how to support optional speaker-count hints

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
2. add a true anonymous audio-diarization Stage 2
3. move identity assignment into Stage 3
4. keep Stage 4 focused on organization, not attribution
