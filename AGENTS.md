# Family Oral History Pipeline - AGENTS.md

## What this repository is

This repository is the working pipeline for a private family oral-history and life-story project.

The long-term goal is to take raw interview audio and turn it into narrative-ready source material that can later become a polished life story document.

This repository is not just a transcription sandbox anymore. It is the implementation workspace for a reproducible multi-step pipeline that should eventually support new interviews with minimal manual setup.

## Long-term vision

The end state of this project is:
1. ingest raw interview audio and session context
2. transcribe the audio accurately
3. separate anonymous speaker turns from the audio
4. map anonymous speakers to real people conservatively
5. reorganize the verbatim content into chronology and theme-based source documents for each interviewee
6. later support narrative drafting from those verified source documents

The intended output is not a polished biography produced in one pass.

The intended output is a chain of reviewable archival artifacts that move from raw source material toward a narrative life-story document without losing provenance.

## Current design constraints

- Run on the user's local machine for now.
- Cloud or LLM-assisted steps are allowed, but cost must be treated as a real constraint.
- Raw audio should remain untouched.
- Verbatim language should be preserved at all times.
- Uncertainty should be surfaced, not hidden.
- The workflow should be reproducible from the command line.
- Terminal execution should expose status in real time.

## Core principles

### 1. Preserve raw sources
Never overwrite, rename, trim, or delete files in `inputs/sessions/<session_id>/audio/` unless explicitly asked.

### 2. Preserve provenance
Each stage should create a new artifact rather than silently mutating the previous one.

### 3. Preserve verbatim language
Do not rewrite spoken language into polished prose during transcription, diarization, or chronology extraction.

Reorganization is allowed.
Paraphrase is not.

### 4. Separate evidence types
This repository will eventually support several kinds of outputs:
- raw transcription
- speaker-attributed conversation text
- chronology and theme-based excerpt documents
- later narrative synthesis

Do not collapse these into one another.

### 5. Be conservative with certainty
When the system is unsure, mark it explicitly.

Preferred confidence labels:
- `Confirmed`
- `Likely`
- `Unclear`
- `Needs review`

### 6. Optimize for reviewability
Every stage should be auditable by a human reviewer.

That means:
- stable file naming
- structured intermediate artifacts
- explicit notes
- references back to the source transcript or timestamps

### 7. Protect privacy
This repository contains private family materials.

Do not publish, upload, or distribute materials unless explicitly asked.

## Current repository layout

Use the structure that exists today unless explicitly changed:

- `inputs/global/participants.yaml`
  - canonical participant and family-reference metadata
- `inputs/sessions/<session_id>/`
  - one session input bundle per recording
- `inputs/sessions/<session_id>/audio/`
  - original interview recordings for that session
- `outputs/<session_id>/stage1/`
  - stage 1 artifacts for that session
- `outputs/<session_id>/stage2/`
  - stage 2 artifacts for that session
- `outputs/<session_id>/stage3/`
  - stage 3 artifacts for that session
- `outputs/<session_id>/stage4/`
  - stage 4 artifacts for that session
- `outputs/<session_id>/runs/`
  - per-stage run metadata
- `src/chronicle/`
  - packaged CLI, validation, and stage implementations
  - intentionally split into smaller responsibility-based modules rather than a few large files
- `docs/`
  - canonical repository documentation
- `agent-context/`
  - planning notes, implementation notes, and handoff context for future work
- `pyproject.toml`
  - canonical Python version, dependencies, and CLI entrypoint metadata
- `.python-version`
  - preferred local Python version for `uv`
- `AGENTS.md`
  - project instructions and architecture guidance

If a new derived directory is needed, create it intentionally instead of improvising ad hoc structure.

## Code organization preference

Prefer smaller modules split by responsibility over a small number of very large files.

For this repository, it is better to have:
- thin CLI command modules
- thin stage orchestration modules
- separate backend / artifact / benchmark / helper modules

than to keep accumulating logic in monolithic `service.py` files.

When refactoring:
- preserve the observable CLI behavior unless explicitly changing it
- preserve artifact contracts unless explicitly changing them
- move code along stable responsibility boundaries, not arbitrary line-count boundaries

## Near-term growth structure

The current structure is already the preferred working layout. Future additions should fit into it rather than reintroducing ad hoc folders.

Likely future additions:
- `docs/examples/`
  - sanitized sample inputs and outputs for public sharing later
- `src/chronicle/stage3/`
  - real speaker-identification logic
- `src/chronicle/stage4/`
  - real organization logic
- `src/chronicle/stage5/`
  - later narrative drafting logic if that stage is implemented
- `tests/`
  - validation and stage-level tests once the interfaces stabilize

## Canonical metadata rules

`inputs/global/participants.yaml` is the canonical metadata source for named people mentioned in interviews.

Agents are explicitly allowed to edit `inputs/global/participants.yaml` when needed to support the pipeline, including:
- adding scaffold person entries
- expanding parents and relationship links
- adding aliases and notes
- improving canonical naming consistency

When editing metadata:
- do not invent certainty
- prefer scaffold entries over omission when a person is clearly relevant
- keep canonical names stable unless there is a clear reason to change them
- avoid deleting information unless explicitly asked

## Session input contract

The intended per-session input should be lightweight and repeatable.

For each interview session, the pipeline should be able to run from:
1. one or more raw audio files
2. participant metadata from `inputs/global/participants.yaml`
3. a session context document that includes:
   - the participants in the conversation
   - people likely to be discussed and searchable in `participants.yaml`
   - two to three paragraphs of contextual background about the conversation

The user should not need to rebuild the pipeline prompting logic by hand for each new interview.

## Target pipeline architecture

### Stage 1 - Raw transcription

Goal:
Produce the most accurate possible transcript from raw audio without trying to identify speakers.

Preferred model family:
- NVIDIA Parakeet

Stage 1 should:
- accept `.m4a` input or convert other supported audio inputs as needed
- preserve timestamps when available
- output raw machine-readable transcription artifacts
- avoid injecting speaker assumptions
- avoid using family-context metadata to change the words

Stage 1 is about speech-to-text quality only.

### Stage 2 - Anonymous audio diarization

Goal:
Take raw session audio and infer anonymous speaker turns conservatively.

Stage 2 should use:
- raw session audio
- optional expected speaker count
- optional min/max speaker-count constraints

Stage 2 should:
- split the waveform into speaker turns
- assign anonymous speaker labels such as `SPEAKER_00`
- preserve uncertainty when clustering is weak
- support `n` speakers up to a reasonable limit
- avoid depending on biography-heavy context

The output should be an anonymous speaker-turn artifact, not a named-person transcript.

### Stage 3 - Speaker identification and reconciliation

Goal:
Combine the Stage 1 transcript with Stage 2 anonymous speaker turns and map those anonymous speakers to real people conservatively.

Stage 3 should:
- use Stage 1 transcript text
- use Stage 2 anonymous speaker turns
- use session context and participant metadata
- assign canonical names only when justified
- preserve uncertainty when identification is weak
- emit a readable speaker-attributed transcript plus a machine artifact

### Stage 4 - Verbatim chronology and theme extraction

Goal:
Transform a speaker-identified conversation into source documents that are useful for later narrative drafting.

Stage 4 should:
- create one organized document per interviewee per session
- reorganize verbatim excerpts under themes and approximate life periods
- preserve the original wording of the speaker
- include references back to transcript sections or timestamps when feasible
- clearly flag uncertainty, chronology gaps, and ambiguous references

The result should be organized, narrative-ready source material, not a polished life story.

### Future Stage 5 - Narrative synthesis

This is a later phase, not the immediate implementation target.

A future narrative stage may:
- draft a coherent life-story document
- weave together chronology excerpts and historical background
- preserve what is sourced from interviews versus what is later synthesis

That future stage must remain downstream of reviewed source artifacts.

## Output expectations by stage

### Stage 1 outputs
- raw transcript text
- machine-readable timestamped transcript artifact
- model and run metadata

### Stage 2 outputs
- anonymous speaker-turn artifact
- timestamps and speaker cluster labels
- notes for overlap or uncertain clustering

### Stage 3 outputs
- speaker-identified conversation transcript
- confidence label per speaking block
- notes for uncertain speaker assignment

### Stage 4 outputs
- one verbatim chronology document per interviewee per session
- sections organized by time period and theme
- notes for ambiguities and items needing manual review

## Standards for all machine-generated content

### Do
- preserve wording
- preserve meaningful repetition
- preserve uncertainty
- preserve emotional or anecdotal phrasing
- keep track of who said what and why the system thinks so

### Do not
- paraphrase testimony into polished prose
- silently correct chronology
- merge multiple voices into one narrative voice
- invent names, dates, relationships, or places
- suppress ambiguity that would matter to later review

### Allowed review tags
- `[unclear]`
- `[inaudible]`
- `[overlap]`
- `[needs review]`
- `[possible proper noun: ...]`

## Suggested implementation approach

Prefer a deterministic artifact pipeline over a vague autonomous system.

In practice, this means:
- explicit stage inputs and outputs
- structured schemas for machine artifacts
- prompt templates or configuration files checked into the repo
- small composable command-line commands
- reviewable markdown and JSON outputs at each stage
- repository-managed Python dependencies via `pyproject.toml` and `uv`
- a packaged CLI under `src/chronicle/`

Do not hide important transformation logic inside a single opaque prompt.

## Environment management

Prefer `uv` over manually managed virtual environments for ongoing work in this repo.

Current expectation:
- dependency metadata lives in `pyproject.toml`
- Python version intent lives in `.python-version`
- reproducible environments should be created with `uv sync`
- the default dependency install path should match the default Stage 1 runtime
- commands should generally be run as `chronicle ...` once the local environment is bootstrapped

Current bootstrap expectation:
1. run `./bin/bootstrap`
2. optionally run `./bin/bootstrap --install-link` if a stable user-level `chronicle` command is desired
3. alternatively, use `source .envrc` for repo-local PATH handling

`chronicle init` is the runtime/model initialization command after the environment exists.

Legacy environments such as `.venv-transcribe/` or experimental one-off environments may still exist locally, but they should not be treated as the long-term source of truth.

## Command-line workflow expectations

The pipeline should eventually be runnable from the terminal end to end.

Target operator experience:
1. add audio files
2. add or update participant metadata
3. add a session context document
4. bootstrap once, then use `chronicle` with a single command or staged subcommands
5. watch live status in the terminal
6. inspect the resulting artifacts

Current primary commands:
- `chronicle init`
- `chronicle validate <session_id>`
- `chronicle transcribe <session_id>`
- `chronicle diarize <session_id>`
- `chronicle identify <session_id>`
- `chronicle organize <session_id>`
- `chronicle run <session_id>`

## Planning and implementation roadmap

### Phase 1 - Planning documents
Create planning documents that define:
- target directory structure
- per-stage input and output contracts
- session context format
- artifact schemas
- evaluation criteria for transcript quality, diarization quality, speaker-identification quality, and organization usefulness

### Phase 2 - Stage 1 implementation
Implement the Parakeet-based transcription stage with reproducible local execution and stored run metadata.

### Phase 3 - stage migration
Before building the new diarization layer:
- move current `stage2` logic into new `stage3` ownership
- move current `stage3` scaffold into future `stage4` ownership
- realign CLI names, output paths, and docs with the new 4-stage model

### Phase 4 - Stage 2 implementation
Implement anonymous audio diarization as the new Stage 2.

### Phase 5 - Stage 3 implementation
Implement speaker identification and conservative name assignment using Stage 1 plus Stage 2 plus session context and participant metadata.

### Phase 6 - Stage 4 implementation
Implement per-interviewee organization that preserves verbatim excerpts and flags uncertainty.

### Phase 7 - CLI and operator workflow
Create a command-line entrypoint that can run individual stages or the full session pipeline and report progress live.

### Phase 8 - Skills and reusable docs
Turn each pipeline stage into:
- a skill
- an implementation document
- a repeatable operational workflow

### Phase 9 - Later narrative layer
After the earlier stages are stable, define the narrative drafting layer separately.

## Guidance for future agents working in this repo

When asked to do broad work here:
1. inspect the current metadata and relevant artifacts first
2. identify which pipeline stage or planning phase the task belongs to
3. preserve existing outputs unless explicitly replacing them
4. create new planning documents and skills when the architecture needs to be made explicit
5. summarize what was created, what remains uncertain, and what still needs review

When changing code structure:
- prefer updating the code layout and docs together
- use skills that the user references in `.agents/skills/`
- update `docs/architecture.md` and any relevant detailed docs under `docs/`
- update `agent-context/next-session-context.md` when the current working state changes materially

If a question requires an unstated assumption:
- do not guess
- record the uncertainty
- ask the user or mark it for planning review
