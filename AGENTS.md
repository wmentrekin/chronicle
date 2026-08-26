# AGENTS.md

## Shared Workflow Framework

This repo imports a shared agent workflow at `.agents/` (git subtree). Read `.agents/AGENTS.md`
before starting substantial work; invoke it with `/work`.

---

## Purpose

This repository implements **Chronicle**, a local-first, multi-stage, agentic pipeline for transforming raw interview audio into structured, reviewable source material.

Core stages:

1. Transcription
2. Diarization
3. Speaker Identification
4. Organization scaffold

The system is:

* CLI-driven (`chronicle`)
* file-system based (not database-first)
* privacy-sensitive (all processing is local by default)

Agents must preserve these properties.

---

## Core Architecture

Chronicle is built around **strict stage boundaries + filesystem contracts**.

### Inputs (source of truth)

```
inputs/
  global/
    participants.yaml
  sessions/
    <session_id>/
      audio/
      session.yaml
      context.md
```

### Outputs (derived artifacts)

```
outputs/
  <session_id>/
    stage1/
    stage2/
    stage3/
    stage4/
    runs/
```

These directories are:

* **stateful**
* **incrementally built**
* **not committed to git**

Agents must treat:

* `inputs/` as immutable source
* `outputs/` as deterministic derivations

Never mix responsibilities between them.

---

## Key Architectural Rules

### 0. Git Branch & PR Isolation (Critical)

Whenever executing feature work or pipeline stage changes via `/work`:

* **Never commit or execute stage feature work directly on `main`.**
* Immediately after the pre-execution checkpoint, create and switch to a feature branch (`git checkout -b work/<feature-slug>`).
* Subagents and execution tasks must operate on the feature branch.
* Complete execution by pushing the feature branch and opening a pull request via GitHub CLI (`gh pr create --fill` or `gh pr create --title ... --body ...`).
* Verification findings must be reported on the PR.

---

### 1. Stage Isolation (Critical)

Each stage must:

* only depend on:

  * `inputs/`
  * prior stage outputs
* produce:

  * clearly scoped artifacts in its own stage directory

No stage should:

* reach forward to later stages
* re-implement logic from earlier stages
* mutate previous stage outputs

---

### 2. Deterministic Outputs

Given the same:

* inputs
* config
* model versions

Outputs should be reproducible.

Avoid:

* hidden randomness
* implicit global state
* non-versioned transformations

---

### 3. Local-First Constraint

This system is designed to:

* run fully offline (or close to it)
* keep sensitive data local

Agents must:

* not introduce remote dependencies without explicit justification
* not leak transcript or audio data externally
* prefer local models (e.g. Ollama, SpeechBrain)

---

### 4. File-System Contracts > In-Memory Assumptions

All stages communicate via:

* files
* structured artifacts

Do not:

* rely on in-memory passing between stages
* create hidden intermediate state

If a stage needs data → it should read it from disk.

---

## CLI Interface

Primary entrypoint:

```bash
chronicle <command> <session_id>
```

Key commands:

```bash
chronicle init
chronicle validate <session_id>
chronicle transcribe <session_id>
chronicle diarize <session_id>
chronicle identify <session_id>
chronicle organize <session_id>
chronicle run <session_id>
```

Agents must:

* maintain CLI stability
* not introduce breaking changes without coordination
* ensure new features integrate cleanly into this flow

---

## Skills and Workflow

### When to Use Planning Skills

Use:

* `plan-manage`
* `plan-design`
* `plan-review`

When:

* introducing new stages
* modifying stage contracts
* changing file formats
* adding new model backends
* altering CLI behavior

Planning output → `docs/plans/`

---

### When to Use Implementation Skills

Use:

* `implement-manage`
* `implement-develop`
* `implement-review`
* `implement-test`

When:

* implementing scoped features
* fixing bugs
* adding backend options
* improving performance

---

## Subagent Strategy

* Manager agent stays high-level
* Dev agents operate per-stage or per-feature
* Review agents enforce:

  * correctness
  * stage isolation
  * CLI consistency
* Test agents validate:

  * file outputs
  * stage transitions
  * CLI execution

Escalate uncertainties instead of guessing:

* file formats
* naming conventions
* stage contracts

---

## Repository-Specific Expectations

### Stage 1 (Transcription)

* Input: audio files
* Output: unified transcript artifact
* Must support multiple backends (e.g. faster-whisper, parakeet)

### Stage 2 (Diarization)

* Input: Stage 1 transcript + audio
* Output: speaker-segmented structure
* Default: SpeechBrain
* Optional: Pyannote (benchmarking)

### Stage 3 (Speaker Identification)

* Input: Stage 2 + participants.yaml + context
* Output: mapped speaker identities
* Modes:

  * `llm`
  * `manual`
  * `align-only`

### Stage 4 (Organization)

* Input: identified transcript
* Output: structured narrative scaffold

Agents must not blur these responsibilities.

---

## Testing Strategy

Prefer lowest-cost validation first:

1. CLI execution
2. File output validation
3. Stage-to-stage compatibility
4. Backend-specific validation
5. End-to-end run

Examples:

```bash
chronicle validate <session_id>
chronicle run <session_id>
```

Avoid:

* unnecessary full pipeline runs
* reprocessing large audio unless required

---

## Model and Dependency Constraints

Defined in `pyproject.toml`.

Important:

* Python 3.11 only
* optional dependency groups per stage

Agents must:

* not break optional backend modularity
* not introduce heavy deps into base install
* keep stage dependencies isolated when possible

---

## Data and Privacy Rules

This repo handles:

* personal interviews
* potentially sensitive family history

Rules:

* never expose raw data externally
* avoid logging sensitive content
* NEVER put actual family names, personal names, or sensitive identifiers in git commit messages, branch names, PR titles, or PR descriptions. Always use session IDs (e.g. `2026-02-24_interview_paternal-grandparents`) or generic participant roles (`interviewer`, `paternal_grandmother`, `paternal_grandfather`).
* do not commit anything under:

  * `inputs/`
  * `outputs/`
  * `models/`

Use:

* `examples/inputs/` for public-safe scaffolding

---

## Task Naming and Tracking

Format:

```
<phase>-<area>-<short-name>
```

Examples:

* `plan-stage3-llm-alignment`
* `impl-stage2-speechbrain-tuning`
* `impl-cli-run-improvements`

Statuses:

* `todo`
* `in_progress`
* `in_review`
* `in_testing`
* `blocked`
* `complete`

Persist in:

* plan docs
* or `docs/plans/TASKS.md`

---

## Documentation Rules

Update docs when changing:

* stage contracts
* file formats
* CLI commands
* backend behavior
* model assumptions

Key locations:

* `README.md`
* `docs/`
* `examples/inputs/`

---

## Escalation Rules

Escalate when:

* stage boundaries are unclear
* output formats are ambiguous
* model choice impacts UX or cost
* privacy implications exist
* large audio reprocessing is required

Do not assume:

* speaker identity logic
* diarization correctness thresholds
* transcript formatting standards

---

## Definition of Done

A task is complete when:

* CLI command works
* outputs are correct and placed properly
* stage contracts are preserved
* review issues resolved
* validation passes
* docs updated if needed

---

## Key Rule

> Chronicle is a deterministic, stage-based, local-first pipeline.
> Preserve stage isolation and file-based contracts above all else.
