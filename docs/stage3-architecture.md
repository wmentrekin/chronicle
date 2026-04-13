# Stage 3 Architecture

This document explains the current state of Stage 3 in code.

## Current state

Stage 3 is not implemented as a real transformation stage yet.
It is currently a scaffold that defines planned output locations for chronology artifacts.

## Current Stage 3 module layout

### `src/chronicle/stage3/service.py`

Purpose:
- expose `planned_stage3_artifacts(...)`

That function takes:
- a `SessionManifest`
- a Stage 3 output directory

and returns the planned chronology artifact paths, one per primary interviewee.

## How Stage 3 is currently used

The current Stage 3 CLI behavior lives in:

- `src/chronicle/cli/stage3.py`

When someone runs:

```bash
chronicle chronology <session_id>
```

the current flow is:

1. validate the session
2. create/resolve the Stage 3 output directory
3. render the planned artifact paths
4. tell the operator that Stage 3 is still scaffold-only

No chronology extraction is performed yet.

## Why this still has a separate module

Even though Stage 3 is minimal today, keeping it as an explicit module is useful because it:
- defines the expected output naming pattern
- keeps the CLI and future implementation aligned
- preserves a stable place for Stage 3 growth later

That is consistent with Chronicle's preference for explicit module boundaries instead of letting future code accrete into unrelated files.
