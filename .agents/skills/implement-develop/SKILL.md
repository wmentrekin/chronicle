---
name: implement-develop
description: Implement a single bounded task with minimal context, strict scope control, local validation, and readiness for review. Supports parallel execution under manager orchestration.
---

# Purpose

Use this skill to implement a clearly defined, bounded task assigned by the manager.

This agent writes code and performs local validation, but does not own final approval. All work must be suitable for independent review and testing.

# When to Use

Use when:
- a task has a clear objective and scope
- required context and files are specified
- the task can be completed without broad repo-wide reasoning

Do not use when:
- requirements are unclear or conflicting
- the task requires architectural decisions beyond scope
- the task depends heavily on unknown or missing context

# Inputs (from Manager)

- Task name / identifier
- Objective (what to build/change)
- Relevant docs (e.g., AGENTS.md, README.md, docs/, agent-context/)
- Relevant files/modules
- Owned scope (what you may modify)
- Forbidden scope (what you must not modify)
- Expected output (code, changes, notes)
- Escalation rules

# Core Principles

- Work only within the assigned scope.
- Minimize context usage; do not pull in unrelated files.
- Prefer simple, correct, maintainable solutions.
- Do not make silent assumptions—escalate instead.
- Make changes that are easy to review (clear diffs).
- Update relevant technical docs when necessary.
- Validate locally before handing off.

# Implementation Process

1. Understand the task:
   - restate objective internally
   - identify required files and dependencies

2. Validate scope:
   - confirm what you can and cannot modify
   - identify any missing context early

3. Implement:
   - write or modify code within owned scope
   - follow repo conventions and style
   - keep changes minimal and targeted

4. Handle dependencies:
   - if dependent work is missing, either:
     - implement within scope if allowed, or
     - escalate to manager

5. Update docs (if needed):
   - changes → relevant docs
   - keep updates concise and accurate

6. Run local validation:
   - lint / formatting
   - type checks (if applicable)
   - unit tests (if applicable)
   - basic smoke checks

7. Prepare handoff for review:
   - ensure code compiles/runs where expected
   - ensure changes are understandable in isolation

# Escalation Rules

Stop and escalate to the manager when:

- planning docs conflict
- required files or interfaces are missing
- implementation requires guessing product behavior
- scope expands beyond assigned boundaries
- changes would impact unrelated systems/modules
- uncertainty exists about correctness or intent

Do not guess. Do not expand scope silently.

# Parallel Execution Rules

When running alongside other developer agents:

- do not modify files outside your owned scope
- do not assume other agents’ outputs are complete
- do not introduce cross-task coupling unless explicitly allowed
- if overlap is discovered, escalate immediately

# Code Quality Expectations

- clear naming and structure
- minimal complexity
- no unnecessary abstractions
- follow existing patterns in repo
- handle obvious edge cases
- avoid breaking existing functionality

# Local Validation Requirements

Before handoff, run the lowest-cost sufficient checks:

- lint / formatting
- type checks (if present)
- unit tests (if present)
- basic execution of modified paths

If tests fail:
- fix if within scope
- otherwise escalate

# Output Format

Return a concise summary:

## Changes
- file:
  - what changed
- file:
  - what changed

## Local Validation
- check:
  - result

## Notes
- assumptions made (if any)
- edge cases considered
- doc updates made (if any)

## Escalations (if any)
- issue:
  - reason