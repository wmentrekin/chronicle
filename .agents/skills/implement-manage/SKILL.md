---
name: implement-manage
description: Orchestrate implementation of one planned feature or one bounded plan stage by delegating work to developer, reviewer, and tester subagents with minimal relevant context; manage iteration, escalation, documentation updates, and final reporting.
---

# Purpose

Use this skill when the task is to implement a defined feature, plan stage, or bounded engineering task using subagents.

This skill is for the manager/orchestrator agent. Its job is to coordinate implementation work, not to carry full project context or do most of the coding itself.

The manager may make small direct fixes when efficient, but all code changes must still be reviewed and tested.

# When to Use

Use this skill when:
- the user provides one or more planning docs or a clearly defined implementation task
- the task can be decomposed into one or more bounded implementation tasks
- parallel developer subagents may help
- the work should go through implementation, review, testing, documentation updates, and final reporting

Do not use this skill when:
- the task is still mostly exploratory or under-specified
- the user is asking for planning rather than implementation
- the change is so small that subagent orchestration would add unnecessary overhead

# Inputs

Expected inputs:
- one or more planning/spec/design docs
- an implementation request from the user
- any relevant source-of-truth docs already in the repo
- repo-specific commands and constraints from AGENTS.md

# Core Principles

- Keep the manager context thin.
- Delegate implementation to subagents with only the minimum relevant context.
- Prefer bounded tasks with explicit ownership.
- Allow parallel developer subagents only when scopes do not materially overlap.
- Require explicit escalation rather than silent assumptions.
- Persist important outcomes into files/docs, not only chat history.
- Ensure all changes are reviewed and tested before being considered complete.
- Documentation updates are part of done.

# Responsibilities

The manager must:

1. Read the implementation request and relevant planning docs.
2. Determine whether the task is sufficiently specified.
3. Decompose the work into one or more bounded implementation tasks.
4. Decide whether tasks can be delegated in parallel or must be sequential.
5. Spawn developer subagents with minimal but sufficient context.
6. Ensure each developer subagent receives:
   - a clear objective
   - relevant docs/files only
   - explicit owned scope
   - forbidden scope
   - expected output
   - escalation rules
7. Close developer subagents after they complete their task.
8. Spawn a reviewer subagent to critique implementation against:
   - planning docs
   - task handoff
   - correctness
   - code quality
   - programming principles
   - integration risks
   - documentation completeness
9. If review fails, route work back to the same developer agent for revision when possible.
10. Spawn a tester subagent to validate the completed work.
11. Ensure docs are updated when code, workflows, interfaces, or assumptions changed.
12. Report final status to the user in the required format.

# Task Decomposition Rules

Decompose into bounded tasks that have:
- clear deliverables
- clear file/module ownership
- clear dependency ordering
- minimal overlap with sibling tasks

For parallel developer subagents:
- assign non-overlapping owned files/functions/modules where possible
- state dependencies explicitly
- if tasks overlap materially, do not run them in parallel

# Escalation Rules

The manager and all subagents must escalate to the user when:
- planning docs conflict
- critical context is missing
- a meaningful product or architecture assumption would be required
- scope expands beyond the planned task
- live actions could be risky or destructive
- the manager cannot confidently determine task boundaries

Do not silently invent requirements.

# Direct Manager Edits

The manager may make small direct code or documentation changes when doing so is clearly lower overhead than spawning another implementation subagent.

Examples:
- tiny wiring fixes
- obvious config updates
- small documentation corrections
- minor follow-up cleanups after review

Any direct edits must still go through review and testing.

# Developer Handoff Requirements

When spawning a developer subagent, provide:
- implement-develop skill instructions and references
- objective
- task identifier/name
- relevant planning docs
- relevant source-of-truth docs
- owned scope
- forbidden scope
- expected output
- escalation instructions
- reminder to update relevant technical docs when necessary
- reminder to run appropriate local tests before handing off

# Review Requirements

Provide the reviewer with:
- implement-review skill instructions and references
- the original planning docs and developer handoff
- the completed implementation output

The reviewer must:
- critique implementation against spec and handoff
- check for scope violations
- check for missing edge cases
- check for maintainability and sound engineering principles
- check whether docs should have been updated
- return either:
  - approved
  - approved with minor follow-up
  - changes required
  - escalate to user

The reviewer should not directly rewrite the task unless the manager explicitly requests that behavior elsewhere.

# Testing Requirements

Provide the tester with:
- implement-test skill instructions and references
- the original planning docs and developer handoff
- the completed implementation output

The tester must validate the finished implementation using the lowest-risk sufficient sequence first, escalating to stronger validation only as needed.

Preferred order:
1. lint / formatting / static analysis
2. type checks
3. unit tests
4. smoke tests
5. integration tests
6. live DB / deploy / environment validation
7. any project-specific high-risk checks

The tester may run live or environment-affecting commands when needed, but should escalate if risk is material or ambiguous.

# Documentation Requirements

The manager must ensure relevant docs are updated when necessary, including:
- planning/progress docs
- schema docs
- pipeline docs
- runbooks
- setup instructions
- usage notes
- task tracking docs

Documentation updates may be performed by developer agents, the manager, or both, but they must be explicitly checked before completion.

# Completion Standard

A task or plan stage is complete only if:
- implementation is done
- review has passed or review feedback has been resolved
- testing has been completed at an appropriate level
- relevant docs have been updated
- unresolved issues are clearly reported

# Recommended Execution Flow

1. Read task + planning docs.
2. Confirm source-of-truth docs and constraints.
3. Decompose into bounded tasks.
4. Spawn 1..N developer subagents as appropriate.
5. Collect outputs and summarize minimally.
6. Spawn review subagent.
7. If review requires changes, send back to the same developer agent when possible.
8. Repeat review/develop loop as needed.
9. Spawn testing subagent.
10. Update/verify docs.
11. Return final report to user.

# Final Report Format

Return a concise report with these sections:

## Changes Made (Files & Functions)
- bullet list of files changed
- bullet list of key functions/modules/components added or modified
- bullet list of relevant doc updates

## Tests Completed
- bullet list of tests/checks run
- include notable outcomes
- mention any checks intentionally not run

## Status
- one of:
  - Complete
  - Complete with follow-up recommended
  - Needs more work
  - Needs more testing
  - Blocked / awaiting clarification

## Notes / Open Questions
- only include if needed
- list remaining risks, assumptions, or user decisions required

# Style Guidance

- Be structured and concise.
- Prefer explicit task boundaries over broad instructions.
- Do not carry unnecessary context forward.
- Do not treat subagents as long-lived memory stores.
- Persist durable outcomes into repo docs/files, then compact.