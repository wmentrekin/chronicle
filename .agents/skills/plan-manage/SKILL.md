---
name: plan-manage
description: Orchestrate requirements gathering, requirements freeze, design, review, iteration, and final plan document creation for one bounded feature or plan stage.
---

# Purpose

Use this skill to manage the full planning workflow for one feature, refactor, system change, or bounded plan stage.

This skill is for the planning/orchestration agent. It owns:
- requirements gathering
- requirements freeze
- design/review loop
- final plan doc creation

The final output must be a clean implementation-ready markdown plan in `docs/plans/`.

# When to Use

Use when:
- the user wants to plan a feature or bounded stage of work
- the implementation is not yet sufficiently specified
- a clean plan doc is needed to feed implementation skills

Do not use when:
- the task is already fully planned and ready for implementation
- the user only wants brainstorming without producing an implementation-ready plan

# Inputs

Expected inputs:
- user request
- relevant repo docs
- source-of-truth docs from AGENTS.md
- existing project constraints
- optional prior planning docs

# Core Principles

- Separate requirements from design.
- Do not allow design to proceed until requirements are sufficiently complete.
- Freeze requirements before design.
- Keep the manager focused on orchestration and decision control.
- Use bounded design/review loops.
- Escalate ambiguity instead of inventing requirements.
- Final plan must be implementation-ready.

# Workflow

## Stage 1: Requirements Gathering

The manager must gather requirements through direct conversation with the user.

This stage may include:
- asking clarifying questions
- reading relevant repo docs
- doing targeted research if needed

The goal is to produce a structured requirements set containing:
- problem statement
- goals
- non-goals
- constraints
- assumptions
- acceptance criteria
- open questions

If key information is missing, continue gathering requirements before moving on.

## Stage 2: Requirements Freeze Gate

Before design begins, confirm that requirements are sufficiently complete.

Requirements are frozen when:
- goals are clear
- non-goals are clear
- constraints are known
- acceptance criteria exist
- open questions are resolved or explicitly deferred

If requirements change materially after freeze:
- either reopen planning
- or create a new iteration/version of the plan

Do not silently absorb new scope into the frozen plan.

## Stage 3: Spawn Design Agent

Once requirements are frozen, spawn a design subagent using the frozen requirements and relevant context only.

The design subagent must produce a structured implementation plan draft.

## Stage 4: Spawn Review Agent

Spawn a review subagent to critique the design draft.

The reviewer must evaluate:
- alignment to frozen requirements
- feasibility
- scope control
- implementation readiness
- risks
- missing task decomposition
- unclear dependencies
- unresolved ambiguities

## Stage 5: Controlled Design/Review Iteration

If review finds issues, send the draft back to the design agent for revision.

The manager controls this loop.

Default rule:
- allow up to 3 design/review cycles
- after that, either:
  - accept if good enough
  - escalate to the user
  - stop and request requirement clarification

## Stage 6: Ready-for-Implementation Contract

Do not finalize the plan until all of the following are true:
- requirements are frozen
- design aligns with requirements
- task breakdown is bounded and clear
- dependencies are explicit
- acceptance criteria are clear
- risks and edge cases are listed
- open questions are empty or explicitly marked non-blocking
- implementation manager should not need major clarification to proceed

## Stage 7: Final Plan Document

Write the final approved plan to:
- `docs/plans/<descriptive-plan-name>.md`

The plan doc should be concise, structured, and ready to feed into `implement-manage`.

# Escalation Rules

Escalate to the user when:
- requirements conflict
- a material product or architecture choice is unresolved
- repo context is insufficient
- review and design continue to disagree after bounded iterations
- new scope appears after freeze and changes the plan materially

Do not make silent product or architectural assumptions.

# Required Final Plan Structure

Every final plan must include:

- Title
- Objective
- Requirements
- Non-Goals
- Constraints
- Design Overview
- Data Model / Interfaces (if applicable)
- Task Breakdown
- Execution Strategy
- Acceptance Criteria
- Risks / Edge Cases
- Open Questions
- Implementation Readiness

`Open Questions` should ideally be empty by finalization.

# Final Report to User

After writing the plan, return:

## Plan Output
- path to plan doc
- short summary of what was planned

## Requirements Status
- frozen / not frozen

## Review Status
- approved / approved with caveats / blocked

## Notes
- any remaining non-blocking concerns
- any follow-up planning recommended

# Style Guidance

- Be structured and concise.
- Ask targeted questions.
- Separate requirements from solution design.
- Do not let planning drift into implementation.
- Produce plans that reduce ambiguity for implementation as much as possible.