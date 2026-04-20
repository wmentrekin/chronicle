---
name: plan-review
description: Critique an implementation plan draft against frozen requirements, feasibility, scope, task decomposition quality, risks, and implementation readiness.
---

# Purpose

Use this skill to review a plan draft before it is finalized for implementation.

This skill is for the review agent. It critiques and validates. It does not own redesign.

# When to Use

Use when:
- a design draft has been produced from frozen requirements
- the manager wants an independent review before finalizing the plan

Do not use when:
- requirements are not frozen
- no concrete plan draft exists

# Inputs

- Frozen requirements
- Plan draft
- Relevant repo/source-of-truth docs
- Any manager notes about constraints or priorities

# Core Principles

- Review against requirements first.
- Check whether the plan is actually implementable.
- Enforce bounded scope.
- Ensure task decomposition is usable by implementation agents.
- Surface risks and missing details clearly.
- Be specific and actionable.

# Review Checklist

## 1. Requirements Alignment
- Does the plan address the frozen requirements?
- Does it avoid introducing new requirements?
- Are acceptance criteria represented clearly?

## 2. Scope Control
- Is scope bounded and realistic?
- Are non-goals respected?
- Is anything included that should be deferred?

## 3. Feasibility
- Is the plan implementable in the current repo/system context?
- Are dependencies realistic?
- Are rollout/migration expectations plausible?

## 4. Task Decomposition Quality
- Are tasks bounded?
- Are ownership boundaries clear?
- Are dependencies explicit?
- Can implementation manager delegate tasks cleanly?
- Are parallelizable tasks identified appropriately?

## 5. Risk Coverage
- Are major risks listed?
- Are edge cases acknowledged?
- Are important unknowns called out?

## 6. Implementation Readiness
- Would `implement-manage` likely need major clarification?
- Are open questions blocking?
- Is the plan concrete enough for implementation?

# Decision Outcomes

Return exactly one of:

- **APPROVED**
- **APPROVED_WITH_CHANGES**
- **CHANGES_REQUIRED**
- **ESCALATE**

# Decision Rules

- APPROVED:
  - aligned, feasible, bounded, and implementation-ready

- APPROVED_WITH_CHANGES:
  - small improvements needed but plan is basically sound

- CHANGES_REQUIRED:
  - meaningful missing pieces, weak decomposition, scope issues, or feasibility concerns

- ESCALATE:
  - conflicting requirements, unresolved architectural/product ambiguity, or missing information that design cannot responsibly resolve

# Feedback Requirements

Feedback must be:
- specific
- actionable
- tied to requirements, feasibility, scope, or implementation readiness

Do not give vague criticism.

# Output Format

## Decision
- APPROVED | APPROVED_WITH_CHANGES | CHANGES_REQUIRED | ESCALATE

## Summary
- 1–3 bullets summarizing the plan quality

## Issues
- section:
  - issue:
  - why it matters:
  - required change:

## Task Decomposition Gaps
- task/area:
  - gap:
  - recommendation:

## Risk Gaps
- risk:
  - missing or weak coverage:

## Open Question Status
- blocking / non-blocking
- list any unresolved items

## Implementation Readiness
- ready / not ready
- short explanation

## Escalation
- question:
  - why it must go back to manager/user