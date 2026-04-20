---
name: plan-design
description: Produce a structured implementation plan from frozen requirements, including design overview, bounded task decomposition, dependencies, risks, and execution strategy.
---

# Purpose

Use this skill to turn frozen requirements into a concrete implementation plan.

This skill is for the design agent. It drafts the plan. It does not own final approval.

# When to Use

Use when:
- requirements have been gathered and frozen
- the manager requests a structured implementation plan

Do not use when:
- requirements are still incomplete or unstable
- the task is still in exploratory discovery mode

# Inputs

- Frozen requirements
- Relevant repo docs
- Relevant architecture/schema/pipeline docs
- Any manager-provided constraints on scope or implementation

# Core Principles

- Design from requirements, not from guesses.
- Keep scope bounded.
- Prefer implementable plans over theoretical completeness.
- Decompose into tasks that an implementation manager can assign clearly.
- Make dependencies explicit.
- Surface risks and edge cases early.
- Do not introduce new requirements.

# Responsibilities

The design agent must produce a plan that includes:

1. Objective
2. Requirements summary
3. Non-goals
4. Constraints
5. Design overview
6. Data model / interfaces / contracts where relevant
7. Bounded task breakdown
8. Execution strategy
9. Acceptance criteria
10. Risks / edge cases
11. Open questions
12. Implementation readiness notes

# Task Decomposition Rules

Tasks must be:
- bounded
- implementable
- reviewable
- testable

Each task should include:
- task name
- goal
- owned scope
- likely files/modules
- dependencies
- whether it can run in parallel
- expected validation

Avoid giant tasks like:
- "refactor backend"
- "fix UI"
- "update pipeline"

Prefer concrete tasks like:
- "add pick protection fields to canonical asset schema"
- "generate gold lineage edge list from silver event_assets"
- "add timeline hover metadata contract for player nodes"

# Execution Strategy

Explicitly state:
- which tasks are sequential
- which tasks may run in parallel
- key ordering constraints
- any migration or rollout concerns

# Risks and Edge Cases

List meaningful risks such as:
- ambiguous source data
- migration complexity
- contract mismatch
- backward compatibility issues
- visualization/data model mismatch
- environment or deploy risks

# Open Questions

Only list true unresolved questions.

Do not leave avoidable ambiguity in the final design draft.

# Prohibited Behavior

Do not:
- invent new product goals
- expand scope beyond frozen requirements
- skip task decomposition
- produce vague plans with no ownership boundaries
- rely on future implementation agents to resolve major design ambiguity

# Output Format

Return a structured markdown draft with:

- Title
- Objective
- Requirements
- Non-Goals
- Constraints
- Design Overview
- Data Model / Interfaces
- Task Breakdown
- Execution Strategy
- Acceptance Criteria
- Risks / Edge Cases
- Open Questions
- Implementation Readiness

# Style Guidance

- Be concrete.
- Be implementation-oriented.
- Prefer clarity over elegance.
- Keep tasks small enough that implementation agents can work with minimal ambiguity.