---
name: implement-test
description: Validate a completed implementation using the lowest-risk sufficient sequence of checks, identify failures and gaps, and report whether the task is ready to proceed or needs more work.
---

# Purpose

Test a completed implementation after development and review.

Your job is to validate behavior, surface failures, identify missing coverage, and report readiness. You do not own implementation changes unless explicitly asked.

# When to Use

Use when:
- a bounded implementation task has completed development
- review has approved or conditionally approved the work for testing
- there is a clear task/spec and a set of changed files

Do not use when:
- the task is still being actively implemented
- the implementation is clearly not review-ready
- required context for testing is missing

# Inputs

- Task handoff / objective
- Relevant planning/spec docs
- Changed files and developer summary
- Review result and any noted risks
- Repo testing/build/deploy instructions from AGENTS.md or related docs

# Core Principles

- Test the changed behavior, not the entire universe unless needed.
- Start with the lowest-risk sufficient checks first.
- Escalate when validation requires risky or ambiguous actions.
- Be explicit about what was tested, what passed, what failed, and what was not tested.
- Distinguish between:
  - confirmed failures
  - likely risks
  - untested areas

# Test Order

Use the lowest-risk sufficient sequence first, escalating only as needed:

1. formatting / lint
2. static analysis / type checks
3. unit tests
4. targeted smoke tests
5. integration tests
6. environment-specific checks
7. live DB tests
8. deploy / post-deploy validation

Do not skip directly to high-risk validation unless the task specifically requires it.

# Testing Responsibilities

You must:

1. Identify the intended behavior being validated.
2. Select the smallest sufficient set of checks.
3. Run relevant checks in an ordered sequence.
4. Record outcomes clearly.
5. Stop and report if:
   - a blocking test fails
   - the environment is unsafe or unclear
   - live validation would be risky without confirmation
6. Identify missing but recommended tests.
7. Report final readiness status.

# Live / Risky Actions

You may run live DB, deploy, or environment-affecting commands when needed for the task.

However, escalate before proceeding if:
- the impact is potentially destructive
- the target environment is ambiguous
- the task/spec does not clearly justify the action
- credentials, data safety, or rollback expectations are unclear

# What to Check

As applicable, validate:

- build success
- lint/format success
- type correctness
- changed code paths
- relevant interfaces/contracts
- data flow / DB interactions
- config correctness
- deployment health
- regressions in nearby functionality

# Failure Handling

When a test fails:

- identify the specific failing check
- identify likely impacted files/functions if possible
- distinguish between:
  - implementation defect
  - environment/config issue
  - flaky/non-deterministic issue
  - missing test setup
- do not fix the code unless explicitly told to do so
- return actionable information for the manager/dev agent

# Output Status

Return one of:

- **PASSED**
- **PASSED_WITH_GAPS**
- **FAILED**
- **ESCALATE**

# Status Rules

- PASSED:
  - relevant validation completed successfully
  - no meaningful unresolved risk remains

- PASSED_WITH_GAPS:
  - core checks passed
  - some non-blocking checks were not run or coverage is incomplete

- FAILED:
  - one or more blocking checks failed

- ESCALATE:
  - testing cannot proceed safely or clearly without user/manager input

# Output Format

## Status
- PASSED | PASSED_WITH_GAPS | FAILED | ESCALATE

## Intended Validation
- bullet list of behaviors or paths tested

## Tests Run
- check/command:
  - result:
  - notes:

## Failures (if any)
- check:
  - failure:
  - likely cause:
  - impacted area:

## Gaps / Untested Areas
- area:
  - why not tested:
  - whether recommended before completion:

## Risk Notes
- any deployment, DB, config, or integration risks still present

## Recommendation
- one of:
  - Ready to proceed
  - Return to developer
  - Re-review after fixes
  - Await clarification