# Stage 3 Ollama Decomposition Plan

## Objective

Refine Stage 3 `llm` mode so local Ollama no longer receives one large speaker-map prompt. The implementation should complete on the target 8 GB CPU-only machine, preserve current Stage 3 modes and artifact paths, and produce reviewable canonical speaker assignments without rewriting transcript text.

## Requirements

- Keep Stage 1, Stage 2, and Stage 3 boundaries intact.
- Preserve `align-only`, `manual`, and `llm` modes.
- Keep all LLM processing local through Ollama only.
- Preserve Stage 1 transcript wording and Stage 2 anonymous speaker provenance.
- Assign only canonical manifest participants; do not invent unknown speakers or non-manifest names.
- Preserve the current Stage 3 artifact contract where possible: `identified_conversation.*`, `aligned_transcript.*`, `speaker_map`, `evidence_summary`, `alignment_summary`, `blocks`, and `llm_usage`.
- Record per-call usage and latency so local performance can be debugged.
- Fail safely with clear messages when Ollama, model setup, speaker-count policy, prompt budget, or decomposed calls cannot produce valid assignments.
- Keep tests independent of a real Ollama server by monkeypatching local API helpers.

## Non-Goals

- No hosted or remote LLM providers.
- No changes to Stage 1 transcription.
- No changes to Stage 2 acoustic diarization.
- No Stage 4 implementation.
- No transcript rewriting, paraphrasing, or cleanup in Stage 3.
- No new CLI command surface.
- No relaxed many-to-one or partial speaker mapping in this first implementation.

## Constraints

- Stage 3 `llm` and `manual` remain strict one-to-one mapping modes: every Stage 2 speaker label maps to exactly one canonical manifest participant, and each participant may be assigned at most once.
- If Stage 2 speakers outnumber manifest participants, fail before any Ollama call with guidance to inspect `align-only` or rerun Stage 2 with tighter speaker-count constraints.
- If manifest participants outnumber Stage 2 speakers, fail before any Ollama call for now because the current artifact contract assumes every manifest participant is represented in the session speaker set.
- Chronicle should describe this as supporting `n` speakers when `n` matches canonical session participants, not as supporting unequal speaker and participant counts.
- Candidate people must start as the full manifest participant set. If the full candidate set exceeds the configured prompt budget, fail before Ollama rather than silently truncating canonical candidates.
- Usage records must not store raw prompt text, near-raw prompt fragments, transcript excerpts, or context excerpts.
- The implementation must remain compatible with the current `chronicle identify` CLI and Stage 3 artifact filenames.
- Existing validation must continue rejecting duplicate assignments and non-manifest people.

## Design Overview

Replace the current single Ollama call in `src/chronicle/stage3/llm.py` with a speaker-centric decomposed workflow:

1. Build deterministic compact speaker dossiers from the existing `evidence_summary`.
2. Order unresolved anonymous speakers deterministically before calling Ollama.
3. Apply valid manual overrides first and exclude those speaker labels from LLM calls.
4. Issue one small Ollama call per unresolved speaker, using only that speaker's dossier, a compact session context excerpt, and the full canonical participant list.
5. Validate each returned assignment immediately, then validate the full aggregated map.
6. If validation fails, run exactly one repair pass targeting only unresolved, conflicting, or invalid speakers.
7. Aggregate per-call usage into `llm_usage` while keeping top-level Stage 3 artifact fields stable.

This bounds prompt size by speaker and candidate budget instead of total session length.

## Data Model / Interfaces

Final `speaker_map` entries should keep their current shape:

- `speaker_label`
- `assigned_person`
- `confidence`
- `candidate_people`
- `source`
- `evidence`
- `notes`

Extend `llm_usage` rather than replacing it. The extended shape should include:

- `provider`
- `model`
- `prompt_version`
- `schema_version`
- `workflow`: `decomposed`
- `input_tokens`: aggregate actual or estimated input tokens
- `output_tokens`: aggregate output tokens
- `call_count`
- `repair_call_count`
- `calls`
- `truncation_or_sampling_notes`

Each `calls` entry should contain only non-sensitive diagnostics:

- `phase`: `assignment` or `repair`
- `speaker_label`
- `attempt`
- `status`
- `input_tokens_estimated`
- `input_tokens_actual`
- `output_tokens`
- `latency_seconds`
- `prompt_hash`
- `content_hash`
- `notes`

Do not store prompt text or model response text in `llm_usage`.

`PROMPT_VERSION` should be bumped to a decomposed prompt version. `SCHEMA_VERSION` should remain unchanged unless implementation requires a breaking artifact change.

## Implementation Tasks

1. Implement deterministic speaker ordering and budget preflight.

   Owned files: `src/chronicle/stage3/llm.py`, `src/chronicle/stage3/evidence.py`, `src/chronicle/stage3/schemas.py`.

   Requirements:

   - Sort unresolved speakers deterministically, for example by descending `total_speaking_seconds`, then `turn_count`, then speaker label.
   - Include all canonical manifest participants in every assignment prompt.
   - Add configurable prompt budget constants for per-call input and output.
   - Estimate per-speaker prompt size before any Ollama call.
   - Fail before Ollama if the full candidate set or speaker dossier exceeds budget.

2. Build compact per-speaker and repair prompt constructors.

   Owned files: `src/chronicle/stage3/prompts.py`.

   Requirements:

   - Add a per-speaker assignment prompt builder.
   - Add a narrow repair prompt builder.
   - Keep prompts JSON-safe and deterministic.
   - Preserve verbatim evidence snippets only inside the prompt sent to local Ollama, not in usage metadata.
   - Use conservative instructions: manifest-only names, no transcript rewrite, no invented people, confidence must use existing labels.

3. Replace single-shot Ollama mapping with decomposed orchestration.

   Owned files: `src/chronicle/stage3/llm.py`.

   Requirements:

   - Keep `run_ollama_speaker_mapping(...)` as the service-facing interface unless a clearer internal wrapper is needed.
   - Call Ollama once per unresolved speaker.
   - Parse and validate each response as a single speaker-map entry.
   - Aggregate manual and LLM entries.
   - Run exactly one repair pass only if validation finds unresolved, duplicate, non-manifest, or otherwise invalid assignments.
   - Fail closed if repair does not produce a complete valid map.
   - Record per-call diagnostics without storing prompt text or response text.

4. Wire Stage 3 service and validation policy.

   Owned files: `src/chronicle/stage3/service.py`, `src/chronicle/stage3/identity.py`.

   Requirements:

   - Enforce the one-to-one speaker-count policy before any Ollama call.
   - Keep `align-only` exempt from the count check.
   - Keep `manual` strict and complete.
   - Keep artifact filenames unchanged.
   - Keep final `blocks` assignment behavior unchanged after a valid `speaker_map` exists.

5. Expand tests.

   Owned files: `tests/test_stage3.py` and any focused new test file if needed.

   Required coverage:

   - Count mismatch fails before Ollama for `llm`.
   - `align-only` remains allowed under count mismatch.
   - Deterministic speaker ordering.
   - Full candidate list inclusion.
   - Prompt budget failure before Ollama.
   - Decomposed calls equal unresolved speaker count on the happy path.
   - `llm_usage` includes aggregate fields and per-call diagnostics.
   - One repair pass succeeds for a targeted conflict.
   - Repair failure raises `StageExecutionError`.
   - No test requires a real Ollama server.

6. Add local benchmark or smoke validation.

   Owned files: existing benchmark/CLI surface if appropriate, plus docs.

   Requirements:

   - Provide a way to run the decomposed Stage 3 LLM path against a representative local session or fixture with a selected Ollama model.
   - Record model, call count, per-call latency, aggregate latency, prompt token estimates, timeout setting, and success/failure.
   - Default validation budget for the target machine: no individual call may exceed a 180-second request budget, and a three-speaker representative run should finish within a 600-second total smoke budget.
   - Budgets may be configurable, but the result must clearly report whether the default target-machine budget passed.
   - The benchmark should demonstrate that the decomposed workflow completes where the previous single-prompt workflow timed out.

7. Update documentation.

   Owned files: `docs/stages/stage3-identification.md`, `docs/artifacts.md`, `docs/cli.md` if CLI help changes, and `agent-context/research/stage3-local-llm-findings.md`.

   Requirements:

   - Document decomposed local Ollama behavior.
   - Document the one-to-one speaker-count policy.
   - Document `llm_usage` diagnostics at a high level.
   - Keep public docs current-state only after implementation.

## Execution Strategy

Implement in this order:

1. Validation and prompt-budget primitives.
2. Prompt builders.
3. Decomposed Ollama orchestration.
4. Service wiring.
5. Unit tests.
6. Local smoke benchmark.
7. Documentation.

This order keeps the control points for prompt size and speaker-count behavior in place before any multi-call orchestration is added.

## Acceptance Criteria

- `chronicle identify <session_id> --mode llm` no longer sends one monolithic speaker-map prompt.
- `llm` and `manual` enforce the one-to-one speaker-count policy before any Ollama call.
- Count mismatch failures tell the user to inspect `align-only` or rerun Stage 2 with better speaker-count constraints.
- Local Ollama calls are bounded per unresolved speaker.
- At most one repair pass runs, and only after initial per-speaker calls fail validation.
- Final artifacts contain valid canonical speaker assignments and preserve transcript text.
- `llm_usage` includes per-call counts, timing, token information, statuses, attempt numbers, and hashes.
- `llm_usage` does not include prompt text, response text, transcript excerpts, or context excerpts.
- Missing model/configuration, count-policy violations, prompt-budget violations, invalid assignments, and failed repair raise explicit `StageExecutionError` messages.
- Existing `manual` and `align-only` behavior remains intact.
- Unit tests pass without a real Ollama server.
- Local smoke validation shows the decomposed path respects the default 180-second per-call and 600-second three-speaker total budgets on the target machine, or clearly reports failure with enough diagnostics to tune prompt budgets.

## Risks / Edge Cases

- Large participant lists may exceed the prompt budget. The first implementation should fail clearly rather than silently truncating canonical candidates.
- Sequential per-speaker calls may still be slow. Usage diagnostics and the smoke benchmark are required to make that visible.
- Repair logic can become iterative if not bounded. The implementation must enforce exactly one repair pass.
- Storing raw prompt content would duplicate sensitive transcript/context material. Store hashes and summaries only.
- A local benchmark result is hardware-sensitive. Record measured behavior rather than treating one machine's timing as universal.
- If Stage 2 produces the wrong number of anonymous speakers, Stage 3 should fail before identity assignment rather than forcing an invalid map.

## Open Questions

None.

## Implementation Readiness

Ready for implementation with `implement-manage`.

The plan is bounded to Stage 3 local Ollama decomposition, preserves stage boundaries and CLI behavior, resolves the speaker-count policy explicitly, and includes implementation tasks, tests, benchmark validation, and documentation updates.
