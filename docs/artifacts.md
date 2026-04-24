# Artifact Contracts

Status: current implementation

This document summarizes the public file contracts Chronicle writes today. Markdown artifacts are human-readable companions; JSON artifacts are the machine source of truth.

## Session input manifest

Path:

```text
inputs/sessions/<session_id>/session.yaml
```

Current fields:

- `session_id`
- `title`
- `interview_date`
- `audio_files`
- `participants`
- `primary_interviewees`
- `people_likely_discussed`
- `context_doc`
- `language`
- `source_format`
- `notes`
- `tags`
- `stage1_model_preference`

Validation requires canonical names in `participants`, `primary_interviewees`, and `people_likely_discussed` to exist in `inputs/global/participants.yaml`.

## Stage 1

Paths:

```text
outputs/<session_id>/stage1/raw_transcript.json
outputs/<session_id>/stage1/raw_transcript.md
```

Current JSON fields:

- `stage`
- `session_id`
- `source_audio_files`
- `audio_files`
- `normalized_audio`
- `model`
- `language`
- `transcript_text`
- `segments`
- `word_timestamps`
- `notes`

Segment entries preserve source audio provenance and session-relative timing.

## Stage 2

Paths:

```text
outputs/<session_id>/stage2/diarization.json
outputs/<session_id>/stage2/diarization.md
```

The current Stage 2 contract uses `diarization.json` and `diarization.md`, not `diarized_conversation.*`.

Current JSON fields:

- `stage`
- `status`
- `backend`
- `session_id`
- `audio_files`
- `speaker_labels`
- `turns`
- `constraints`
- `models`
- `runtime`
- `notes`

Current turn entries:

- `turn_id`
- `speaker_label`
- `source_audio`
- `source_start_seconds`
- `source_end_seconds`
- `session_start_seconds`
- `session_end_seconds`

## Stage 3

Paths:

```text
outputs/<session_id>/stage3/identified_conversation.json
outputs/<session_id>/stage3/identified_conversation.md
outputs/<session_id>/stage3/aligned_transcript.json
outputs/<session_id>/stage3/aligned_transcript.md
```

The output path depends on mode:

- `llm` and `manual` write `identified_conversation.*`
- `align-only` writes `aligned_transcript.*`

Current JSON fields:

- `stage`
- `schema_version`
- `session_id`
- `mode`
- `backend`
- `source_stage1_artifact`
- `source_stage2_artifact`
- `participants_file`
- `context_doc`
- `speaker_map`
- `evidence_summary`
- `alignment_summary`
- `blocks`
- `backend_usage`
- `llm_usage`
- `notes`

Field notes:

- `backend` is set for automatic Stage 3 runs and records the selected backend id.
- `backend_usage` is backend-specific metadata for automatic runs. Current implementations use it for items such as provider/model details, workflow/reference mode, assignment counts, token summaries, or enrollment coverage.
- `llm_usage` is present only for backends that invoke local Ollama.

Current speaker-map entries:

- `speaker_label`
- `assigned_person`
- `confidence`
- `candidate_people`
- `source`
- `evidence`
- `notes`

Current block entries:

- `block_id`
- `speaker_label`
- `speaker_label_candidates`
- `speaker`
- `confidence`
- `candidate_people`
- `start_time`
- `end_time`
- `source_audio`
- `text`
- `source_turn_ids`
- `source_segment_ids`
- `source_stage1_segment_ids`
- `alignment`
- `notes`

## Stage 3 benchmark reports

Paths:

```text
outputs/<session_id>/runs/stage3-benchmark.<timestamp>.json
outputs/<session_id>/runs/stage3-benchmark.<timestamp>.md
outputs/<session_id>/runs/stage3-benchmark.<timestamp>/<backend>/stage3/...
```

Current benchmark JSON fields:

- `stage`
- `session_id`
- `truth_file`
- `participants_file`
- `backends`
- `cpu_feasibility_notes`
- `benchmark_root`
- `results`
- `recommendation`

Current per-backend result fields:

- `backend`
- `status`
- `runtime_seconds`
- `step_runtimes`
- `output_paths`
- `notes`
- `backend_usage`
- `llm_usage`
- `enrollment_coverage`
- `cpu_feasibility_notes`
- `correct_assignments`
- `total_assignments`
- `exact_assignment_accuracy`
- `mismatches`
- `error`

Current recommendation fields:

- `recommended_backend`
- `basis`
- `compared_backends`
- `tie_threshold_percentage_points`
- `winning_accuracy`
- `winning_runtime_seconds`

## Stage 4

Stage 4 does not yet have a stable public artifact contract. The current `organize` command only validates the session and reserves the Stage 4 output directory.

## Notes on markdown companions

- Markdown files are derived views for review.
- JSON files remain the authoritative artifacts for downstream stages.
- Stage boundaries should be preserved: Stage 1 writes transcription, Stage 2 writes anonymous turns, Stage 3 writes identification, and Stage 4 is still scaffold-only.
