# Stage 3 Architecture

Stage 3 identifies real speakers by reconciling two prior artifacts:

- Stage 1 provides the verbatim transcript and transcript timestamps.
- Stage 2 provides anonymous diarized speaker turns such as `SPEAKER_00`.

Stage 3 does not transcribe audio, redo diarization, or polish the transcript. It preserves Stage 1 wording and Stage 2 anonymous speaker provenance, then maps anonymous labels to canonical manifest participants when justified.

## Command

Default LLM-backed identification:

```bash
chronicle identify <session_id>
```

Explicit modes:

```bash
chronicle identify <session_id> --mode llm
chronicle identify <session_id> --mode manual --speaker-map inputs/sessions/<session_id>/speaker-map.yaml
chronicle identify <session_id> --mode align-only
```

Model precedence in `llm` mode:

1. `--model <model-name>`
2. `CHRONICLE_STAGE3_MODEL`
3. built-in default `gpt-5.4-mini`

`llm` mode requires `OPENAI_API_KEY`. If the key is missing, Chronicle fails before writing final identity artifacts and prints setup instructions. `align-only` and complete `manual` mode do not call OpenAI.

## Inputs

Required:

- `outputs/<session_id>/stage1/raw_transcript.json`
- `outputs/<session_id>/stage2/diarization.json`
- `inputs/global/participants.yaml`
- `inputs/sessions/<session_id>/session.yaml`
- `inputs/sessions/<session_id>/context.md`

Optional:

- `inputs/sessions/<session_id>/speaker-map.yaml`

Manual speaker maps use canonical manifest participants only:

```yaml
speaker_map:
  SPEAKER_00: Canonical Participant A
  SPEAKER_01: Canonical Participant B
```

Unknown or arbitrary speaker names are not valid in the first Stage 3 contract. If the diarized speaker count does not match `manifest.participants`, identity modes fail with guidance to inspect `--mode align-only` or rerun Stage 2 with tighter speaker-count constraints.

## Outputs

Final identity modes write:

- `outputs/<session_id>/stage3/identified_conversation.json`
- `outputs/<session_id>/stage3/identified_conversation.md`

Local alignment mode writes:

- `outputs/<session_id>/stage3/aligned_transcript.json`
- `outputs/<session_id>/stage3/aligned_transcript.md`

The separate align-only artifact names are intentional. Anonymous alignment is useful for review, but it is not completed speaker identification.

## Artifact Contract

Stage 3 JSON artifacts use schema version `1.0` and include:

- `stage`
- `schema_version`
- `session_id`
- `mode`
- `source_stage1_artifact`
- `source_stage2_artifact`
- `participants_file`
- `context_doc`
- `speaker_map`
- `evidence_summary`
- `alignment_summary`
- `blocks`
- `llm_usage`
- `notes`

Markdown is rendered from JSON and is not the machine source of truth.

Each block preserves:

- anonymous `speaker_label` or `speaker_label_candidates`
- assigned canonical `speaker` when available
- confidence
- source audio path
- Stage 1 segment references
- Stage 2 turn references
- alignment confidence and overlap metrics
- verbatim Stage 1 text

## Internal Flow

1. Load Stage 1, Stage 2, session, participant, and context inputs.
2. Normalize source-audio paths so multi-file sessions do not cross-align audio.
3. Align Stage 1 segments to Stage 2 turns by source file and timestamp overlap.
4. Mark boundary-crossing or weak-overlap segments as `Needs review` instead of forcing attribution.
5. Build deterministic per-speaker evidence summaries.
6. Validate manual speaker-map overrides if provided.
7. In `llm` mode, send compact evidence, context excerpts, and participant metadata to OpenAI for speaker-label mapping.
8. Validate all assignments against `manifest.participants`.
9. Apply the speaker map deterministically to aligned blocks.
10. Write JSON, Markdown, and run metadata.

Raw audio is never uploaded to OpenAI.

## Module Layout

- `service.py`: orchestration
- `inputs.py`: Stage 1/Stage 2/context/participant loading and source normalization
- `alignment.py`: source-aware timestamp alignment and ambiguity handling
- `evidence.py`: deterministic speaker evidence summaries
- `manual.py`: manual speaker-map loading and validation
- `llm.py`: OpenAI configuration, model precedence, request execution, and usage metadata
- `prompts.py`: versioned prompt construction
- `identity.py`: speaker-map validation and block-level identity application
- `artifacts.py`: output paths and Markdown rendering
- `schemas.py`: schema constants, defaults, confidence labels, and thresholds

This split is intentional. Stage 3 should remain inspectable and testable rather than accumulating all behavior in one large service file.
