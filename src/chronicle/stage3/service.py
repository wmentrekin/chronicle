"""Stage 3 speaker identification orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import StageExecutionError
from ..paths import repo_relative
from ..session import SessionManifest
from .alignment import align_transcript_to_diarization
from .artifacts import output_paths_for_mode, write_stage3_artifacts
from .evidence import build_evidence_summary
from .identity import apply_speaker_map_to_blocks, normalize_llm_entries, validate_speaker_count
from .inputs import load_stage3_inputs
from .llm import (
    require_ollama_config,
    resolve_stage3_model,
    run_ollama_decomposed_backend,
)
from .manual import load_manual_speaker_map, validate_manual_speaker_map
from .refmatch import run_speechbrain_hybrid, run_speechbrain_refmatch
from .schemas import (
    AUTOMATIC_MODES,
    MODES,
    SCHEMA_VERSION,
    backend_uses_llm,
    build_ollama_backend_usage,
    empty_llm_usage,
    resolve_stage3_backend,
)


def _artifact_metadata(
    *,
    manifest: SessionManifest,
    mode: str,
    resolved_backend: str | None,
    inputs: Any,
    alignment_summary: dict[str, Any],
    evidence_summary: dict[str, Any],
    blocks: list[dict[str, Any]],
    speaker_map: list[dict[str, Any]],
    notes: list[str],
    backend_usage: dict[str, Any] | None,
    llm_usage: dict[str, Any] | None,
) -> dict[str, Any]:
    artifact = {
        "stage": "stage3_speaker_identification",
        "schema_version": SCHEMA_VERSION,
        "session_id": manifest.session_id,
        "mode": mode,
        "backend": resolved_backend,
        "source_stage1_artifact": repo_relative(inputs.stage1_path),
        "source_stage2_artifact": repo_relative(inputs.stage2_path),
        "participants_file": repo_relative(inputs.participants_file),
        "context_doc": repo_relative(inputs.context_path),
        "speaker_map": speaker_map,
        "evidence_summary": evidence_summary,
        "alignment_summary": alignment_summary,
        "blocks": blocks,
        "notes": notes,
    }
    if backend_usage is not None:
        artifact["backend_usage"] = backend_usage
    if llm_usage is not None:
        artifact["llm_usage"] = llm_usage
    return artifact


def execute_stage3(
    *,
    manifest: SessionManifest,
    stage1_dir: Path,
    stage2_dir: Path,
    stage3_dir: Path,
    participants_file: Path,
    force: bool,
    mode: str = "llm",
    model: str | None = None,
    backend: str | None = None,
    speaker_map_path: Path | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    if mode not in MODES:
        raise StageExecutionError(f"Invalid Stage 3 mode `{mode}`. Expected one of: {', '.join(sorted(MODES))}")

    resolved_backend = resolve_stage3_backend(backend) if mode in AUTOMATIC_MODES else None
    resolved_model = (
        resolve_stage3_model(model)
        if resolved_backend in {"ollama_decomposed", "speechbrain_hybrid"}
        else None
    )
    json_path, markdown_path = output_paths_for_mode(stage3_dir, mode)
    if not force and json_path.exists() and markdown_path.exists():
        return [], [repo_relative(json_path), repo_relative(markdown_path)], [
            "Stage 3 artifacts already exist; rerun with `--force` to overwrite them."
        ], {"mode": mode, "backend": resolved_backend}

    if resolved_backend and backend_uses_llm(resolved_backend):
        require_ollama_config(resolved_model)
    elif resolved_backend not in {None, "speechbrain_refmatch", "speechbrain_hybrid"}:
        raise StageExecutionError(
            f"Stage 3 backend `{resolved_backend}` is not implemented yet. "
            "Use `ollama_decomposed`, `--mode manual`, or `--mode align-only`."
        )

    inputs = load_stage3_inputs(
        manifest=manifest,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        participants_file=participants_file,
    )
    speaker_labels = [str(label) for label in inputs.stage2_artifact.get("speaker_labels", [])]
    validate_speaker_count(speaker_labels=speaker_labels, participants=manifest.participants, mode=mode)

    aligned_blocks, alignment_summary = align_transcript_to_diarization(
        manifest=manifest,
        stage1_artifact=inputs.stage1_artifact,
        stage2_artifact=inputs.stage2_artifact,
    )
    evidence_summary, evidence_notes = build_evidence_summary(
        manifest=manifest,
        blocks=aligned_blocks,
        speaker_labels=speaker_labels,
    )

    manual_map = load_manual_speaker_map(speaker_map_path)
    manual_entries = validate_manual_speaker_map(
        manual_map=manual_map,
        speaker_labels=speaker_labels,
        participants=manifest.participants,
        mode=mode,
    )

    speaker_map: list[dict[str, Any]] = []
    llm_usage: dict[str, Any] | None = None
    backend_usage: dict[str, Any] | None = None
    notes = list(evidence_notes)
    blocks = aligned_blocks

    if mode == "align-only":
        notes.append("Align-only mode did not assign real people to anonymous speakers.")
    elif mode == "manual":
        speaker_map = manual_entries
        blocks = apply_speaker_map_to_blocks(blocks=aligned_blocks, speaker_map=speaker_map)
    else:
        if resolved_backend == "speechbrain_refmatch":
            speaker_map, backend_usage = run_speechbrain_refmatch(
                manifest=manifest,
                inputs=inputs,
                speaker_labels=speaker_labels,
                participants=manifest.participants,
                manual_entries=manual_entries,
                cache_root=stage3_dir / "embeddings",
            )
        elif resolved_backend == "speechbrain_hybrid":
            speaker_map, backend_usage, llm_usage = run_speechbrain_hybrid(
                manifest=manifest,
                inputs=inputs,
                speaker_labels=speaker_labels,
                participants=manifest.participants,
                manual_entries=manual_entries,
                evidence_summary=evidence_summary,
                cache_root=stage3_dir / "embeddings",
                llm_model=resolved_model or resolve_stage3_model(None),
            )
        else:
            raw_llm_entries, backend_usage, llm_usage = run_ollama_decomposed_backend(
                manifest=manifest,
                context_text=inputs.context_text,
                participants_by_name=inputs.participants_by_name,
                evidence_summary=evidence_summary,
                manual_entries=manual_entries,
                model=resolved_model,
            )
            speaker_map = normalize_llm_entries(
                raw_entries=raw_llm_entries,
                speaker_labels=speaker_labels,
                participants=manifest.participants,
                manual_entries=manual_entries,
            )
        blocks = apply_speaker_map_to_blocks(blocks=aligned_blocks, speaker_map=speaker_map)

    if resolved_backend == "ollama_decomposed" and llm_usage is None:
        llm_usage = empty_llm_usage(resolved_model, ["LLM usage unavailable."])
        backend_usage = build_ollama_backend_usage(
            model=resolved_model,
            llm_usage=llm_usage,
            manual_entries=manual_entries,
            evidence_summary={},
            llm_entry_count=0,
        )

    artifact = _artifact_metadata(
        manifest=manifest,
        mode=mode,
        resolved_backend=resolved_backend,
        inputs=inputs,
        alignment_summary=alignment_summary,
        evidence_summary=evidence_summary,
        blocks=blocks,
        speaker_map=speaker_map,
        notes=notes,
        backend_usage=backend_usage,
        llm_usage=llm_usage,
    )

    output_paths = write_stage3_artifacts(stage_dir=stage3_dir, mode=mode, artifact=artifact)
    metadata = {
        "mode": mode,
        "backend": resolved_backend,
        "model": (llm_usage or {}).get("model") or (resolved_model if resolved_backend == "ollama_decomposed" else None),
        "provider": (llm_usage or {}).get("provider") or ("ollama" if resolved_backend == "ollama_decomposed" else None),
        "prompt_version": llm_usage["prompt_version"] if llm_usage else None,
        "schema_version": SCHEMA_VERSION,
        "source_stage1_artifact": repo_relative(inputs.stage1_path),
        "source_stage2_artifact": repo_relative(inputs.stage2_path),
        "participants_file": repo_relative(inputs.participants_file),
        "context_doc": repo_relative(inputs.context_path),
        "speaker_map_path": repo_relative(speaker_map_path) if speaker_map_path else None,
        "backend_usage": backend_usage,
        "llm_usage": llm_usage,
    }
    return [repo_relative(path) for path in output_paths], [], notes, metadata
