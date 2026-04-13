"""Input loading and metadata helpers for Stage 2."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..exceptions import StageExecutionError
from ..paths import repo_relative
from ..session import SessionManifest, resolve_audio_path, resolve_context_path
from ..stage1.artifacts import legacy_stage1_output_paths, session_stage1_output_paths
from ..utils import load_yaml


def load_participant_records(participants_file: Path) -> dict[str, dict[str, Any]]:
    payload = load_yaml(participants_file)
    participants = payload.get("participants")
    if not isinstance(participants, list):
        raise StageExecutionError(
            f"`participants` list missing or invalid in {repo_relative(participants_file)}"
        )

    records: dict[str, dict[str, Any]] = {}
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        canonical_name = participant.get("canonical_name")
        if isinstance(canonical_name, str) and canonical_name.strip():
            records[canonical_name.strip()] = participant
    return records


def cleaned_name_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9 ]+", " ", value).strip()


def build_participant_aliases(participant: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    canonical_name = participant.get("canonical_name")
    short_name = participant.get("short_name")

    for raw_value in (canonical_name, short_name):
        if not isinstance(raw_value, str):
            continue
        cleaned = cleaned_name_token(raw_value)
        if cleaned:
            aliases.add(cleaned.lower())

    if isinstance(canonical_name, str) and canonical_name.strip():
        first_name_parts = cleaned_name_token(canonical_name).split()
        if first_name_parts:
            aliases.add(first_name_parts[0].lower())
    return {alias for alias in aliases if alias}


def text_contains_alias(text: str, alias: str) -> bool:
    if not alias:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(part) for part in alias.split()) + r"\b"
    return re.search(pattern, text) is not None


def parse_markdown_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).strip()] = text[start:end].strip()
    return sections


def split_sentences(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", collapsed)
    return [part.strip() for part in parts if part.strip()]


def build_interviewee_context_clues(
    manifest: SessionManifest,
    participants_by_name: dict[str, dict[str, Any]],
    context_text: str,
    tokenize_for_matching: Any,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    sections = parse_markdown_sections(context_text)
    background_text = sections.get("Background", context_text)
    sentences = split_sentences(background_text)

    aliases_by_name = {
        name: build_participant_aliases(participants_by_name.get(name, {"canonical_name": name}))
        for name in manifest.primary_interviewees
    }
    clue_tokens: dict[str, set[str]] = {name: set() for name in manifest.primary_interviewees}

    for index, sentence in enumerate(sentences):
        lower_sentence = sentence.lower()
        for name in manifest.primary_interviewees:
            aliases = aliases_by_name.get(name, set())
            if not any(text_contains_alias(lower_sentence, alias) for alias in aliases):
                continue

            clue_tokens[name].update(tokenize_for_matching(sentence))
            if index + 1 >= len(sentences):
                continue

            next_sentence = sentences[index + 1]
            next_lower = next_sentence.lower()
            mentions_other_interviewee = any(
                any(text_contains_alias(next_lower, alias) for alias in aliases_by_name.get(other_name, set()))
                for other_name in manifest.primary_interviewees
                if other_name != name
            )
            if not mentions_other_interviewee:
                clue_tokens[name].update(tokenize_for_matching(next_sentence))

    return clue_tokens, aliases_by_name


def load_stage1_segments(
    manifest: SessionManifest,
    stage1_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    session_json_path, _ = session_stage1_output_paths(stage1_dir)
    if session_json_path.exists():
        payload = json.loads(session_json_path.read_text(encoding="utf-8"))
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise StageExecutionError(f"Invalid Stage 1 artifact format: {repo_relative(session_json_path)}")

        combined_segments: list[dict[str, Any]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            combined_segments.append(
                {
                    "source_audio": str(segment.get("source_audio") or ""),
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": str(segment.get("text", "")).strip(),
                    "source_segment_id": segment.get("source_segment_id", segment.get("segment_id")),
                    "source_stage1_artifact": repo_relative(session_json_path),
                }
            )

        if not combined_segments:
            raise StageExecutionError("Stage 1 session artifact contained no transcript segments to diarize.")

        return combined_segments, [repo_relative(session_json_path)]

    combined_segments: list[dict[str, Any]] = []
    artifact_paths: list[str] = []

    for audio_file in manifest.audio_files:
        json_path, _ = legacy_stage1_output_paths(stage1_dir, audio_file)
        if not json_path.exists():
            raise StageExecutionError(
                "Stage 2 requires existing Stage 1 artifacts. Missing: "
                f"{repo_relative(json_path)}. Run `chronicle transcribe {manifest.session_id}` first."
            )

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise StageExecutionError(f"Invalid Stage 1 artifact format: {repo_relative(json_path)}")

        artifact_paths.append(repo_relative(json_path))
        audio_label = repo_relative(resolve_audio_path(manifest, audio_file))
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            combined_segments.append(
                {
                    "source_audio": audio_label,
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": str(segment.get("text", "")).strip(),
                    "source_segment_id": segment.get("segment_id"),
                    "source_stage1_artifact": repo_relative(json_path),
                }
            )

    if not combined_segments:
        raise StageExecutionError("Stage 1 artifacts contained no transcript segments to diarize.")

    return combined_segments, artifact_paths


def load_stage2_context_text(manifest: SessionManifest) -> str:
    return resolve_context_path(manifest).read_text(encoding="utf-8")
