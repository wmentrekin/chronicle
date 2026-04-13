"""Stage 2 semantic diarization logic."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from ..exceptions import StageExecutionError
from ..paths import repo_relative
from ..session import SessionManifest, resolve_audio_path, resolve_context_path
from ..stage1.service import legacy_stage1_output_paths, session_stage1_output_paths
from ..utils import load_yaml, normalize_text, write_json


STAGE2_QUESTION_PATTERNS = (
    r"^okay\b.*\brecording\b",
    r"^i guess i want to start\b",
    r"\bdo you\b",
    r"\bdid you\b",
    r"\bwere you\b",
    r"\bwas it\b",
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bwould you\b",
    r"\bwhat was it like\b",
    r"\bwhat do you\b",
    r"\bhow did\b",
    r"\bwhere did\b",
    r"\bwhen did\b",
    r"\bwhy did\b",
    r"\bwho was\b",
    r"\bdo you remember\b",
    r"\byou have any memories\b",
)
STAGE2_QUESTION_CONTINUATION_PREFIXES = (
    "but obviously",
    "but like",
    "and what",
    "and where",
    "and when",
    "and how",
    "and why",
    "and who",
    "before cindy",
)
STAGE2_REPORTED_SPEECH_MARKERS = (
    r"\bi said\b",
    r"\bshe said\b",
    r"\bhe said\b",
    r"\bmother said\b",
    r"\bdaddy said\b",
)
STAGE2_SHORT_ACKS = {
    "yeah",
    "yes",
    "yep",
    "right",
    "okay",
    "ok",
    "uh-huh",
    "mm-hmm",
    "sure",
    "no",
}
STAGE2_STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "before",
    "being",
    "between",
    "but",
    "by",
    "could",
    "did",
    "do",
    "does",
    "down",
    "during",
    "each",
    "effects",
    "family",
    "for",
    "from",
    "get",
    "going",
    "grew",
    "growing",
    "had",
    "has",
    "have",
    "hear",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "know",
    "like",
    "little",
    "made",
    "make",
    "many",
    "me",
    "more",
    "most",
    "much",
    "my",
    "never",
    "now",
    "of",
    "off",
    "on",
    "once",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "people",
    "really",
    "said",
    "say",
    "see",
    "she",
    "so",
    "some",
    "something",
    "still",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "think",
    "this",
    "those",
    "through",
    "time",
    "to",
    "too",
    "up",
    "us",
    "very",
    "was",
    "we",
    "well",
    "went",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "yall",
    "yeah",
    "you",
    "your",
}


def stage2_output_paths(stage_dir: Path) -> tuple[Path, Path]:
    return (
        stage_dir / "diarized_conversation.json",
        stage_dir / "diarized_conversation.md",
    )


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


def normalize_match_token(token: str) -> str:
    normalized = token.lower().replace("'", "")
    if len(normalized) > 5 and normalized.endswith("ies"):
        return normalized[:-3] + "y"
    for suffix in ("ing", "ied", "ers", "ed", "es", "s"):
        if len(normalized) > 4 and normalized.endswith(suffix):
            trimmed = normalized[: -len(suffix)]
            if trimmed:
                return trimmed
    return normalized


def tokenize_for_matching(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z']+", text)
    normalized_tokens = {
        normalize_match_token(token)
        for token in tokens
        if len(token) >= 3
    }
    return {token for token in normalized_tokens if token and token not in STAGE2_STOPWORDS}


def split_sentences(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", collapsed)
    return [part.strip() for part in parts if part.strip()]


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


def build_interviewee_context_clues(
    manifest: SessionManifest,
    participants_by_name: dict[str, dict[str, Any]],
    context_text: str,
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


def classify_stage2_segment_type(text: str) -> str:
    normalized = normalize_text(text).lower()
    if not normalized:
        return "response"

    if any(normalized.startswith(prefix) for prefix in STAGE2_QUESTION_CONTINUATION_PREFIXES):
        return "question"

    words = normalized.split()
    if words and words[0] in {"what", "when", "where", "why", "who", "how"}:
        return "question"

    if len(words) <= 4 and normalized in STAGE2_SHORT_ACKS:
        return "ack"

    if any(re.search(pattern, normalized) for pattern in STAGE2_REPORTED_SPEECH_MARKERS):
        return "response"

    if any(re.search(pattern, normalized) for pattern in STAGE2_QUESTION_PATTERNS):
        return "question"

    return "response"


def resolve_stage2_segment_types(segments: list[dict[str, Any]]) -> list[str]:
    raw_types = [classify_stage2_segment_type(str(segment.get("text", ""))) for segment in segments]
    resolved = list(raw_types)
    for index, segment_type in enumerate(raw_types):
        if segment_type != "ack":
            continue

        previous_type = next(
            (raw_types[cursor] for cursor in range(index - 1, -1, -1) if raw_types[cursor] != "ack"),
            None,
        )
        next_type = next(
            (raw_types[cursor] for cursor in range(index + 1, len(raw_types)) if raw_types[cursor] != "ack"),
            None,
        )
        resolved[index] = previous_type or next_type or "response"

    for index, segment in enumerate(segments):
        if resolved[index] != "response":
            continue
        previous_type = next(
            (resolved[cursor] for cursor in range(index - 1, -1, -1) if resolved[cursor] != "ack"),
            None,
        )
        next_type = next(
            (resolved[cursor] for cursor in range(index + 1, len(resolved)) if resolved[cursor] != "ack"),
            None,
        )
        normalized = normalize_text(str(segment.get("text", ""))).lower()
        words = normalized.split()
        is_question_continuation = (
            previous_type == "question"
            and (
                next_type == "question"
                or (
                    len(words) <= 14
                    and any(normalized.startswith(prefix) for prefix in ("but", "and", "so", "yeah"))
                    and ("you" in words or "your" in words or "yall" in normalized or "you're" in normalized)
                )
            )
        )
        is_interviewer_interjection = (
            previous_type == "response"
            and next_type == "response"
            and len(words) <= 12
            and any(normalized.startswith(prefix) for prefix in ("so", "yeah", "okay", "right"))
            and not any(re.search(pattern, normalized) for pattern in STAGE2_REPORTED_SPEECH_MARKERS)
        )
        if is_question_continuation or is_interviewer_interjection:
            resolved[index] = "question"
    return resolved


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


def build_stage2_candidate_blocks(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segment_types = resolve_stage2_segment_types(segments)
    blocks: list[dict[str, Any]] = []

    for segment, segment_type in zip(segments, segment_types):
        text = normalize_text(str(segment.get("text", "")))
        if not text:
            continue

        if not blocks or blocks[-1]["block_type"] != segment_type or blocks[-1]["source_audio"] != segment["source_audio"]:
            blocks.append(
                {
                    "block_type": segment_type,
                    "source_audio": segment["source_audio"],
                    "start_time": segment.get("start"),
                    "end_time": segment.get("end"),
                    "text_parts": [text],
                    "source_segment_ids": [segment.get("source_segment_id")],
                    "source_stage1_artifact": segment.get("source_stage1_artifact"),
                }
            )
            continue

        blocks[-1]["end_time"] = segment.get("end")
        blocks[-1]["text_parts"].append(text)
        blocks[-1]["source_segment_ids"].append(segment.get("source_segment_id"))

    for block in blocks:
        block["text"] = normalize_text(" ".join(block.pop("text_parts")))
    return blocks


def score_question_targets(
    text: str,
    primary_interviewees: list[str],
    clue_tokens_by_name: dict[str, set[str]],
    aliases_by_name: dict[str, set[str]],
) -> dict[str, dict[str, Any]]:
    lower_text = text.lower()
    tokens = tokenize_for_matching(text)
    scores: dict[str, dict[str, Any]] = {}
    for name in primary_interviewees:
        alias_hits = sorted(
            alias for alias in aliases_by_name.get(name, set()) if text_contains_alias(lower_text, alias)
        )
        keyword_hits = sorted(tokens & clue_tokens_by_name.get(name, set()))
        scores[name] = {
            "score": len(alias_hits) * 4 + len(keyword_hits),
            "alias_hits": alias_hits,
            "keyword_hits": keyword_hits,
        }
    return scores


def score_response_candidates(
    text: str,
    primary_interviewees: list[str],
    clue_tokens_by_name: dict[str, set[str]],
) -> dict[str, dict[str, Any]]:
    tokens = tokenize_for_matching(text)
    scores: dict[str, dict[str, Any]] = {}
    for name in primary_interviewees:
        keyword_hits = sorted(tokens & clue_tokens_by_name.get(name, set()))
        scores[name] = {
            "score": len(keyword_hits),
            "keyword_hits": keyword_hits,
        }
    return scores


def choose_best_names(scores: dict[str, dict[str, Any]]) -> list[str]:
    if not scores:
        return []
    top_score = max((entry.get("score", 0) for entry in scores.values()), default=0)
    if top_score <= 0:
        return []
    return sorted(name for name, entry in scores.items() if entry.get("score", 0) == top_score)


def build_clue_note(name: str, clue_entry: dict[str, Any]) -> Optional[str]:
    keyword_hits = clue_entry.get("keyword_hits") or []
    alias_hits = clue_entry.get("alias_hits") or []
    if not alias_hits and not keyword_hits:
        return None

    details: list[str] = []
    if alias_hits:
        details.append("direct name cue")
    if keyword_hits:
        details.append(f"{len(keyword_hits)} session-context clue(s)")
    return f"Matched {name} via " + " and ".join(details) + "."


def detect_metadata_mentions(
    text: str,
    names: list[str],
    participants_by_name: dict[str, dict[str, Any]],
) -> list[str]:
    lower_text = text.lower()
    mentions: list[str] = []
    for name in names:
        participant = participants_by_name.get(name)
        if participant is None:
            continue
        aliases = build_participant_aliases(participant)
        if any(text_contains_alias(lower_text, alias) for alias in aliases):
            mentions.append(name)
    return sorted(set(mentions))


def assign_stage2_blocks(
    manifest: SessionManifest,
    candidate_blocks: list[dict[str, Any]],
    participants_by_name: dict[str, dict[str, Any]],
    context_text: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    clue_tokens_by_name, aliases_by_name = build_interviewee_context_clues(
        manifest, participants_by_name, context_text
    )

    interviewer_candidates = [
        name
        for name in manifest.participants
        if participants_by_name.get(name, {}).get("role") == "interviewer"
    ]
    if not interviewer_candidates:
        interviewer_candidates = [
            name for name in manifest.participants if name not in set(manifest.primary_interviewees)
        ]

    interviewer_name = interviewer_candidates[0] if len(interviewer_candidates) == 1 else None
    assigned_blocks: list[dict[str, Any]] = []
    current_target: Optional[str] = None
    current_interviewee: Optional[str] = None

    for index, block in enumerate(candidate_blocks, 1):
        notes: list[str] = []
        confidence = "Needs review"
        speaker = "Unknown speaker"
        candidate_speakers: list[str] = []
        block_text = block["text"]
        targeted_interviewee: Optional[str] = None

        if block["block_type"] == "question":
            if interviewer_name:
                speaker = interviewer_name
                confidence = "Confirmed"
                notes.append("Question phrasing matches the interviewer role.")
            else:
                candidate_speakers = interviewer_candidates or manifest.participants
                notes.append("Question phrasing is clear, but interviewer identity is not unique in session metadata.")

            if len(manifest.primary_interviewees) == 1:
                current_target = manifest.primary_interviewees[0]
                notes.append(f"Only one primary interviewee is listed: {current_target}.")
            else:
                target_scores = score_question_targets(
                    block_text,
                    manifest.primary_interviewees,
                    clue_tokens_by_name,
                    aliases_by_name,
                )
                best_names = choose_best_names(target_scores)
                ranked_scores = sorted(
                    (entry.get("score", 0) for entry in target_scores.values()),
                    reverse=True,
                )
                top_score = ranked_scores[0] if ranked_scores else 0
                second_score = ranked_scores[1] if len(ranked_scores) > 1 else 0
                if len(best_names) == 1 and top_score >= 2 and top_score >= second_score + 2:
                    current_target = best_names[0]
                    targeted_interviewee = current_target
                    clue_note = build_clue_note(current_target, target_scores[current_target])
                    if clue_note:
                        notes.append(f"Likely addressed to {current_target}. {clue_note}")
                elif current_interviewee:
                    current_target = current_interviewee
                    targeted_interviewee = current_target
                    notes.append(
                        f"No unique target clue found; keeping the follow-up question attached to {current_target}."
                    )
                else:
                    current_target = None
                    notes.append(
                        "No unique interviewee target was found in the question; the next response should remain conservative."
                    )

        else:
            if len(manifest.primary_interviewees) == 1:
                speaker = manifest.primary_interviewees[0]
                confidence = "Confirmed"
                notes.append(f"Only one primary interviewee is listed: {speaker}.")
            else:
                response_scores = score_response_candidates(
                    block_text,
                    manifest.primary_interviewees,
                    clue_tokens_by_name,
                )
                best_names = choose_best_names(response_scores)

                if current_target and (
                    not best_names or best_names == [current_target] or current_target in best_names
                ):
                    speaker = current_target
                    confidence = "Likely"
                    notes.append(f"Block follows a question aimed at {current_target}.")
                    clue_entry = response_scores.get(current_target)
                    clue_note = build_clue_note(current_target, clue_entry or {})
                    if clue_note:
                        notes.append(clue_note)
                elif len(best_names) == 1:
                    speaker = best_names[0]
                    confidence = "Likely"
                    clue_note = build_clue_note(best_names[0], response_scores[best_names[0]])
                    if clue_note:
                        notes.append(clue_note)
                elif current_interviewee and not best_names:
                    speaker = current_interviewee
                    confidence = "Unclear"
                    notes.append(
                        f"No new clue shifted the topic, so the speaker was carried forward from the previous interviewee block: {current_interviewee}."
                    )
                else:
                    speaker = "Unknown speaker"
                    candidate_speakers = best_names or list(manifest.primary_interviewees)
                    confidence = "Unclear" if best_names else "Needs review"
                    if best_names:
                        notes.append(
                            "Multiple interviewees matched the local context clues: "
                            + ", ".join(best_names)
                            + "."
                        )
                    else:
                        notes.append(
                            "The response block did not contain enough unique context to assign one interviewee safely."
                        )

            if speaker in manifest.primary_interviewees:
                current_interviewee = speaker

        mentioned_people = detect_metadata_mentions(
            block_text,
            manifest.people_likely_discussed,
            participants_by_name,
        )
        if mentioned_people:
            notes.append("Named-person mentions match session metadata: " + ", ".join(mentioned_people[:3]) + ".")

        assigned_blocks.append(
            {
                "block_id": index,
                "source_audio": block["source_audio"],
                "block_type": block["block_type"],
                "speaker": speaker,
                "confidence": confidence,
                "start_time": block["start_time"],
                "end_time": block["end_time"],
                "text": block_text,
                "notes": notes,
                "candidate_speakers": candidate_speakers,
                "targeted_interviewee": targeted_interviewee,
                "source_segment_ids": block["source_segment_ids"],
                "source_stage1_artifact": block.get("source_stage1_artifact"),
            }
        )

    stage_notes = [
        "Stage 2 uses local semantic heuristics over the Stage 1 transcript; it does not perform audio-based diarization.",
        "Speaker labels are conservative and preserve uncertainty rather than forcing attribution.",
    ]
    return assigned_blocks, stage_notes


def reconcile_stage2_question_targets(
    blocks: list[dict[str, Any]],
    primary_interviewees: list[str],
) -> None:
    primary_names = set(primary_interviewees)
    for index in range(len(blocks) - 1):
        question_block = blocks[index]
        response_block = blocks[index + 1]
        if question_block.get("block_type") != "question" or response_block.get("block_type") != "response":
            continue

        targeted_interviewee = question_block.get("targeted_interviewee")
        response_speaker = response_block.get("speaker")
        if (
            not isinstance(targeted_interviewee, str)
            or response_speaker not in primary_names
            or targeted_interviewee == response_speaker
        ):
            continue

        question_block["candidate_speakers"] = sorted(
            set(question_block.get("candidate_speakers", [])) | {targeted_interviewee, response_speaker}
        )
        question_block["targeted_interviewee"] = None
        question_block["notes"] = [
            note for note in question_block.get("notes", []) if not note.startswith("Likely addressed to ")
        ]
        question_block["notes"].append(
            f"Question target remains uncertain; the following response aligned better with {response_speaker}."
        )


def write_stage2_markdown(
    path: Path,
    manifest: SessionManifest,
    artifact: dict[str, Any],
) -> None:
    lines = [
        "# Diarized Conversation",
        "",
        f"- **Session ID:** {manifest.session_id}",
        f"- **Stage:** {artifact['stage']}",
        "",
        "## Transcript",
        "",
    ]

    current_audio: Optional[str] = None
    for block in artifact["blocks"]:
        if block["source_audio"] != current_audio:
            current_audio = block["source_audio"]
            lines.extend([f"### {current_audio}", ""])

        timestamp = ""
        if block.get("start_time") and block.get("end_time"):
            timestamp = f" [{block['start_time']} - {block['end_time']}]"
        lines.append(f"**{block['speaker']}** `[{block['confidence']}]`{timestamp}")
        lines.append(block["text"] or "[inaudible]")
        lines.append("")
        if block.get("candidate_speakers"):
            lines.append("Candidate speakers: " + ", ".join(block["candidate_speakers"]))
            lines.append("")
        if block.get("notes"):
            lines.append("Notes:")
            for note in block["notes"]:
                lines.append(f"- {note}")
            lines.append("")

    if artifact.get("notes"):
        lines.extend(["## Notes", ""])
        for note in artifact["notes"]:
            lines.append(f"- {note}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def execute_stage2(
    manifest: SessionManifest,
    stage1_dir: Path,
    stage2_dir: Path,
    participants_file: Path,
    force: bool,
) -> tuple[list[str], list[str], list[str]]:
    json_path, markdown_path = stage2_output_paths(stage2_dir)
    if not force and json_path.exists() and markdown_path.exists():
        return [], [repo_relative(json_path), repo_relative(markdown_path)], [
            "Stage 2 artifacts already exist; rerun with `--force` to overwrite them."
        ]

    participants_by_name = load_participant_records(participants_file)
    context_text = resolve_context_path(manifest).read_text(encoding="utf-8")
    stage1_segments, stage1_artifacts = load_stage1_segments(manifest, stage1_dir)
    candidate_blocks = build_stage2_candidate_blocks(stage1_segments)
    assigned_blocks, stage_notes = assign_stage2_blocks(
        manifest=manifest,
        candidate_blocks=candidate_blocks,
        participants_by_name=participants_by_name,
        context_text=context_text,
    )
    reconcile_stage2_question_targets(assigned_blocks, manifest.primary_interviewees)

    artifact_payload = {
        "stage": "stage2_semantic_diarization",
        "session_id": manifest.session_id,
        "participants_file": repo_relative(participants_file),
        "source_stage1_artifacts": stage1_artifacts,
        "block_count": len(assigned_blocks),
        "blocks": assigned_blocks,
        "notes": stage_notes,
    }
    write_json(json_path, artifact_payload)
    write_stage2_markdown(markdown_path, manifest, artifact_payload)
    return [repo_relative(json_path), repo_relative(markdown_path)], [], stage_notes
