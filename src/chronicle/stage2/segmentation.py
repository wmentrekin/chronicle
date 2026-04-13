"""Segmentation and low-level text matching for Stage 2."""

from __future__ import annotations

import re

from ..exceptions import StageExecutionError
from ..utils import normalize_text


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
    "a","about","after","again","all","also","an","and","any","are","as","at","be","because","before","being",
    "between","but","by","could","did","do","does","down","during","each","effects","family","for","from","get",
    "going","grew","growing","had","has","have","hear","her","here","hers","him","his","how","i","if","in","into",
    "is","it","its","just","know","like","little","made","make","many","me","more","most","much","my","never",
    "now","of","off","on","once","one","only","or","other","our","out","over","people","really","said","say",
    "see","she","so","some","something","still","such","than","that","the","their","them","then","there","these",
    "they","thing","think","this","those","through","time","to","too","up","us","very","was","we","well","went",
    "were","what","when","where","which","who","why","with","would","yall","yeah","you","your",
}


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


def resolve_stage2_segment_types(segments: list[dict[str, object]]) -> list[str]:
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


def build_stage2_candidate_blocks(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    segment_types = resolve_stage2_segment_types(segments)
    blocks: list[dict[str, object]] = []

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
