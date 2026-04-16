"""Speaker scoring and assignment for Stage 3."""

from __future__ import annotations

from typing import Any, Optional

from ..session import SessionManifest
from ..utils import normalize_text
from .inputs import build_interviewee_context_clues, build_participant_aliases, text_contains_alias
from .segmentation import tokenize_for_matching


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


def assign_stage3_blocks(
    manifest: SessionManifest,
    candidate_blocks: list[dict[str, Any]],
    participants_by_name: dict[str, dict[str, Any]],
    context_text: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    clue_tokens_by_name, aliases_by_name = build_interviewee_context_clues(
        manifest, participants_by_name, context_text, tokenize_for_matching
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
        "Stage 3 currently uses local semantic heuristics over the Stage 1 transcript and session context.",
        "This is a transitional implementation until a real anonymous audio-diarization Stage 2 exists.",
        "Speaker labels are conservative and preserve uncertainty rather than forcing attribution.",
    ]
    return assigned_blocks, stage_notes


def reconcile_stage3_question_targets(
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
