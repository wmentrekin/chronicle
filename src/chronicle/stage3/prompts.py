"""Versioned OpenAI prompt construction for Stage 3."""

from __future__ import annotations

import json
from typing import Any

from ..session import SessionManifest
from .schemas import PROMPT_VERSION


SYSTEM_PROMPT = """You map anonymous diarized speaker labels to known session participants.
Return only valid JSON. Do not rewrite transcript text. Do not invent people.
Valid assigned_person and candidate_people values must come only from candidate_people.
people_likely_discussed is context only and is not a valid assignment target unless that name is also in candidate_people.
Use confidence conservatively: Confirmed, Likely, Unclear, or Needs review."""


def build_speaker_map_prompt(
    *,
    manifest: SessionManifest,
    context_text: str,
    participants_by_name: dict[str, dict[str, Any]],
    evidence_summary: dict[str, Any],
    manual_entries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    candidate_people = [
        {
            "canonical_name": name,
            "metadata": participants_by_name.get(name, {}),
        }
        for name in manifest.participants
    ]
    payload = {
        "prompt_version": PROMPT_VERSION,
        "session": {
            "session_id": manifest.session_id,
            "title": manifest.title,
            "interview_date": manifest.interview_date,
            "language": manifest.language,
            "primary_interviewees": manifest.primary_interviewees,
        },
        "candidate_people": candidate_people,
        "people_likely_discussed": manifest.people_likely_discussed,
        "context_excerpt": context_text[:6000],
        "manual_assignments": manual_entries,
        "anonymous_speakers": [
            {"speaker_label": label, **summary}
            for label, summary in sorted(evidence_summary.items())
            if label not in {entry["speaker_label"] for entry in manual_entries}
        ],
        "rules": [
            "Assign every remaining anonymous speaker to exactly one candidate person.",
            "Do not assign anyone outside candidate_people.",
            "Do not duplicate assigned_person values.",
            "Use Needs review only if evidence is too weak, but still keep candidate_people manifest-only.",
            "Return JSON with one top-level key: speaker_map.",
        ],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True, indent=2)},
    ]


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    # Conservative rough estimate; exact accounting is recorded from the provider when available.
    return max(1, sum(len(message.get("content", "")) for message in messages) // 4)
