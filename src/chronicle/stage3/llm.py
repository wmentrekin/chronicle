"""OpenAI-backed speaker-map assignment for Stage 3."""

from __future__ import annotations

import json
import os
from typing import Any

from ..exceptions import StageExecutionError
from ..session import SessionManifest
from .prompts import build_speaker_map_prompt, estimate_tokens
from .schemas import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS, empty_llm_usage


def resolve_stage3_model(cli_model: str | None) -> str:
    if cli_model:
        return cli_model
    return os.environ.get("CHRONICLE_STAGE3_MODEL", "gpt-5.4-mini")


def resolve_max_input_tokens() -> int:
    return _env_int("CHRONICLE_STAGE3_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS)


def resolve_max_output_tokens() -> int:
    return _env_int("CHRONICLE_STAGE3_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise StageExecutionError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise StageExecutionError(f"{name} must be greater than zero.")
    return parsed


def require_openai_config(model: str) -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    raise StageExecutionError(
        "Stage 3 `llm` mode requires OPENAI_API_KEY before final identity artifacts are written.\n\n"
        "Setup:\n"
        "1. Add `OPENAI_API_KEY=...` to .env or export it in your shell.\n"
        f"2. Default Stage 3 model is `{model}`.\n"
        "3. Override the model with `CHRONICLE_STAGE3_MODEL=...` or `--model ...`.\n"
        "4. Use `chronicle identify <session_id> --mode align-only` for local anonymous alignment.\n"
        "5. Use `chronicle identify <session_id> --mode manual --speaker-map <path>` for local identity assignment."
    )


def run_openai_speaker_mapping(
    *,
    manifest: SessionManifest,
    context_text: str,
    participants_by_name: dict[str, dict[str, Any]],
    evidence_summary: dict[str, Any],
    manual_entries: list[dict[str, Any]],
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_openai_config(model)
    messages = build_speaker_map_prompt(
        manifest=manifest,
        context_text=context_text,
        participants_by_name=participants_by_name,
        evidence_summary=evidence_summary,
        manual_entries=manual_entries,
    )
    estimated_input_tokens = estimate_tokens(messages)
    max_input_tokens = resolve_max_input_tokens()
    max_output_tokens = resolve_max_output_tokens()
    usage = empty_llm_usage(model)
    usage["input_tokens"] = estimated_input_tokens
    if estimated_input_tokens > max_input_tokens:
        raise StageExecutionError(
            "Stage 3 evidence prompt exceeds CHRONICLE_STAGE3_MAX_INPUT_TOKENS "
            f"({estimated_input_tokens} > {max_input_tokens}). Reduce evidence limits or raise the limit explicitly."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise StageExecutionError(
            "The OpenAI Python package is not installed. Run `uv sync` or `./bin/bootstrap`, then retry."
        ) from exc

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_output_tokens,
            temperature=0,
        )
    except Exception as exc:
        raise StageExecutionError(f"OpenAI speaker mapping failed: {exc}") from exc

    choice = response.choices[0]
    content = choice.message.content
    if not content:
        raise StageExecutionError("OpenAI speaker mapping returned an empty response.")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise StageExecutionError("OpenAI speaker mapping did not return valid JSON.") from exc

    response_usage = getattr(response, "usage", None)
    if response_usage is not None:
        usage["input_tokens"] = getattr(response_usage, "prompt_tokens", usage["input_tokens"])
        usage["output_tokens"] = getattr(response_usage, "completion_tokens", None)

    speaker_map = payload.get("speaker_map")
    if not isinstance(speaker_map, list):
        raise StageExecutionError("OpenAI speaker mapping response must contain a `speaker_map` list.")
    return speaker_map, usage
