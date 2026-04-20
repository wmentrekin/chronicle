"""Local Ollama speaker-map assignment for Stage 3."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..exceptions import StageExecutionError
from ..session import SessionManifest
from .prompts import build_speaker_map_prompt, estimate_tokens
from .schemas import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    empty_llm_usage,
)

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS = 600
OLLAMA_PULL_TIMEOUT_SECONDS = 1800


def resolve_stage3_model(cli_model: str | None) -> str:
    if cli_model:
        return cli_model
    return os.environ.get("CHRONICLE_STAGE3_MODEL", DEFAULT_MODEL)


def resolve_ollama_host() -> str:
    return (
        os.environ.get("CHRONICLE_OLLAMA_HOST")
        or os.environ.get("OLLAMA_HOST")
        or DEFAULT_OLLAMA_HOST
    ).rstrip("/")


def resolve_max_input_tokens() -> int:
    return _env_int("CHRONICLE_STAGE3_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS)


def resolve_max_output_tokens() -> int:
    return _env_int("CHRONICLE_STAGE3_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)


def resolve_ollama_request_timeout() -> int:
    return _env_int("CHRONICLE_OLLAMA_REQUEST_TIMEOUT_SECONDS", DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS)


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


def require_ollama_config(model: str) -> None:
    available_models = list_ollama_models()
    if model in available_models:
        return
    raise StageExecutionError(
        "Stage 3 `llm` mode requires a local Ollama model before final identity artifacts are written.\n\n"
        "Setup:\n"
        "1. Install and start Ollama so the local API is reachable.\n"
        f"2. Pull the Stage 3 model with `ollama pull {model}` or run `chronicle init`.\n"
        "3. Override the model with `CHRONICLE_STAGE3_MODEL=...` or `--model ...`.\n"
        "4. Use `chronicle identify <session_id> --mode align-only` for local anonymous alignment.\n"
        "5. Use `chronicle identify <session_id> --mode manual --speaker-map <path>` for local identity assignment."
    )


def list_ollama_models() -> set[str]:
    payload = _ollama_get_json("/api/tags")
    models = payload.get("models")
    if not isinstance(models, list):
        raise StageExecutionError("Ollama `/api/tags` response did not contain a `models` list.")
    names: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        name = model.get("name") or model.get("model")
        if isinstance(name, str):
            names.add(name)
    return names


def pull_ollama_model(model: str) -> None:
    _ollama_post_json(
        "/api/pull",
        {"name": model, "stream": False},
        timeout=OLLAMA_PULL_TIMEOUT_SECONDS,
    )


def run_ollama_speaker_mapping(
    *,
    manifest: SessionManifest,
    context_text: str,
    participants_by_name: dict[str, dict[str, Any]],
    evidence_summary: dict[str, Any],
    manual_entries: list[dict[str, Any]],
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_ollama_config(model)
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

    payload = _ollama_post_json(
        "/api/chat",
        {
            "model": model,
            "messages": messages,
            "think": False,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": max_output_tokens,
            },
        },
    )
    message = payload.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not content:
        raise StageExecutionError("Ollama speaker mapping returned an empty response.")
    try:
        response_payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise StageExecutionError("Ollama speaker mapping did not return valid JSON.") from exc

    usage["input_tokens"] = payload.get("prompt_eval_count", usage["input_tokens"])
    usage["output_tokens"] = payload.get("eval_count")

    speaker_map = response_payload.get("speaker_map")
    if not isinstance(speaker_map, list):
        raise StageExecutionError("Ollama speaker mapping response must contain a `speaker_map` list.")
    return speaker_map, usage


def _ollama_get_json(path: str) -> dict[str, Any]:
    request = Request(f"{resolve_ollama_host()}{path}", method="GET")
    return _send_ollama_request(request, timeout=resolve_ollama_request_timeout())


def _ollama_post_json(
    path: str,
    payload: dict[str, Any],
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    request = Request(
        f"{resolve_ollama_host()}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _send_ollama_request(request, timeout=timeout or resolve_ollama_request_timeout())


def _send_ollama_request(request: Request, *, timeout: int) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise StageExecutionError(f"Ollama API request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise StageExecutionError(
            "Could not reach local Ollama API. Start Ollama, then retry. "
            f"Host: {resolve_ollama_host()}. Error: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise StageExecutionError(f"Ollama API request timed out after {timeout} seconds.") from exc
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise StageExecutionError("Ollama API returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise StageExecutionError("Ollama API returned a non-object JSON response.")
    return payload
