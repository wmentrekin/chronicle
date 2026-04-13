"""Session manifest parsing and validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .exceptions import SessionValidationError
from .paths import DEFAULT_PARTICIPANTS_FILE, repo_relative, session_manifest_path
from .utils import load_yaml


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REQUIRED_CONTEXT_HEADINGS = (
    "## Participants",
    "## People Likely Discussed",
    "## Background",
)


@dataclass
class SessionManifest:
    session_id: str
    title: str
    interview_date: Optional[str]
    audio_files: list[str]
    participants: list[str]
    primary_interviewees: list[str]
    people_likely_discussed: list[str]
    context_doc: str
    language: str = "en"
    source_format: Optional[str] = None
    notes: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    stage1_model_preference: Optional[str] = None
    manifest_path: Optional[str] = None


@dataclass
class ValidationReport:
    manifest_path: str
    participants_file: str
    manifest: Optional[SessionManifest] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def manifest_root(manifest: SessionManifest) -> Path:
    if manifest.manifest_path:
        return Path(manifest.manifest_path).resolve().parent
    return session_manifest_path(manifest.session_id).resolve().parent


def resolve_manifest_path(session_id: str) -> Path:
    return session_manifest_path(session_id)


def resolve_manifest_relative_path(manifest: SessionManifest, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    return manifest_root(manifest) / candidate


def resolve_context_path(manifest: SessionManifest) -> Path:
    return resolve_manifest_relative_path(manifest, manifest.context_doc)


def resolve_audio_path(manifest: SessionManifest, audio_file: str) -> Path:
    return resolve_manifest_relative_path(manifest, audio_file)


def ensure_string_list(data: Any, field_name: str, report: ValidationReport) -> list[str]:
    if not isinstance(data, list) or any(not isinstance(item, str) or not item.strip() for item in data):
        report.errors.append(f"`{field_name}` must be a non-empty list of strings.")
        return []
    return [item.strip() for item in data]


def ensure_optional_string_list(
    data: Any, field_name: str, report: ValidationReport
) -> list[str]:
    if data is None:
        return []
    if not isinstance(data, list) or any(not isinstance(item, str) or not item.strip() for item in data):
        report.errors.append(f"`{field_name}` must be a list of strings when provided.")
        return []
    return [item.strip() for item in data]


def validate_iso_date(value: Optional[str], report: ValidationReport) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        report.errors.append("`interview_date` must be an ISO date string or null.")
        return None
    try:
        date.fromisoformat(value)
    except ValueError:
        report.errors.append("`interview_date` must use YYYY-MM-DD format when provided.")
        return None
    return value


def build_canonical_name_index(participants_file: Path, report: ValidationReport) -> set[str]:
    if not participants_file.exists():
        report.errors.append(f"Participants file not found: {repo_relative(participants_file)}")
        return set()

    payload = load_yaml(participants_file)
    participants = payload.get("participants")
    if not isinstance(participants, list):
        report.errors.append(
            f"`participants` list missing or invalid in {repo_relative(participants_file)}"
        )
        return set()

    canonical_names: set[str] = set()
    for participant in participants:
        if not isinstance(participant, dict):
            report.errors.append("Every participant entry must be a mapping.")
            continue
        canonical_name = participant.get("canonical_name")
        if not isinstance(canonical_name, str) or not canonical_name.strip():
            report.errors.append("Every participant entry must have a non-empty `canonical_name`.")
            continue
        canonical_names.add(canonical_name.strip())
    return canonical_names


def parse_manifest(manifest_path: Path, report: ValidationReport) -> Optional[SessionManifest]:
    if not manifest_path.exists():
        report.errors.append(f"Session manifest not found: {repo_relative(manifest_path)}")
        return None

    payload = load_yaml(manifest_path)
    if not isinstance(payload, dict):
        report.errors.append("Session manifest must be a YAML mapping.")
        return None

    session_id = payload.get("session_id")
    title = payload.get("title")
    context_doc = payload.get("context_doc")

    if not isinstance(session_id, str) or not session_id.strip():
        report.errors.append("`session_id` is required and must be a string.")
        session_id = ""
    else:
        session_id = session_id.strip()
        if not SESSION_ID_PATTERN.match(session_id):
            report.errors.append(
                "`session_id` must be filename-safe and use only letters, numbers, `.`, `_`, and `-`."
            )

    if not isinstance(title, str) or not title.strip():
        report.errors.append("`title` is required and must be a string.")
        title = ""
    else:
        title = title.strip()

    if not isinstance(context_doc, str) or not context_doc.strip():
        report.errors.append("`context_doc` is required and must be a string path.")
        context_doc = ""
    else:
        context_doc = context_doc.strip()

    manifest = SessionManifest(
        session_id=session_id,
        title=title,
        interview_date=validate_iso_date(payload.get("interview_date"), report),
        audio_files=ensure_string_list(payload.get("audio_files"), "audio_files", report),
        participants=ensure_string_list(payload.get("participants"), "participants", report),
        primary_interviewees=ensure_string_list(
            payload.get("primary_interviewees"), "primary_interviewees", report
        ),
        people_likely_discussed=ensure_string_list(
            payload.get("people_likely_discussed"), "people_likely_discussed", report
        ),
        context_doc=context_doc,
        language=payload.get("language", "en") if isinstance(payload.get("language", "en"), str) else "en",
        source_format=payload.get("source_format")
        if isinstance(payload.get("source_format"), str)
        else None,
        notes=payload.get("notes") if isinstance(payload.get("notes"), str) else None,
        tags=ensure_optional_string_list(payload.get("tags"), "tags", report),
        stage1_model_preference=payload.get("stage1_model_preference")
        if isinstance(payload.get("stage1_model_preference"), str)
        else None,
        manifest_path=manifest_path.as_posix(),
    )

    if manifest_path.parent.name != manifest.session_id:
        report.warnings.append(
            "Manifest directory name does not match `session_id`; keeping the manifest value as canonical."
        )

    if not manifest.audio_files:
        report.errors.append("`audio_files` must contain at least one audio path.")
    if not manifest.participants:
        report.errors.append("`participants` must contain at least one participant.")
    if not manifest.primary_interviewees:
        report.errors.append("`primary_interviewees` must contain at least one interviewee.")

    return manifest


def validate_manifest_paths(manifest: SessionManifest, report: ValidationReport) -> None:
    for audio_file in manifest.audio_files:
        audio_path = resolve_audio_path(manifest, audio_file)
        if not audio_path.exists():
            icloud_placeholder = audio_path.with_name(f".{audio_path.name}.icloud")
            if icloud_placeholder.exists():
                report.errors.append(
                    "Audio file is not downloaded locally: "
                    f"{repo_relative(audio_path)} (found iCloud placeholder at "
                    f"{repo_relative(icloud_placeholder)}). Download the file locally and retry."
                )
            else:
                report.errors.append(f"Audio file not found: {repo_relative(audio_path)}")

    context_path = resolve_context_path(manifest)
    if not context_path.exists():
        report.errors.append(f"Context document not found: {repo_relative(context_path)}")


def validate_names(manifest: SessionManifest, canonical_names: set[str], report: ValidationReport) -> None:
    for field_name, values in (
        ("participants", manifest.participants),
        ("primary_interviewees", manifest.primary_interviewees),
        ("people_likely_discussed", manifest.people_likely_discussed),
    ):
        for name in values:
            if name not in canonical_names:
                report.errors.append(f"`{field_name}` contains unknown canonical name: {name}")

    unknown_interviewees = set(manifest.primary_interviewees) - set(manifest.participants)
    for name in sorted(unknown_interviewees):
        report.errors.append(
            f"`primary_interviewees` must be a subset of `participants` but includes: {name}"
        )


def validate_context_doc(context_path: Path, report: ValidationReport) -> None:
    if not context_path.exists():
        return

    text = context_path.read_text(encoding="utf-8")
    for heading in REQUIRED_CONTEXT_HEADINGS:
        if heading not in text:
            report.errors.append(f"Context document is missing required heading: {heading}")

    background_match = re.search(r"## Background\s*(.+)$", text, flags=re.DOTALL)
    if not background_match:
        return

    background_text = background_match.group(1).strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", background_text) if part.strip()]
    if len(paragraphs) < 2 or len(paragraphs) > 3:
        report.warnings.append("Context document background should usually be 2-3 paragraphs.")


def validate_session_manifest(
    manifest_path: Path, participants_file: Path = DEFAULT_PARTICIPANTS_FILE
) -> ValidationReport:
    report = ValidationReport(
        manifest_path=repo_relative(manifest_path),
        participants_file=repo_relative(participants_file),
    )

    canonical_names = build_canonical_name_index(participants_file, report)
    manifest = parse_manifest(manifest_path, report)
    if manifest is None:
        return report

    report.manifest = manifest
    validate_manifest_paths(manifest, report)
    if canonical_names:
        validate_names(manifest, canonical_names, report)
    validate_context_doc(resolve_context_path(manifest), report)

    if report.ok:
        report.infos.append("Session manifest passed validation.")
        report.infos.append(
            f"Validated {len(manifest.audio_files)} audio file(s) and "
            f"{len(manifest.participants)} participant(s)."
        )

    return report


def validate_session(session_id: str, participants_file: Path = DEFAULT_PARTICIPANTS_FILE) -> ValidationReport:
    return validate_session_manifest(resolve_manifest_path(session_id), participants_file)


def render_validation_report(
    report: ValidationReport, console: Console, as_json: bool = False
) -> None:
    if as_json:
        payload = {
            "manifest_path": report.manifest_path,
            "participants_file": report.participants_file,
            "ok": report.ok,
            "manifest": asdict(report.manifest) if report.manifest else None,
            "errors": report.errors,
            "warnings": report.warnings,
            "infos": report.infos,
        }
        console.print_json(data=payload)
        return

    summary = Table(title="Session Validation")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Manifest", report.manifest_path)
    summary.add_row("Participants file", report.participants_file)
    summary.add_row("Status", "OK" if report.ok else "FAILED")
    if report.manifest is not None:
        summary.add_row("Session ID", report.manifest.session_id)
        summary.add_row("Title", report.manifest.title)
        summary.add_row("Audio files", str(len(report.manifest.audio_files)))
        summary.add_row("Participants", str(len(report.manifest.participants)))
    console.print(summary)

    if report.infos:
        console.print(Panel("\n".join(f"- {item}" for item in report.infos), title="Info"))
    if report.warnings:
        console.print(Panel("\n".join(f"- {item}" for item in report.warnings), title="Warnings"))
    if report.errors:
        console.print(
            Panel("\n".join(f"- {item}" for item in report.errors), title="Errors", style="red")
        )


def require_valid_session(
    session_id: str,
    console: Console,
    participants_file: Path = DEFAULT_PARTICIPANTS_FILE,
) -> SessionManifest:
    report = validate_session(session_id, participants_file)
    render_validation_report(report, console)
    if not report.ok or report.manifest is None:
        raise SessionValidationError(f"Session `{session_id}` is invalid.")
    return report.manifest
