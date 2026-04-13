"""Repository path helpers."""

from __future__ import annotations

from pathlib import Path


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "AGENTS.md").exists():
            return parent
    return current.parents[2]


REPO_ROOT = find_repo_root()
INPUTS_ROOT = REPO_ROOT / "inputs"
GLOBAL_INPUTS_ROOT = INPUTS_ROOT / "global"
SESSION_INPUTS_ROOT = INPUTS_ROOT / "sessions"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
MODELS_ROOT = REPO_ROOT / "models"
DOCS_ROOT = REPO_ROOT / "docs"
AGENT_CONTEXT_ROOT = REPO_ROOT / "agent-context"
DEFAULT_PARTICIPANTS_FILE = GLOBAL_INPUTS_ROOT / "participants.yaml"
DEFAULT_STAGE1_CACHE_DIR = REPO_ROOT / ".cache" / "huggingface" / "hub"
DEFAULT_PARAKEET_MODEL_DIR = MODELS_ROOT / "parakeet-ctc-0.6b"


def input_session_dir(session_id: str) -> Path:
    return SESSION_INPUTS_ROOT / session_id


def input_audio_dir(session_id: str) -> Path:
    return input_session_dir(session_id) / "audio"


def session_manifest_path(session_id: str) -> Path:
    return input_session_dir(session_id) / "session.yaml"


def session_context_path(session_id: str) -> Path:
    return input_session_dir(session_id) / "context.md"


def output_session_dir(session_id: str) -> Path:
    return OUTPUTS_ROOT / session_id


def ensure_output_dirs(session_id: str) -> dict[str, Path]:
    root = output_session_dir(session_id)
    directories = {
        "root": root,
        "runs": root / "runs",
        "stage1": root / "stage1",
        "stage2": root / "stage2",
        "stage3": root / "stage3",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()
