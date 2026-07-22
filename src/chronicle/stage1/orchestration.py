"""Stage 1 cloud orchestration planning."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from shlex import join as shell_join
from typing import Any, Literal

from rich.console import Console

from ..exceptions import StageExecutionError


def _normalize_repo_url(url: str) -> str:
    if url.startswith("git@github.com:"):
        return "https://github.com/" + url.removeprefix("git@github.com:")
    return url


def _default_repo_url() -> str:
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "https://github.com/<owner>/chronicle.git"
    url = result.stdout.strip()
    return _normalize_repo_url(url) if url else "https://github.com/<owner>/chronicle.git"


@dataclass(frozen=True)
class GcpStage1Config:
    project_id: str
    instance_name: str
    session_id: str
    backend: str = "whisper"
    zone: str = "us-central1-a"
    machine_type: str = "e2-standard-8"
    gpu_enabled: bool = False
    gpu_type: str = "nvidia-l4"
    gpu_count: int = 1
    boot_disk_size: str = "50GB"
    image_family: str = "ubuntu-2404-lts-amd64"
    image_project: str = "ubuntu-os-cloud"
    tags: str = "chronicle-stage1"
    labels: str = "chronicle=1,stage=stage1,mode=cloud"
    model_name: str = "large-v3-turbo"
    python_version: str = "3.11"
    worker_repo_dir: str = "/home/{user}/chronicle"
    local_output_dir: str = "./outputs"
    local_participants_file: str = "inputs/global/participants.yaml"
    repo_url: str = field(default_factory=_default_repo_url)
    repo_ref: str = "remote"

    def resolved_worker_repo_dir(self, worker_user: str) -> str:
        return self.worker_repo_dir.format(user=worker_user)


@dataclass(frozen=True)
class GcpStage1Plan:
    config: GcpStage1Config
    worker_user: str
    local_session_dir: str
    commands: dict[str, list[str]]

    def asdict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "worker_user": self.worker_user,
            "local_session_dir": self.local_session_dir,
            "commands": self.commands,
        }

    def shell_command(self, step: str) -> str:
        try:
            command = self.commands[step]
        except KeyError as exc:
            raise KeyError(f"Unknown Stage 1 cloud step: {step}") from exc
        return shell_join(command)


Stage1CloudStep = Literal[
    "preflight",
    "vm_create",
    "clone_repo",
    "bootstrap",
    "upload_session",
    "upload_participants",
    "run_stage1",
    "download_outputs",
    "teardown",
]


def _worker_repo_dir(config: GcpStage1Config, worker_user: str) -> str:
    return config.resolved_worker_repo_dir(worker_user)


def build_gcp_stage1_plan(
    *,
    config: GcpStage1Config,
    local_session_dir: Path,
    worker_user: str,
) -> GcpStage1Plan:
    worker_repo_dir = _worker_repo_dir(config, worker_user)
    worker_session_root = f"{worker_repo_dir}/inputs/sessions"
    worker_global_root = f"{worker_repo_dir}/inputs/global"
    worker_output_root = f"{worker_repo_dir}/outputs"
    instance_ref = config.instance_name

    preflight = [
        "gcloud",
        "services",
        "list",
        "--enabled",
        "--project",
        config.project_id,
    ]
    resolved_machine_type = config.machine_type
    if config.gpu_enabled and config.machine_type == "e2-standard-8":
        if "l4" in config.gpu_type.lower():
            resolved_machine_type = "g2-standard-4"
        else:
            resolved_machine_type = "n1-standard-4"

    vm_create = [
        "gcloud",
        "compute",
        "instances",
        "create",
        config.instance_name,
        "--project",
        config.project_id,
        "--zone",
        config.zone,
        "--machine-type",
        resolved_machine_type,
        "--boot-disk-size",
        config.boot_disk_size,
        "--image-family",
        config.image_family,
        "--image-project",
        config.image_project,
        "--restart-on-failure",
        "--tags",
        config.tags,
        "--labels",
        config.labels,
    ]
    if config.gpu_enabled:
        vm_create.extend(
            [
                "--accelerator",
                f"type={config.gpu_type},count={config.gpu_count}",
                "--maintenance-policy",
                "TERMINATE",
            ]
        )

    ssh = [
        "gcloud",
        "compute",
        "ssh",
        instance_ref,
        "--project",
        config.project_id,
        "--zone",
        config.zone,
    ]
    scp = [
        "gcloud",
        "compute",
        "scp",
        "--project",
        config.project_id,
        "--zone",
        config.zone,
    ]

    clone_repo = ssh + [
        "--command",
        (
            "set -euo pipefail && "
            f"if [ ! -d '{worker_repo_dir}/.git' ]; then "
            f"git clone --branch {config.repo_ref} {config.repo_url} {worker_repo_dir}; "
            "else "
            f"cd {worker_repo_dir} && git fetch origin {config.repo_ref} && git checkout {config.repo_ref} && git pull --ff-only origin {config.repo_ref}; "
            "fi"
        ),
    ]

    group_flag = "--group stage1-whisper" if config.backend == "whisper" else "--group stage1-parakeet"
    model_fetch_cmd = f" && uv run --python {config.python_version} chronicle models fetch parakeet" if config.backend == "parakeet" else ""
    bootstrap = ssh + [
        "--command",
        (
            "set -euo pipefail && "
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            'export PATH="$HOME/.local/bin:$PATH" && '
            f"uv python install {config.python_version} && "
            f"cd {worker_repo_dir} && "
            f"uv sync --python {config.python_version} --group dev {group_flag}"
            f"{model_fetch_cmd}"
        ),
    ]
    upload_session = scp + ["--recurse", local_session_dir.as_posix(), f"{instance_ref}:{worker_session_root}/"]
    upload_participants = scp + [config.local_participants_file, f"{instance_ref}:{worker_global_root}/participants.yaml"]
    device_arg = "cuda" if config.gpu_enabled else "cpu"
    run_stage1 = ssh + [
        "--command",
        (
            "set -euo pipefail && "
            f"cd {worker_repo_dir} && "
            'export PATH="$HOME/.local/bin:$PATH" && '
            f"uv run --python {config.python_version} chronicle transcribe {config.session_id} --local-worker "
            f"--backend {config.backend} --model {config.model_name} --device {device_arg}"
        ),
    ]
    download_outputs = scp + ["--recurse", f"{instance_ref}:{worker_output_root}/{config.session_id}", f"{config.local_output_dir}/"]
    teardown = [
        "gcloud",
        "compute",
        "instances",
        "delete",
        config.instance_name,
        "--project",
        config.project_id,
        "--zone",
        config.zone,
        "--quiet",
    ]

    return GcpStage1Plan(
        config=config,
        worker_user=worker_user,
        local_session_dir=local_session_dir.as_posix(),
        commands={
            "preflight": preflight,
            "vm_create": vm_create,
            "clone_repo": clone_repo,
            "bootstrap": bootstrap,
            "upload_session": upload_session,
            "upload_participants": upload_participants,
            "run_stage1": run_stage1,
            "download_outputs": download_outputs,
            "teardown": teardown,
        },
    )


def run_gcp_stage1_plan(
    plan: GcpStage1Plan,
    *,
    console: Console | None = None,
    keep_instance: bool = False,
) -> None:
    step_order: list[Stage1CloudStep] = [
        "vm_create",
        "clone_repo",
        "bootstrap",
        "upload_session",
        "upload_participants",
        "run_stage1",
        "download_outputs",
    ]
    created_instance = False
    try:
        for step in step_order:
            command = plan.commands[step]
            if console is not None:
                console.print(f"[bold]Stage 1 cloud:[/bold] `{step}`")
            subprocess.run(command, check=True)
            if step == "vm_create":
                created_instance = True
    except subprocess.CalledProcessError as exc:
        raise StageExecutionError(f"Stage 1 cloud step `{step}` failed with exit code {exc.returncode}.") from exc
    finally:
        if created_instance and not keep_instance:
            teardown = plan.commands["teardown"]
            if console is not None:
                console.print("[bold]Stage 1 cloud:[/bold] `teardown`")
            try:
                subprocess.run(teardown, check=True)
            except subprocess.CalledProcessError as exc:
                raise StageExecutionError(
                    f"Stage 1 cloud teardown failed with exit code {exc.returncode}."
                ) from exc
