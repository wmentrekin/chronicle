"""Stage 2 cloud orchestration planning."""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
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


def _default_repo_ref() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        branch = result.stdout.strip()
        if branch and branch != "HEAD":
            return branch
    except Exception:
        pass
    return "main"


@dataclass(frozen=True)
class GcpStage2Config:
    project_id: str
    instance_name: str
    session_id: str
    backend: str = "speechbrain"
    zone: str = "us-central1-a"
    machine_type: str = "e2-standard-8"
    gpu_enabled: bool = False
    gpu_type: str = "nvidia-l4"
    gpu_count: int = 1
    boot_disk_size: str = "50GB"
    image_family: str = "ubuntu-2404-lts-amd64"
    image_project: str = "ubuntu-os-cloud"
    tags: str = "chronicle-stage2"
    labels: str = "chronicle=1,stage=stage2,mode=cloud"
    python_version: str = "3.11"
    worker_repo_dir: str = "/home/{user}/chronicle"
    local_output_dir: str = "./outputs"
    local_participants_file: str = "inputs/global/participants.yaml"
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    repo_url: str = field(default_factory=_default_repo_url)
    repo_ref: str = field(default_factory=_default_repo_ref)

    def resolved_worker_repo_dir(self, worker_user: str) -> str:
        return self.worker_repo_dir.format(user=worker_user)


@dataclass(frozen=True)
class GcpStage2Plan:
    config: GcpStage2Config
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
            raise KeyError(f"Unknown Stage 2 cloud step: {step}") from exc
        return shell_join(command)


Stage2CloudStep = Literal[
    "preflight",
    "vm_create",
    "clone_repo",
    "bootstrap",
    "upload_session",
    "upload_participants",
    "run_stage2",
    "download_outputs",
    "teardown",
]


def _worker_repo_dir(config: GcpStage2Config, worker_user: str) -> str:
    return config.resolved_worker_repo_dir(worker_user)


def build_gcp_stage2_plan(
    *,
    config: GcpStage2Config,
    local_session_dir: Path,
    worker_user: str,
) -> GcpStage2Plan:
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
    resolved_image_family = config.image_family
    resolved_image_project = config.image_project
    if config.gpu_enabled:
        if config.machine_type == "e2-standard-8":
            if "l4" in config.gpu_type.lower():
                resolved_machine_type = "g2-standard-4"
            else:
                resolved_machine_type = "n1-standard-4"
        if config.image_family == "ubuntu-2404-lts-amd64" and config.image_project == "ubuntu-os-cloud":
            resolved_image_family = "common-cu129-ubuntu-2404-nvidia-580"
            resolved_image_project = "deeplearning-platform-release"

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
        resolved_image_family,
        "--image-project",
        resolved_image_project,
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

    group_flag = "--group stage2-pyannote" if config.backend == "pyannote" else "--group stage2-speechbrain"
    bootstrap = ssh + [
        "--command",
        (
            "set -euo pipefail && "
            f"mkdir -p {worker_repo_dir}/inputs/sessions {worker_repo_dir}/inputs/global {worker_repo_dir}/outputs && "
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            'export PATH="$HOME/.local/bin:$PATH" && '
            f"uv python install {config.python_version} && "
            f"cd {worker_repo_dir} && "
            f"uv sync --python {config.python_version} --group dev {group_flag}"
        ),
    ]
    upload_session = scp + ["--recurse", local_session_dir.as_posix(), f"{instance_ref}:{worker_session_root}/"]
    upload_participants = scp + [config.local_participants_file, f"{instance_ref}:{worker_global_root}/participants.yaml"]
    device_arg = "cuda" if config.gpu_enabled else "cpu"
    speaker_flags = ""
    if config.num_speakers is not None:
        speaker_flags += f" --num-speakers {config.num_speakers}"
    if config.min_speakers is not None:
        speaker_flags += f" --min-speakers {config.min_speakers}"
    if config.max_speakers is not None:
        speaker_flags += f" --max-speakers {config.max_speakers}"

    run_stage2 = ssh + [
        "--command",
        (
            "set -euo pipefail && "
            f"cd {worker_repo_dir} && "
            'export PATH="$HOME/.local/bin:$PATH" && '
            f"uv run --python {config.python_version} chronicle diarize {config.session_id} --local-worker "
            f"--device {device_arg}{speaker_flags} --force"
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

    return GcpStage2Plan(
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
            "run_stage2": run_stage2,
            "download_outputs": download_outputs,
            "teardown": teardown,
        },
    )


def run_gcp_stage2_plan(
    plan: GcpStage2Plan,
    *,
    console: Console | None = None,
    keep_instance: bool = False,
) -> None:
    step_order: list[Stage2CloudStep] = [
        "vm_create",
        "clone_repo",
        "bootstrap",
        "upload_session",
        "upload_participants",
        "run_stage2",
        "download_outputs",
    ]
    created_instance = False
    current_plan = plan

    candidate_zones = [
        plan.config.zone,
        "us-east1-d",
        "us-central1-b",
        "us-central1-c",
        "us-east4-b",
        "us-east4-c",
        "us-east1-c",
        "us-west1-a",
        "us-west1-b",
        "us-west4-a",
        "us-west4-b",
        "us-east4-a",
    ]
    seen: set[str] = set()
    zones_to_try = [z for z in candidate_zones if not (z in seen or seen.add(z))]

    try:
        for step in step_order:
            if step == "vm_create":
                vm_created_successfully = False
                last_exc: subprocess.CalledProcessError | None = None
                for candidate_zone in zones_to_try:
                    if candidate_zone != current_plan.config.zone:
                        alt_config = replace(current_plan.config, zone=candidate_zone)
                        current_plan = build_gcp_stage2_plan(
                            config=alt_config,
                            local_session_dir=Path(current_plan.local_session_dir),
                            worker_user=current_plan.worker_user,
                        )
                    command = current_plan.commands["vm_create"]
                    if console is not None:
                        console.print(f"[bold]Stage 2 cloud:[/bold] `vm_create` (zone: {candidate_zone})")
                    res = subprocess.run(command)
                    if res.returncode == 0:
                        created_instance = True
                        vm_created_successfully = True
                        if console is not None:
                            console.print("[bold]Stage 2 cloud:[/bold] Waiting for SSH daemon initialization (10s)...")
                        time.sleep(10)
                        break
                    else:
                        last_exc = subprocess.CalledProcessError(res.returncode, command)
                        if console is not None:
                            console.print(
                                f"[yellow]Zone `{candidate_zone}` failed (likely stockout). Trying fallback zone...[/yellow]"
                            )
                if not vm_created_successfully:
                    raise StageExecutionError(
                        f"Stage 2 cloud step `vm_create` failed across all candidate zones ({', '.join(zones_to_try)})."
                    ) from last_exc
            else:
                command = current_plan.commands[step]
                if console is not None:
                    console.print(f"[bold]Stage 2 cloud:[/bold] `{step}`")

                max_retries = 3 if step in ("clone_repo", "bootstrap", "run_stage2") else 1
                for attempt in range(1, max_retries + 1):
                    res = subprocess.run(command)
                    if res.returncode == 0:
                        break
                    if attempt < max_retries and res.returncode == 255:
                        if console is not None:
                            console.print(f"[yellow]SSH attempt {attempt} failed (code 255). Retrying in 5s...[/yellow]")
                        time.sleep(5)
                    else:
                        raise StageExecutionError(f"Stage 2 cloud step `{step}` failed with exit code {res.returncode}.")
    except subprocess.CalledProcessError as exc:
        raise StageExecutionError(f"Stage 2 cloud step `{step}` failed with exit code {exc.returncode}.") from exc
    finally:
        if created_instance and not keep_instance:
            teardown = current_plan.commands["teardown"]
            if console is not None:
                console.print("[bold]Stage 2 cloud:[/bold] `teardown`")
            try:
                subprocess.run(teardown, check=True)
            except subprocess.CalledProcessError as exc:
                raise StageExecutionError(
                    f"Stage 2 cloud teardown failed with exit code {exc.returncode}."
                ) from exc
