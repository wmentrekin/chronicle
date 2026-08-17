"""Stage 3 cloud orchestration planning."""

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
class GcpStage3Config:
    project_id: str
    instance_name: str
    session_id: str
    mode: str = "llm"
    backend: str = "ollama_decomposed"
    model: str = "llama3.2"
    zone: str = "us-central1-a"
    machine_type: str = "e2-standard-8"
    gpu_enabled: bool = False
    gpu_type: str = "nvidia-l4"
    gpu_count: int = 1
    boot_disk_size: str = "50GB"
    image_family: str = "ubuntu-2404-lts-amd64"
    image_project: str = "ubuntu-os-cloud"
    tags: str = "chronicle-stage3"
    labels: str = "chronicle=1,stage=stage3,mode=cloud"
    python_version: str = "3.11"
    worker_repo_dir: str = "/home/{user}/chronicle"
    local_output_dir: str = "./outputs"
    local_participants_file: str = "inputs/global/participants.yaml"
    speaker_map_file: str | None = None
    repo_url: str = field(default_factory=_default_repo_url)
    repo_ref: str = field(default_factory=_default_repo_ref)

    def resolved_worker_repo_dir(self, worker_user: str) -> str:
        return self.worker_repo_dir.format(user=worker_user)


@dataclass(frozen=True)
class GcpStage3Plan:
    config: GcpStage3Config
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
            raise KeyError(f"Unknown Stage 3 cloud step: {step}") from exc
        return shell_join(command)


Stage3CloudStep = Literal[
    "preflight",
    "vm_create",
    "clone_repo",
    "bootstrap",
    "upload_session",
    "upload_prior_stages",
    "upload_participants",
    "run_stage3",
    "download_outputs",
    "teardown",
]


def resolve_gcp_worker_user() -> str:
    try:
        res = subprocess.run(
            ["gcloud", "config", "get-value", "account"],
            check=True,
            capture_output=True,
            text=True,
        )
        account = res.stdout.strip()
        if account and "@" in account:
            user = account.split("@")[0]
            sanitized = "".join(c if c.isalnum() or c in "_-" else "_" for c in user)
            if sanitized and sanitized[0].isalpha():
                return sanitized
    except Exception:
        pass

    try:
        import getpass
        local_user = getpass.getuser()
        sanitized = "".join(c if c.isalnum() or c in "_-" else "_" for c in local_user)
        if sanitized and sanitized[0].isalpha():
            return sanitized
    except Exception:
        pass

    return "chronicle"


def sanitize_gcp_instance_name(raw_name: str) -> str:
    cleaned = "".join(c.lower() if (c.isalnum() or c == "-") else "-" for c in raw_name)
    cleaned = cleaned.strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"vm-{cleaned}" if cleaned else "vm-stage3"
    return cleaned[:63].rstrip("-")


def default_stage3_instance_name(session_id: str) -> str:
    return sanitize_gcp_instance_name(f"chronicle-stage3-{session_id}")


def build_gcp_stage3_plan(
    *,
    config: GcpStage3Config,
    local_session_dir: Path,
    worker_user: str | None = None,
) -> GcpStage3Plan:
    worker_user = worker_user or resolve_gcp_worker_user()
    remote_repo = config.resolved_worker_repo_dir(worker_user)
    target = f"{worker_user}@{config.instance_name}"

    preflight = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        config.instance_name,
        "--project",
        config.project_id,
        "--zone",
        config.zone,
    ]

    machine = config.machine_type
    if config.gpu_enabled:
        if config.gpu_type == "nvidia-l4":
            machine = "g2-standard-4"
        elif config.gpu_type == "nvidia-tesla-t4":
            machine = "n1-standard-4"

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
        machine,
        "--image-family",
        config.image_family,
        "--image-project",
        config.image_project,
        "--boot-disk-size",
        config.boot_disk_size,
        "--tags",
        config.tags,
        "--labels",
        config.labels,
    ]

    if config.gpu_enabled:
        vm_create.extend([
            "--accelerator",
            f"type={config.gpu_type},count={config.gpu_count}",
            "--maintenance-policy",
            "TERMINATE",
        ])

    clone_repo = [
        "gcloud",
        "compute",
        "ssh",
        target,
        "--project",
        config.project_id,
        "--zone",
        config.zone,
        "--command",
        f"git clone {config.repo_url} {remote_repo} && cd {remote_repo} && git checkout {config.repo_ref}",
    ]

    bootstrap_script = (
        f"cd {remote_repo} && "
        "sudo apt-get update && sudo apt-get install -y curl ffmpeg git build-essential && "
        "curl -fsSL https://ollama.com/install.sh | sh && "
        "sudo systemctl start ollama || (ollama serve >/dev/null 2>&1 &) && "
        "sleep 5 && "
        f"ollama pull {config.model} && "
        "curl -LsSf https://astral.sh/uv/install.sh | sh && "
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "uv venv --python 3.11 && "
        "uv sync --all-groups && "
        "mkdir -p inputs/sessions inputs/global outputs"
    )

    bootstrap = [
        "gcloud",
        "compute",
        "ssh",
        target,
        "--project",
        config.project_id,
        "--zone",
        config.zone,
        "--command",
        bootstrap_script,
    ]

    upload_session = [
        "gcloud",
        "compute",
        "scp",
        "--recurse",
        local_session_dir.as_posix(),
        f"{target}:{remote_repo}/inputs/sessions/",
        "--project",
        config.project_id,
        "--zone",
        config.zone,
    ]

    upload_prior_stages = [
        "gcloud",
        "compute",
        "scp",
        "--recurse",
        f"{config.local_output_dir}/{config.session_id}",
        f"{target}:{remote_repo}/outputs/",
        "--project",
        config.project_id,
        "--zone",
        config.zone,
    ]

    upload_participants = [
        "gcloud",
        "compute",
        "scp",
        config.local_participants_file,
        f"{target}:{remote_repo}/inputs/global/participants.yaml",
        "--project",
        config.project_id,
        "--zone",
        config.zone,
    ]

    run_cmd = (
        f"cd {remote_repo} && "
        "sudo systemctl start ollama || (ollama serve >/dev/null 2>&1 &) && "
        "sleep 5 && "
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        f"uv run chronicle identify {config.session_id} --mode {config.mode} --backend {config.backend} --model {config.model} --force"
    )

    run_stage3 = [
        "gcloud",
        "compute",
        "ssh",
        target,
        "--project",
        config.project_id,
        "--zone",
        config.zone,
        "--command",
        run_cmd,
    ]

    local_out = Path(config.local_output_dir) / config.session_id
    local_out.mkdir(parents=True, exist_ok=True)

    download_outputs = [
        "gcloud",
        "compute",
        "scp",
        "--recurse",
        f"{target}:{remote_repo}/outputs/{config.session_id}/stage3",
        local_out.as_posix(),
        "--project",
        config.project_id,
        "--zone",
        config.zone,
    ]

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

    return GcpStage3Plan(
        config=config,
        worker_user=worker_user,
        local_session_dir=local_session_dir.as_posix(),
        commands={
            "preflight": preflight,
            "vm_create": vm_create,
            "clone_repo": clone_repo,
            "bootstrap": bootstrap,
            "upload_session": upload_session,
            "upload_prior_stages": upload_prior_stages,
            "upload_participants": upload_participants,
            "run_stage3": run_stage3,
            "download_outputs": download_outputs,
            "teardown": teardown,
        },
    )


def run_gcp_stage3_plan(
    plan: GcpStage3Plan,
    *,
    console: Console | None = None,
    keep_instance: bool = False,
) -> None:
    step_order: list[Stage3CloudStep] = [
        "vm_create",
        "clone_repo",
        "bootstrap",
        "upload_session",
        "upload_prior_stages",
        "upload_participants",
        "run_stage3",
        "download_outputs",
    ]
    created_instance = False
    current_plan = plan

    candidate_configs: list[tuple[str, str]] = [
        (plan.config.zone, plan.config.gpu_type),
        ("us-central1-a", "nvidia-l4"),
        ("us-central1-b", "nvidia-l4"),
        ("us-central1-c", "nvidia-l4"),
        ("us-east1-c", "nvidia-l4"),
        ("us-east1-d", "nvidia-l4"),
        ("us-east4-a", "nvidia-l4"),
        ("us-east4-b", "nvidia-l4"),
        ("us-east4-c", "nvidia-l4"),
        ("us-west1-a", "nvidia-l4"),
        ("us-west1-b", "nvidia-l4"),
        ("us-west4-a", "nvidia-l4"),
        ("us-west4-b", "nvidia-l4"),
        ("us-south1-a", "nvidia-l4"),
        ("us-south1-b", "nvidia-l4"),
        ("us-central1-a", "nvidia-tesla-t4"),
        ("us-central1-b", "nvidia-tesla-t4"),
        ("us-central1-c", "nvidia-tesla-t4"),
        ("us-central1-f", "nvidia-tesla-t4"),
        ("us-east1-b", "nvidia-tesla-t4"),
        ("us-east1-c", "nvidia-tesla-t4"),
        ("us-east1-d", "nvidia-tesla-t4"),
        ("us-east4-a", "nvidia-tesla-t4"),
        ("us-east4-b", "nvidia-tesla-t4"),
        ("us-west1-a", "nvidia-tesla-t4"),
        ("us-west1-b", "nvidia-tesla-t4"),
        ("us-west1-c", "nvidia-tesla-t4"),
        ("us-west2-a", "nvidia-tesla-t4"),
        ("us-west2-b", "nvidia-tesla-t4"),
        ("us-west3-a", "nvidia-tesla-t4"),
        ("us-west4-a", "nvidia-tesla-t4"),
        ("us-south1-a", "nvidia-tesla-t4"),
        ("us-south1-b", "nvidia-tesla-t4"),
    ] if plan.config.gpu_enabled else [(z, plan.config.gpu_type) for z in ["us-central1-a", "us-east1-d", "us-central1-b", "us-central1-c"]]

    seen_configs: set[tuple[str, str]] = set()
    configs_to_try = [c for c in candidate_configs if not (c in seen_configs or seen_configs.add(c))]

    try:
        for step in step_order:
            if step == "vm_create":
                vm_created_successfully = False
                last_exc: subprocess.CalledProcessError | None = None
                for candidate_zone, candidate_gpu in configs_to_try:
                    candidate_machine = "g2-standard-4" if candidate_gpu == "nvidia-l4" else "n1-standard-4"
                    alt_config = replace(
                        current_plan.config,
                        zone=candidate_zone,
                        gpu_type=candidate_gpu,
                        machine_type=candidate_machine if current_plan.config.gpu_enabled else current_plan.config.machine_type,
                    )
                    current_plan = build_gcp_stage3_plan(
                        config=alt_config,
                        local_session_dir=Path(current_plan.local_session_dir),
                        worker_user=current_plan.worker_user,
                    )
                    command = current_plan.commands["vm_create"]
                    if console is not None:
                        console.print(f"[bold]Stage 3 cloud:[/bold] `vm_create` (zone: {candidate_zone}, gpu: {candidate_gpu})")
                    res = subprocess.run(command)
                    if res.returncode == 0:
                        created_instance = True
                        vm_created_successfully = True
                        if console is not None:
                            console.print("[bold]Stage 3 cloud:[/bold] Waiting for SSH daemon initialization (20s)...")
                        time.sleep(20)
                        break
                    else:
                        last_exc = subprocess.CalledProcessError(res.returncode, command)
                        if console is not None:
                            console.print(
                                f"[yellow]Config `{candidate_zone}` ({candidate_gpu}) failed (likely stockout). Trying fallback...[/yellow]"
                            )
                if not vm_created_successfully:
                    raise StageExecutionError(
                        f"Stage 3 cloud step `vm_create` failed across all candidate configs."
                    ) from last_exc
            else:
                command = current_plan.commands[step]
                if console is not None:
                    console.print(f"[bold]Stage 3 cloud:[/bold] `{step}`")
                
                # SSH/SCP retry logic for key propagation and network resilience
                if step in ("clone_repo", "bootstrap", "upload_session", "upload_prior_stages", "upload_participants", "download_outputs"):
                    step_success = False
                    for attempt in range(1, 5):
                        res = subprocess.run(command)
                        if res.returncode == 0:
                            step_success = True
                            break
                        if console is not None:
                            console.print(f"[yellow]Stage 3 cloud `{step}` attempt {attempt} failed, retrying in 5s...[/yellow]")
                        time.sleep(5)
                    if not step_success:
                        raise StageExecutionError(f"Stage 3 cloud step `{step}` failed after 4 attempts.")
                else:
                    res = subprocess.run(command)
                    if res.returncode != 0:
                        raise StageExecutionError(f"Stage 3 cloud step `{step}` failed with exit code {res.returncode}.")
    finally:
        if created_instance and not keep_instance:
            if console is not None:
                console.print("[bold]Stage 3 cloud:[/bold] `teardown`")
            subprocess.run(current_plan.commands["teardown"])
