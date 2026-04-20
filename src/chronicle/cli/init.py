"""Initialization and model management commands."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.panel import Panel

from ..exceptions import StageExecutionError
from ..paths import DEFAULT_PARAKEET_MODEL_DIR, repo_relative
from ..stage1.service import DEFAULT_STAGE1_PARAKEET_MODEL, ensure_local_parakeet_model
from ..stage3.llm import DEFAULT_MODEL as DEFAULT_STAGE3_OLLAMA_MODEL
from ..stage3.llm import list_ollama_models, pull_ollama_model, resolve_stage3_model
from .common import console, install_chronicle_link


def register(app: typer.Typer) -> None:
    models_app = typer.Typer(add_completion=False, help="Manage local model files.")
    app.add_typer(models_app, name="models")

    @app.command("init")
    def init_command(
        model_name: str = typer.Option(
            DEFAULT_STAGE1_PARAKEET_MODEL,
            "--model",
            help="Remote Hugging Face model id to fetch into Chronicle-managed storage.",
        ),
        parakeet_model_dir: Path = typer.Option(
            DEFAULT_PARAKEET_MODEL_DIR,
            "--parakeet-model-dir",
            file_okay=False,
            dir_okay=True,
            help="Chronicle-managed local directory for Parakeet model files.",
        ),
        skip_model_download: bool = typer.Option(
            False,
            "--skip-model-download",
            help="Skip fetching the managed local Parakeet model and only validate the local runtime.",
        ),
        stage3_model: str | None = typer.Option(
            None,
            "--stage3-model",
            help=(
                "Ollama model to validate or pull for Stage 3 speaker identification. "
                f"Defaults to CHRONICLE_STAGE3_MODEL or `{DEFAULT_STAGE3_OLLAMA_MODEL}`."
            ),
        ),
        skip_ollama_setup: bool = typer.Option(
            False,
            "--skip-ollama-setup",
            help="Skip local Ollama model validation for Stage 3.",
        ),
        install_link: bool = typer.Option(
            False,
            "--install-link",
            help="Install/update a stable user-level `chronicle` symlink.",
        ),
        link_path: Path = typer.Option(
            Path.home() / ".local" / "bin" / "chronicle",
            "--link-path",
            file_okay=True,
            dir_okay=False,
            help="Destination path for the installed Chronicle symlink.",
        ),
    ) -> None:
        """Initialize Chronicle's default local runtime."""
        if not skip_model_download:
            try:
                resolved = ensure_local_parakeet_model(
                    model_name=model_name,
                    model_dir=parakeet_model_dir,
                    allow_download=True,
                )
            except StageExecutionError as exc:
                console.print(Panel(str(exc), title="Initialization Failed", style="red"))
                raise typer.Exit(code=1) from exc
            console.print(
                Panel(
                    f"Parakeet model files are available at `{repo_relative(resolved)}`.",
                    title="Model Ready",
                )
            )

        if not skip_ollama_setup:
            resolved_stage3_model = resolve_stage3_model(stage3_model)
            try:
                available_models = list_ollama_models()
            except StageExecutionError as exc:
                console.print(Panel(str(exc), title="Ollama Setup Failed", style="red"))
                raise typer.Exit(code=1) from exc

            if resolved_stage3_model not in available_models:
                if not sys.stdin.isatty():
                    console.print(
                        Panel(
                            (
                                f"Ollama model `{resolved_stage3_model}` is not installed and this shell is non-interactive.\n"
                                f"Run `ollama pull {resolved_stage3_model}` or rerun `chronicle init` in an interactive shell."
                            ),
                            title="Ollama Model Missing",
                            style="red",
                        )
                    )
                    raise typer.Exit(code=1)
                should_pull = typer.confirm(
                    f"Ollama model `{resolved_stage3_model}` is not installed. Pull it now?",
                    default=True,
                )
                if not should_pull:
                    console.print(
                        Panel(
                            f"Ollama model `{resolved_stage3_model}` is required for default Stage 3 `llm` mode.",
                            title="Ollama Model Missing",
                            style="red",
                        )
                    )
                    raise typer.Exit(code=1)
                try:
                    pull_ollama_model(resolved_stage3_model)
                except StageExecutionError as exc:
                    console.print(Panel(str(exc), title="Ollama Pull Failed", style="red"))
                    raise typer.Exit(code=1) from exc

            console.print(
                Panel(
                    (
                        f"Ollama model `{resolved_stage3_model}` is available in Ollama's local model store. "
                        "Chronicle does not copy Ollama models into repo storage."
                    ),
                    title="Stage 3 Model Ready",
                )
            )

        if install_link:
            try:
                installed_link = install_chronicle_link(link_path)
            except StageExecutionError as exc:
                console.print(Panel(str(exc), title="Link Install Failed", style="red"))
                raise typer.Exit(code=1) from exc
            console.print(
                Panel(
                    (
                        f"Installed `chronicle` symlink at `{installed_link.as_posix()}`.\n"
                        "Ensure the parent directory is on your PATH."
                    ),
                    title="Link Ready",
                )
            )

        console.print(
            Panel(
                "Chronicle is ready. Use `chronicle transcribe <session_id>` for Stage 1.",
                title="Initialization Complete",
            )
        )

    @models_app.command("fetch")
    def fetch_model_command(
        model_key: str = typer.Argument(..., help="Model key to fetch. Currently supported: parakeet."),
        model_name: str = typer.Option(
            DEFAULT_STAGE1_PARAKEET_MODEL,
            "--model",
            help="Remote Hugging Face model id to fetch.",
        ),
        model_dir: Path = typer.Option(
            DEFAULT_PARAKEET_MODEL_DIR,
            "--model-dir",
            file_okay=False,
            dir_okay=True,
            help="Local destination for the downloaded model files.",
        ),
    ) -> None:
        """Download a managed local model for Chronicle."""
        if model_key.strip().lower() != "parakeet":
            console.print(Panel("Only `parakeet` is currently supported.", title="Unsupported Model", style="red"))
            raise typer.Exit(code=1)
        try:
            resolved = ensure_local_parakeet_model(
                model_name=model_name,
                model_dir=model_dir,
                allow_download=True,
            )
        except StageExecutionError as exc:
            console.print(Panel(str(exc), title="Model Fetch Failed", style="red"))
            raise typer.Exit(code=1) from exc
        console.print(
            Panel(
                f"Parakeet model files are available at `{repo_relative(resolved)}`.",
                title="Model Ready",
            )
        )
