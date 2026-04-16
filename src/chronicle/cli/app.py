"""Typer app construction."""

from __future__ import annotations

import typer

from . import benchmark, init, stage1, stage2, stage3, stage4, validate


app = typer.Typer(
    add_completion=False,
    help="Chronicle multi-stage agentic audio-processing workflow CLI.",
    no_args_is_help=True,
)

validate.register(app)
init.register(app)
stage1.register(app)
benchmark.register(app)
stage2.register(app)
stage3.register(app)
stage4.register(app)


def main() -> None:
    app()
