"""Command-line entry point for VisionEval."""

from pathlib import Path

import typer

from visioneval.core.runner import run_suite

app = typer.Typer(name="visioneval", help="Evaluate image-classification models for regressions before deployment.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Group VisionEval commands under one stable CLI entry point."""


@app.command()
def run(
    suite: Path | None = typer.Argument(None), update_baseline: bool = typer.Option(False)
) -> None:
    """Run a suite and return a non-zero exit code for regressions."""
    if suite is None:
        typer.echo("VisionEval Phase 1 scaffold is ready; provide a suite path to evaluate.")
        return
    result = run_suite(suite, update_baseline=update_baseline)
    typer.echo(f"Accuracy: {result.summary.accuracy:.4f}")
    if result.regression and result.regression.is_regression:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()