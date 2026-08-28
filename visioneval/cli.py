"""Command-line entry point for VisionEval."""

from pathlib import Path

import typer

app = typer.Typer(
    name="visioneval",
    help="Evaluate vision models: Phase 1 classification CI, plus multimodal eval.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Group VisionEval commands under one stable CLI entry point."""


@app.command()
def run(
    suite: Path | None = typer.Argument(None),
    update_baseline: bool = typer.Option(False),
) -> None:
    """Run a Phase 1 classification suite and return a non-zero exit code for regressions."""
    if suite is None:
        typer.echo("VisionEval Phase 1 scaffold is ready; provide a suite path to evaluate.")
        return
    from visioneval.core.runner import run_suite

    result = run_suite(suite, update_baseline=update_baseline)
    typer.echo(f"Accuracy: {result.summary.accuracy:.4f}")
    if result.regression and result.regression.is_regression:
        raise typer.Exit(code=1)


@app.command("multimodal")
def multimodal(
    config: Path = typer.Argument(..., help="Path to a multimodal eval YAML config."),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write a JSON report."),
    markdown_out: Path | None = typer.Option(None, "--markdown-out", help="Write a Markdown report."),
) -> None:
    """Run the multimodal evaluation layer (metrics, robustness, profiling, reports)."""
    from visioneval.multimodal.pipeline import run_multimodal_eval

    result = run_multimodal_eval(config, json_path=json_out, markdown_path=markdown_out)
    n_clean = sum(1 for row in result["samples"] if not row.get("corruption"))
    typer.echo(
        f"Multimodal eval '{result['name']}': {n_clean} clean sample-model pairs, "
        f"{len(result['samples'])} total rows."
    )


if __name__ == "__main__":
    app()
