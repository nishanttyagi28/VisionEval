"""Command-line entry point for VisionEval."""

from pathlib import Path

import typer

app = typer.Typer(
    name="visioneval",
    help="Evaluate vision models: Phase 1 classification CI, plus multimodal eval.",
    no_args_is_help=True,
)

traps_app = typer.Typer(
    name="traps",
    help="Living VLM hallucination traps (opt-in SQLite memory).",
    no_args_is_help=True,
)
app.add_typer(traps_app, name="traps")

DEFAULT_TRAPS_DB = Path("artifacts/traps.sqlite3")
DEFAULT_TRAPS_LOCK = Path("artifacts/baselines/traps.json")


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
    name = result["name"]
    total = len(result["samples"])
    typer.echo(f"Multimodal eval '{name}': {n_clean} clean sample-model pairs, {total} total rows.")
    traps = result.get("traps")
    if isinstance(traps, dict):
        typer.echo(f"Open traps: {traps.get('open', 0)}")


def _echo_traps(traps, heading: str) -> None:
    typer.echo(f"{heading}: {len(traps)}")
    for trap in traps:
        flag = "retired" if trap.retired else "open"
        typer.echo(
            f"  {trap.trap_id}  {trap.probe_type}  {trap.last_outcome}  "
            f"fails={trap.fail_count}  passes={trap.consecutive_passes}  [{flag}]"
        )


@traps_app.command("list")
def traps_list(
    db: Path = typer.Option(DEFAULT_TRAPS_DB, "--db", help="SQLite path for vlm_traps."),
    status: str = typer.Option("all", "--status", help="open | retired | all"),
) -> None:
    """List open and/or retired living traps."""
    from visioneval.traps.store import TrapStore

    store = TrapStore(db)
    status = status.lower().strip()
    if status not in {"open", "retired", "all"}:
        raise typer.BadParameter("status must be open, retired, or all")
    if status in {"open", "all"}:
        _echo_traps(store.list_traps(retired=False), "open")
    if status in {"retired", "all"}:
        _echo_traps(store.list_traps(retired=True), "retired")


@traps_app.command("harvest")
def traps_harvest(
    report: Path = typer.Argument(..., help="Multimodal JSON report path."),
    db: Path = typer.Option(DEFAULT_TRAPS_DB, "--db"),
    generate_hard_negatives: bool = typer.Option(False, "--generate-hard-negatives"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Harvest POPE / judge / caption failures from a multimodal JSON report."""
    from visioneval.traps.harvest import harvest_report
    from visioneval.traps.store import TrapStore

    store = TrapStore(db)
    summary = harvest_report(
        report, store, generate_hard_negatives=generate_hard_negatives, seed=seed
    )
    typer.echo(
        f"Harvested {summary.created} new, {summary.updated} updated, "
        f"{summary.hard_negatives} hard-negatives. Open: {store.count_open()}."
    )


@traps_app.command("run")
def traps_run(
    db: Path = typer.Option(DEFAULT_TRAPS_DB, "--db"),
    config: Path | None = typer.Option(None, "--config", help="Multimodal YAML for VLM + samples."),
    model_name: str | None = typer.Option(None, "--model", help="Model name from the YAML config."),
    budget: int = typer.Option(32, "--budget"),
    generate_hard_negatives: bool = typer.Option(False, "--generate-hard-negatives"),
    seed: int = typer.Option(0, "--seed"),
    retire_after: int = typer.Option(2, "--retire-after"),
    check_baseline: Path | None = typer.Option(None, "--check-baseline"),
) -> None:
    """Replay open traps against a Fake or configured VLM. Open traps consume budget first."""
    from visioneval.models.factory import build_model
    from visioneval.traps.runner import default_fake_for_traps, run_open_traps
    from visioneval.traps.store import TrapStore

    store = TrapStore(db)
    samples = None
    root = None
    if config is not None:
        from visioneval.multimodal.config import load_multimodal_config

        loaded = load_multimodal_config(config)
        root = str(config.parent)
        samples = {sample.id: sample for sample in loaded.samples}
        spec = loaded.models[0]
        if model_name:
            matched = [item for item in loaded.models if item.name == model_name]
            if not matched:
                raise typer.BadParameter(f"no model named {model_name!r} in {config}")
            spec = matched[0]
        payload = spec.to_factory_dict()
        if spec.kind == "fake":
            payload["object_map"] = {sample.id: list(sample.objects) for sample in loaded.samples}
        model = build_model(payload)
    else:
        model = default_fake_for_traps(store.list_open())
    result = run_open_traps(
        store,
        model,
        budget=budget,
        retire_after=retire_after,
        generate_hard_negatives=generate_hard_negatives,
        seed=seed,
        samples=samples,
        root=root,
        check_baseline=check_baseline,
    )
    typer.echo(
        f"Traps run: evaluated {result.evaluated}, passed {result.passed}, "
        f"failed {result.failed}, retired this run {result.retired}, still open {result.still_open}."
    )
    if result.regression is not None and result.regression.is_regression:
        reappeared = ",".join(result.regression.reappeared) or "-"
        worse = ",".join(result.regression.worse) or "-"
        typer.echo(f"Trap regression: reappeared {reappeared} worse {worse}")
        raise typer.Exit(code=1)


@traps_app.command("update-baseline")
def traps_update_baseline(
    db: Path = typer.Option(DEFAULT_TRAPS_DB, "--db"),
    lockfile: Path = typer.Option(DEFAULT_TRAPS_LOCK, "--lockfile"),
) -> None:
    """Lock current open-trap ids/outcomes as a git-trackable JSON file."""
    from visioneval.traps.baseline import save_trap_baseline
    from visioneval.traps.store import TrapStore

    payload = save_trap_baseline(lockfile, TrapStore(db))
    n_open = len(payload.get("open_traps") or {})
    n_retired = len(payload.get("retired_ids") or [])
    typer.echo(f"Wrote trap baseline {lockfile} ({n_open} open, {n_retired} retired).")


if __name__ == "__main__":
    app()
