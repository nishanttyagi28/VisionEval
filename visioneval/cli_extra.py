"""Extra CLI commands: maps, traps gate, and TruthGraph verify (imported once from cli.py)."""

from __future__ import annotations

from pathlib import Path

import typer

from visioneval.cli import DEFAULT_TRAPS_DB, DEFAULT_TRAPS_LOCK, app, traps_app


@app.command("maps")
def maps(
    report: Path | None = typer.Argument(
        None,
        help="Multimodal JSON report. Optional when --db is set.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of human text."),
    traps_db: Path | None = typer.Option(
        None,
        "--db",
        help="Living-traps SQLite. Open traps are included in the map.",
    ),
) -> None:
    """Black-box hallucination map: where the model consistently fails. CPU-only."""
    from visioneval.maps.hallucination import build_map, format_human, to_json

    if report is None and traps_db is None:
        raise typer.BadParameter("provide a REPORT.json and/or --db traps.sqlite3")
    if report is not None and not report.is_file():
        raise typer.BadParameter(f"report not found: {report}")
    hallucination_map = build_map(report, traps_db=traps_db)
    typer.echo(to_json(hallucination_map) if json_output else format_human(hallucination_map), nl=False)


@traps_app.command("gate")
def traps_gate(
    db: Path = typer.Option(DEFAULT_TRAPS_DB, "--db"),
    lockfile: Path = typer.Option(DEFAULT_TRAPS_LOCK, "--lockfile", help="Git-trackable traps lockfile."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-actionable JSON."),
) -> None:
    """CI release blocker: compare living traps DB to a lockfile. Exit 1 on regression."""
    from visioneval.traps.baseline import (
        compare_trap_baseline,
        format_trap_regression,
        load_trap_baseline,
        trap_regression_json,
    )
    from visioneval.traps.store import TrapStore

    if not lockfile.is_file():
        raise typer.BadParameter(f"lockfile not found: {lockfile}")
    regression = compare_trap_baseline(load_trap_baseline(lockfile), TrapStore(db))
    typer.echo(
        trap_regression_json(regression) if json_output else format_trap_regression(regression),
        nl=False,
    )
    if regression.is_regression:
        raise typer.Exit(code=1)


@app.command("verify")
def verify(
    report_or_suite: Path = typer.Argument(
        ...,
        help="Multimodal JSON report or YAML/JSON verify case suite.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON dossier instead of human text."),
    include_corrupted: bool = typer.Option(
        False,
        "--include-corrupted",
        help="Also verify corrupted/noise-sweep rows (default: clean rows only).",
    ),
) -> None:
    """TruthGraph-style claim verification against visual ground-truth evidence. CPU-only."""
    from visioneval.verify.adapter import build_dossier, format_human, to_json

    if not report_or_suite.is_file():
        raise typer.BadParameter(f"input not found: {report_or_suite}")
    dossier = build_dossier(report_or_suite, skip_corrupted=not include_corrupted)
    typer.echo(to_json(dossier) if json_output else format_human(dossier), nl=False)


# Alias matching the TruthGraph product name.
app.command("truth")(verify)
