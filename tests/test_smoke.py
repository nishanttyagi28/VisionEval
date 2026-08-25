"""Smoke tests for the public package and CLI scaffold."""

from typer.testing import CliRunner

from visioneval import __version__
from visioneval.cli import app


def test_package_exposes_version() -> None:
    """Keep the package importable for downstream and CI users."""
    assert __version__ == "0.1.0"


def test_cli_reports_scaffold_status() -> None:
    """Keep the initial command-line entry point executable."""
    result = CliRunner().invoke(app, ["run"])

    assert result.exit_code == 0
    assert "scaffold is ready" in result.stdout
