"""The version in pyproject.toml has release notes, indexed in docs/releases/README.md."""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASES = ROOT / "docs" / "releases"


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def test_installed_package_reports_the_project_version() -> None:
    assert version("agent-experiments") == project_version()


def test_release_notes_exist_for_the_project_version() -> None:
    current = project_version()
    notes = RELEASES / f"v{current}.md"
    assert notes.is_file(), f"{notes} is missing"
    heading = next(
        line for line in notes.read_text().splitlines() if line.startswith("# ")
    )
    assert f"v{current}" in heading
    assert f"](v{current}.md)" in (RELEASES / "README.md").read_text()
