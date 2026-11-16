# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def lake_url(monkeypatch):
    """Lake's address from the environment: the only place a turn's credential goes
    (agent-experiments#18). Tests that need it unset delete it."""
    monkeypatch.setenv("LAKE_URL", "https://lake.test")
