"""Shared fixtures and helpers. Tests marked ``real`` run only with KLEIN_API_REAL_TESTS=1."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REAL_TESTS = os.environ.get("KLEIN_API_REAL_TESTS") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if REAL_TESTS:
        return
    skip = pytest.mark.skip(reason="real engine run: set KLEIN_API_REAL_TESTS=1")
    for item in items:
        if "real" in item.keywords:
            item.add_marker(skip)


def make_png(
    path: Path, size: tuple[int, int] = (320, 240), color: str = "red", mode: str = "RGB"
) -> bytes:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path, format="PNG")
    return path.read_bytes()


os.environ.setdefault("PYTHONUNBUFFERED", "1")


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fake machine for `klein-api setup` (see tests/setup_world.py)."""
    from tests.setup_world import World, use_small_assets

    use_small_assets(monkeypatch, tmp_path)
    return World(tmp_path)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """For tests that run `klein-api setup` through the CLI: a bug (or a check that does not
    stop it) must never write the real ~/.config, ~/.local or ~/Library."""
    home = tmp_path / "isolated-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KLEIN_API_CONFIG", str(home / ".config" / "klein-api" / "config.toml"))
    return home
