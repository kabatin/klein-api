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
