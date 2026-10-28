"""Shared by the setup modules."""

from __future__ import annotations

from collections.abc import Callable

Emit = Callable[[str], None]


class SetupError(RuntimeError):
    """A setup step cannot continue; the message tells the user what to do."""
