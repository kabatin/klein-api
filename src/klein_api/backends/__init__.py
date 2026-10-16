"""Backends: the engines that actually run FLUX.2 [klein]."""

from __future__ import annotations

from klein_api.backends.base import KleinBackend
from klein_api.settings import Settings


def build_backend(settings: Settings) -> KleinBackend:
    from klein_api.backends.drawthings import DrawThingsCliBackend

    return DrawThingsCliBackend.from_settings(settings)


__all__ = ["KleinBackend", "build_backend"]
