"""The backend contract: how a FLUX.2 [klein] engine is driven.

A backend receives a backend-neutral :class:`GenerationRequest` and returns raw images. It
chooses everything engine-specific — generation size, model file, steps, guidance — from the
request's quality tier. It never sees HTTP, client schemas or client-chosen paths, and its own
output (logs, error text) never reaches clients unredacted. Swapping the backend must not
change the HTTP contract.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from klein_api.checks import Check

Quality = Literal["draft", "standard"]

# Backend-reported progress over the whole job, 0.0..1.0.
ProgressCallback = Callable[[float], None]


@dataclass(frozen=True)
class GenerationRequest:
    job_id: str
    prompt: str
    quality: Quality
    # width / height the client asked for; the backend picks a close size it supports
    aspect_ratio: float
    # image i uses seed + i
    seed: int
    count: int
    # upright RGB PNGs inside the job directory; the first is the main reference
    references: tuple[Path, ...] = ()


@dataclass(frozen=True)
class GenerationPlan:
    width: int
    height: int
    count: int
    estimate_seconds: float


@dataclass(frozen=True)
class GenerationResult:
    plan: GenerationPlan
    duration_s: float
    # one fresh, non-empty file per image, in order; complete only when every image succeeded
    outputs: tuple[Path, ...] = ()
    canceled: bool = False
    timed_out: bool = False
    # operator-facing reason (may contain paths: the queue redacts it before storing)
    failure: str | None = None
    failure_code: str = "generation_failed"
    # engine-reported time spent denoising (summed over images), when reported
    sampling_seconds: float | None = None
    log_path: Path | None = None

    @property
    def ok(self) -> bool:
        return (
            len(self.outputs) == self.plan.count
            and not self.canceled
            and not self.timed_out
            and self.failure is None
        )

    def describe_failure(self) -> str:
        if self.canceled:
            return "canceled"
        if self.timed_out:
            return f"timed out after {self.duration_s:.0f}s"
        return self.failure or "no output written"


class KleinBackend(ABC):
    name: ClassVar[str]
    # True when ``cancel`` really stops the engine (frees the GPU), not just the bookkeeping.
    supports_cancel: ClassVar[bool]

    @property
    def max_reference_images(self) -> int:
        """How many reference images the engine honours (0: references are refused)."""
        return 0

    @abstractmethod
    def typical_seconds(self, quality: Quality) -> float:
        """Expected seconds per image (a client hint, also the basis of estimates)."""

    @abstractmethod
    def plan(self, request: GenerationRequest) -> GenerationPlan:
        """Generation size for a request, and how long the whole job should take."""

    @abstractmethod
    def health(self) -> list[Check]:
        """Can this backend generate right now (engine installed, model files present)?"""

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        run_dir: Path,
        *,
        timeout_s: float,
        cancel: threading.Event,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Run one job (all images). Must return (not raise) for engine failures, cancel,
        timeout."""

    def recover(self, run_dir: Path) -> str | None:
        """Clean up after a run the server lost track of (restart). Returns what was done."""
        return None

    def startup(self) -> str | None:
        """Called once when the queue starts: clean up what a crashed server left (beyond
        single runs). Returns what was done."""
        return None

    def shutdown(self) -> None:
        """Called when the queue stops: release long-lived engine resources."""
        return None

    def set_peer_check(self, check: Callable[[], bool] | None) -> None:
        """``check()`` is True while another GPU server on the machine is busy (backends that
        hold GPU memory between jobs give it back then)."""
        return None
