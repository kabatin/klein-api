"""One draw-things-cli job: ``count`` client processes, one after another, sharing a time
budget. Each run must prove success: exit 0, no ``Error:`` line and a fresh, non-empty file."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from klein_api.backends.base import (
    GenerationPlan,
    GenerationRequest,
    GenerationResult,
    ProgressCallback,
)
from klein_api.backends.drawthings_output import OutputWatch
from klein_api.backends.process import ProcessOutcome, run_process

if TYPE_CHECKING:
    from klein_api.backends.drawthings import DrawThingsCliBackend


def raw_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"raw-{index}.png"


class JobRun:
    """One job: ``count`` engine processes, one after another, sharing a time budget."""

    def __init__(
        self,
        backend: DrawThingsCliBackend,
        cli: str,
        request: GenerationRequest,
        plan: GenerationPlan,
        run_dir: Path,
        on_progress: ProgressCallback | None,
        remote_port: int | None = None,
    ) -> None:
        self._backend = backend
        self._remote_port = remote_port
        self._cli = cli
        self._request = request
        self._plan = plan
        self._run_dir = run_dir
        self._on_progress = on_progress
        self._sampling: list[float] = []

    def all(self, timeout_s: float, cancel: threading.Event) -> GenerationResult:
        started = time.monotonic()
        outputs: list[Path] = []
        for index in range(self._plan.count):
            remaining = timeout_s - (time.monotonic() - started)
            failed = self._one(index, max(remaining, 0.0), cancel)
            if isinstance(failed, GenerationResult):
                for path in outputs:
                    path.unlink(missing_ok=True)
                return replace(failed, duration_s=time.monotonic() - started)
            outputs.append(failed)
        return GenerationResult(
            plan=self._plan,
            duration_s=time.monotonic() - started,
            outputs=tuple(outputs),
            sampling_seconds=sum(self._sampling) if self._sampling else None,
            log_path=self._run_dir / f"backend-{self._plan.count - 1}.log",
        )

    def _report(self, index: int, fraction: float) -> None:
        if self._on_progress is not None:
            self._on_progress((index + fraction) / self._plan.count)

    def _one(
        self, index: int, timeout_s: float, cancel: threading.Event
    ) -> Path | GenerationResult:
        raw = raw_path(self._run_dir, index)
        raw.unlink(missing_ok=True)  # a stale file must never pass for a fresh one
        log_path = self._run_dir / f"backend-{index}.log"
        watch = OutputWatch(lambda fraction: self._report(index, fraction))
        argv = self._backend.command(
            self._cli, self._request, self._plan, self._run_dir, index, self._remote_port
        )
        try:
            outcome = run_process(
                argv,
                self._run_dir,
                log_path,
                timeout_s=timeout_s,
                cancel=cancel,
                on_line=watch.line,
                stop_grace_s=self._backend.options.stop_grace_s,
                # piped stdout is block-buffered: progress would only arrive at exit
                use_pty=True,
            )
        except (OSError, ValueError) as exc:  # e.g. missing binary, NUL byte in an argument
            return self._failed(0.0, log_path, failure=f"cannot start draw-things-cli: {exc}")
        if watch.sampling_seconds is not None:
            self._sampling.append(watch.sampling_seconds)
        return self._conclude(index, raw, outcome, watch, log_path)

    def _conclude(
        self, index: int, raw: Path, outcome: ProcessOutcome, watch: OutputWatch, log_path: Path
    ) -> Path | GenerationResult:
        """Exit 0 alone proves nothing: no ``Error:`` line and a fresh, non-empty file too."""
        if outcome.still_running:
            return self._failed(
                outcome.duration_s,
                log_path,
                failure="draw-things-cli did not exit after SIGKILL",
                failure_code="engine_unstoppable",
            )
        if outcome.canceled or outcome.timed_out:
            raw.unlink(missing_ok=True)  # possibly half-written
            return self._failed(
                outcome.duration_s,
                log_path,
                canceled=outcome.canceled,
                timed_out=outcome.timed_out,
            )
        if outcome.returncode != 0 or watch.error is not None:
            raw.unlink(missing_ok=True)
            reason = watch.error or f"draw-things-cli exited with code {outcome.returncode}"
            return self._failed(outcome.duration_s, log_path, failure=f"image {index}: {reason}")
        written = self._written(raw, watch)
        if written is None:
            return self._failed(
                outcome.duration_s,
                log_path,
                failure=f"image {index}: draw-things-cli wrote no image",
            )
        self._report(index, 1.0)
        return written

    def _written(self, raw: Path, watch: OutputWatch) -> Path | None:
        """The file at ``--output``; else the ``Wrote:`` path if it is inside the run dir (in
        case the CLI adjusts the name)."""
        candidates = [raw]
        if watch.wrote is not None:
            reported = (self._run_dir / watch.wrote).resolve()
            if reported.is_relative_to(self._run_dir.resolve()) and reported != raw.resolve():
                candidates.append(reported)
        for path in candidates:
            if path.is_file() and path.stat().st_size > 0:
                if path != raw:
                    path.replace(raw)
                return raw
        return None

    def _failed(
        self,
        duration_s: float,
        log_path: Path,
        *,
        failure: str | None = None,
        failure_code: str = "generation_failed",
        canceled: bool = False,
        timed_out: bool = False,
    ) -> GenerationResult:
        return GenerationResult(
            plan=self._plan,
            duration_s=duration_s,
            canceled=canceled,
            timed_out=timed_out,
            failure=failure,
            failure_code=failure_code,
            sampling_seconds=sum(self._sampling) if self._sampling else None,
            log_path=log_path,
        )
