"""Model files: ``draw-things-cli models ensure --model FILE --models-dir DIR``.

``models ensure`` (draw-things-cli v26.1011.0, ``Apps/DrawThingsCLI/DrawThingsCLI.swift``)
resolves the model, adds its registered dependencies (text encoder, VAE) unless
``--no-include-dependencies``, and downloads what is missing with its resumable downloader:
each file goes to ``FILE.partial`` (preallocated to full size) with a ``.partial.map`` of
finished segments, and is checked against its SHA-256 before it is renamed.

Because the partial file is preallocated, a growing directory cannot tell a live download
from a stalled one; the CLI's ``\\r`` progress output can. When it stays silent for
``stall_s``, the CLI is stopped (SIGINT first: the partial file and its map are kept) and
started again, continuing where it left off.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from klein_api.backends.process import run_process
from klein_api.setup.errors import Emit, SetupError
from klein_api.setup.pins import Component

STALL_S = 600.0
POLL_S = 5.0
MAX_RESTARTS = 30
ENSURE_TIMEOUT_S = 12 * 3600.0

# (argv, log file, cancel, on_line, on_activity) -> exit code (None: could not be stopped)
EnsureRun = Callable[
    [Sequence[str], Path, threading.Event, Callable[[str], None], Callable[[], None]],
    int | None,
]


def run_ensure(
    argv: Sequence[str],
    log_path: Path,
    cancel: threading.Event,
    on_line: Callable[[str], None],
    on_activity: Callable[[], None],
) -> int | None:
    # a pty: the CLI block-buffers a pipe and would show no progress until it exits
    outcome = run_process(
        argv,
        log_path.parent,
        log_path,
        timeout_s=ENSURE_TIMEOUT_S,
        cancel=cancel,
        on_line=on_line,
        on_output=on_activity,
        use_pty=True,
    )
    return outcome.returncode


def ensure_argv(cli: Path, component: Component, models_dir: Path) -> list[str]:
    return [
        str(cli),
        "models",
        "ensure",
        "--model",
        component.model_file,
        "--models-dir",
        str(models_dir),
    ]


def missing_files(component: Component, models_dir: Path) -> list[str]:
    return [name for name in component.files if not (models_dir / name).is_file()]


class ModelEnsurer:
    def __init__(
        self,
        cli: Path,
        models_dir: Path,
        work_dir: Path,
        *,
        run: EnsureRun = run_ensure,
        emit: Emit = lambda _: None,
        stall_s: float = STALL_S,
        poll_s: float = POLL_S,
        max_restarts: int = MAX_RESTARTS,
    ) -> None:
        self._cli = cli
        self._models_dir = models_dir
        self._work_dir = work_dir
        self._run = run
        self._emit = emit
        self._stall_s = stall_s
        self._poll_s = poll_s
        self._max_restarts = max_restarts

    def ensure(self, component: Component) -> None:
        argv = ensure_argv(self._cli, component, self._models_dir)
        for attempt in range(self._max_restarts + 1):
            code, stalled = self._run_once(component, argv)
            if not stalled and code == 0:
                break
            reason = "no progress" if stalled else f"draw-things-cli exited with {code}"
            if attempt == self._max_restarts:
                raise SetupError(
                    f"{component.id}: download did not finish after {self._max_restarts} "
                    f"restarts ({reason}); log: {self._log(component)}"
                )
            self._emit(f"{component.id}: {reason}; resuming ({attempt + 1}/{self._max_restarts})")
        missing = missing_files(component, self._models_dir)
        if missing:
            raise SetupError(
                f"{component.id}: still missing after `models ensure`: {', '.join(missing)} "
                f"(log: {self._log(component)})"
            )

    def _log(self, component: Component) -> Path:
        return self._work_dir / component.id / "ensure.log"

    def _run_once(self, component: Component, argv: list[str]) -> tuple[int | None, bool]:
        cancel = threading.Event()
        box: dict[str, int | None] = {}
        last = [time.monotonic()]
        log = self._log(component)
        log.parent.mkdir(parents=True, exist_ok=True)

        def touch() -> None:
            last[0] = time.monotonic()

        def work() -> None:
            box["code"] = self._run(argv, log, cancel, self._emit, touch)

        self._emit(f"{component.id}: draw-things-cli models ensure --model {component.model_file}")
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        stalled = False
        while thread.is_alive():
            thread.join(timeout=self._poll_s)
            if thread.is_alive() and time.monotonic() - last[0] >= self._stall_s:
                stalled = True
                cancel.set()
                thread.join()
        return box.get("code"), stalled
