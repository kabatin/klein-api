"""A resident Draw Things gRPC server, so consecutive ``standard`` jobs skip the model load.

Measured on the target Mac (draw-things-cli v26.1011.0, 9B): a cold process per image takes
26.5 s; through a resident ``gRPCServerCLI-macOS`` started with a weights cache (``-w``) the next
images take 19.6 / 19.3 s (46.4 -> 38.8 s with one reference), pixel-identical. Without ``-w``
there is no gain. The server holds ~9.6 GB while the 9B is resident, so:

- it is started lazily by the first ``standard`` job and kept while such jobs keep coming;
- it is stopped after ``idle_seconds`` without a job, as soon as another GPU server on the Mac is
  seen busy while it idles (they do not count Draw Things processes), before a ``draft`` job
  (mixing models on one server made the next 9B job slower than a cold one), after a job that
  did not succeed (a canceled client may leave the server generating), and on shutdown;
- it always binds ``127.0.0.1`` (by default it would listen on every interface without auth);
- a port already answered by something else is never used (it is not our server).

The process is tracked by a pid file in ``state_dir`` so a restarted klein-api stops a server
left behind by a crashed one (pid + process name + start time must match).
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from klein_api.backends.process import PID_FILE, child_env, reap_orphan, stop_group
from klein_api.logs import event

log = logging.getLogger("klein_api.resident")

SERVER_NAME = "gRPCServerCLI-macOS"
BIND_ADDRESS = "127.0.0.1"
START_TIMEOUT_S = 120.0
PORT_CHECK_TIMEOUT_S = 1.0
LOG_FILE = "server.log"

PeerCheck = Callable[[], bool]


@dataclass(frozen=True)
class ResidentConfig:
    server_path: Path
    models_dir: Path
    port: int
    cache_gb: int
    idle_seconds: float
    poll_seconds: float
    state_dir: Path
    stop_grace_s: float = 15.0
    start_timeout_s: float = START_TIMEOUT_S


def server_command(config: ResidentConfig) -> list[str]:
    """The only way the server is started: loopback, no TLS (local only), weights cache."""
    return [
        str(config.server_path),
        "-a",
        BIND_ADDRESS,
        "-p",
        str(config.port),
        "--no-tls",
        "-w",
        str(config.cache_gb),
        str(config.models_dir),
    ]


def port_open(port: int, timeout_s: float = PORT_CHECK_TIMEOUT_S) -> bool:
    try:
        with socket.create_connection((BIND_ADDRESS, port), timeout=timeout_s):
            return True
    except OSError:
        return False


class ResidentServer:
    """Owns at most one server process. Thread-safe: the worker acquires/releases it, a
    monitor thread stops it when idle or when another GPU server needs the memory."""

    def __init__(self, config: ResidentConfig, *, clock: Callable[[], float] = time.monotonic):
        self.config = config
        self._clock = clock
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._in_use = False
        self._last_used = clock()
        self._peer_busy: PeerCheck | None = None
        self._closed = threading.Event()
        self._monitor: threading.Thread | None = None

    # -- state -------------------------------------------------------------------------
    def set_peer_check(self, check: PeerCheck | None) -> None:
        self._peer_busy = check

    @property
    def pid(self) -> int | None:
        with self._lock:
            return self._proc.pid if self._proc is not None else None

    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def healthy(self) -> bool:
        return self.running() and port_open(self.config.port)

    # -- use ---------------------------------------------------------------------------
    def acquire(self, job_id: str) -> bool:
        """Make sure a healthy server is up and mark it in use. One start attempt (a first
        start or a restart of a dead server); False means: run this job without it."""
        with self._lock:
            if self._closed.is_set():
                return False
            if not self.healthy():
                if self._proc is not None:
                    self._stop_locked("not answering", job_id=job_id)
                if not self._start_locked(job_id):
                    return False
            self._in_use = True
            return True

    def release(self, *, discard: bool = False, job_id: str | None = None) -> None:
        with self._lock:
            self._in_use = False
            self._last_used = self._clock()
            if discard:
                self._stop_locked("the job did not succeed", job_id=job_id)

    def stop(self, reason: str) -> None:
        with self._lock:
            self._stop_locked(reason)

    def shutdown(self) -> None:
        self._closed.set()
        self.stop("klein-api is stopping")

    def reap_leftover(self) -> str | None:
        """Stop a server a crashed klein-api left running (our pid file, our binary)."""
        return reap_orphan(self.config.state_dir, Path(self.config.server_path).name)

    # -- process -----------------------------------------------------------------------
    def _start_locked(self, job_id: str) -> bool:
        config = self.config
        if port_open(config.port):
            event(
                log,
                "resident.port_taken",
                level=logging.WARNING,
                job_id=job_id,
                port=config.port,
            )
            return False
        if not os.access(config.server_path, os.X_OK):
            event(log, "resident.missing", level=logging.WARNING, job_id=job_id)
            return False
        config.state_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            with (config.state_dir / LOG_FILE).open("ab") as server_log:
                proc = subprocess.Popen(
                    server_command(config),
                    stdin=subprocess.DEVNULL,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=child_env(),
                )
        except OSError as exc:
            event(
                log,
                "resident.start_failed",
                level=logging.WARNING,
                job_id=job_id,
                reason=type(exc).__name__,
            )
            return False
        (config.state_dir / PID_FILE).write_text(str(proc.pid), encoding="utf-8")
        self._proc = proc
        while time.monotonic() - started < config.start_timeout_s:
            if proc.poll() is not None:
                break
            if port_open(config.port, timeout_s=0.5):
                self._last_used = self._clock()
                event(
                    log,
                    "resident.started",
                    job_id=job_id,
                    pid=proc.pid,
                    port=config.port,
                    start_seconds=round(time.monotonic() - started, 3),
                )
                self._ensure_monitor()
                return True
            time.sleep(0.1)
        self._stop_locked("did not start listening", job_id=job_id, level=logging.WARNING)
        return False

    def _stop_locked(
        self, reason: str, *, job_id: str | None = None, level: int = logging.INFO
    ) -> None:
        proc, self._proc = self._proc, None
        self._in_use = False
        if proc is None:
            return
        stop_group(proc, self.config.stop_grace_s)
        if proc.poll() is not None:
            # keep the pid file of a server we could not stop: a restart reaps it
            (self.config.state_dir / PID_FILE).unlink(missing_ok=True)
        event(log, "resident.stopped", level=level, job_id=job_id, pid=proc.pid, reason=reason)

    # -- monitor -----------------------------------------------------------------------
    def _ensure_monitor(self) -> None:
        if self._monitor is None or not self._monitor.is_alive():
            self._monitor = threading.Thread(
                target=self._watch, name="klein-api-resident", daemon=True
            )
            self._monitor.start()

    def _watch(self) -> None:
        """While the server idles: stop it after ``idle_seconds``, or at once when another
        GPU server is busy. Never touches a server that is in use."""
        while not self._closed.wait(self.config.poll_seconds):
            with self._lock:
                if self._proc is None:
                    return
                if self._in_use:
                    continue
                if self._clock() - self._last_used >= self.config.idle_seconds:
                    self._stop_locked("idle")
                    return
            check = self._peer_busy
            busy = False
            if check is not None:
                with contextlib.suppress(Exception):  # a failing check must not kill this
                    busy = check()
            if busy:
                with self._lock:
                    if self._proc is not None and not self._in_use:
                        self._stop_locked("another GPU server is busy")
                        return
