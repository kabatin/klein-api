"""Run an engine as a child process in its own process group: stream its output, stop it
on cancel/timeout (SIGINT first, then SIGKILL), and leave a pid file so a restarted server
can stop a generation the previous server process left behind."""

from __future__ import annotations

import contextlib
import os
import pty
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

PID_FILE = "backend.pid"
# `ps -o lstart` is a fixed-width "Tue Oct  6 21:43:10 2026"
_LSTART_WIDTH = 24
PID_START_SLACK_S = 10.0
# After SIGKILL, how long to wait for a GPU process to finish tearing down before giving up.
KILL_WAIT_S = 120.0
LineCallback = Callable[[str], None]


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int | None
    duration_s: float
    canceled: bool
    timed_out: bool

    @property
    def still_running(self) -> bool:
        """The engine could not be stopped (its pid file is kept for restart recovery)."""
        return self.returncode is None


def child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment for tools we start: ours, minus klein-api's own settings (the token)."""
    base = {key: value for key, value in os.environ.items() if not key.startswith("KLEIN_API_")}
    return {**base, **(extra or {})}


def run_process(
    argv: Sequence[str],
    run_dir: Path,
    log_path: Path,
    *,
    timeout_s: float,
    cancel: threading.Event,
    on_line: LineCallback | None = None,
    stop_grace_s: float = 20.0,
    env: Mapping[str, str] | None = None,
    use_pty: bool = False,
) -> ProcessOutcome:
    """Run ``argv`` to completion (or until canceled / timed out), feeding each output line
    (stdout and stderr merged) to ``on_line`` and to ``log_path``.

    ``use_pty`` gives the child a pseudo-terminal as stdout: programs that block-buffer a
    pipe (draw-things-cli prints every progress line only at exit) then flush per line.
    Whatever goes wrong here after the child started (even Ctrl-C), the child is stopped.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    pid_file = run_dir / PID_FILE
    started = time.monotonic()
    master, slave = pty.openpty() if use_pty else (None, None)
    try:
        # line-buffered, so `tail -f` on the log shows a running job live
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            try:
                proc = subprocess.Popen(
                    list(argv),
                    cwd=run_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=slave if slave is not None else subprocess.PIPE,
                    stderr=slave if slave is not None else subprocess.STDOUT,
                    start_new_session=True,  # own process group: a stop reaches children too
                    env=child_env(env),
                )
            finally:
                if slave is not None:
                    os.close(slave)
            canceled, timed_out = _supervise(
                proc, pid_file, master, log, on_line, started, timeout_s, cancel, stop_grace_s
            )
    finally:
        if master is not None:
            os.close(master)
    return ProcessOutcome(
        returncode=proc.returncode,
        duration_s=time.monotonic() - started,
        canceled=canceled,
        timed_out=timed_out,
    )


def _supervise(
    proc: subprocess.Popen[bytes],
    pid_file: Path,
    master: int | None,
    log: TextIO,
    on_line: LineCallback | None,
    started: float,
    timeout_s: float,
    cancel: threading.Event,
    stop_grace_s: float,
) -> tuple[bool, bool]:
    reader: threading.Thread | None = None
    try:
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        source = master if master is not None else _fileno(proc)
        reader = threading.Thread(target=_pump, args=(source, log, on_line), daemon=True)
        reader.start()
        return _wait(proc, started, timeout_s, cancel, stop_grace_s)
    except BaseException:
        stop_group(proc, stop_grace_s)
        raise
    finally:
        if reader is not None:
            reader.join(timeout=10)
        if proc.poll() is not None:
            # keep the pid file of an engine we could not stop: a restart reaps it
            pid_file.unlink(missing_ok=True)


def _fileno(proc: subprocess.Popen[bytes]) -> int:
    if proc.stdout is None:  # pragma: no cover - stdout=PIPE always sets it
        raise RuntimeError("child has no stdout pipe")
    return proc.stdout.fileno()


def _chunks(fd: int) -> Iterator[bytes]:
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError:  # EIO: the pty's other end closed when the child exited
            return
        if not chunk:
            return
        yield chunk


def _pump(source: int, log: TextIO, on_line: LineCallback | None) -> None:
    """Forward complete lines; the log gets each line prefixed with seconds since start.

    Keeps draining even if logging or the callback fails: a reader that stopped would leave
    the engine blocked on a full pipe until the job timeout.
    """
    started = time.monotonic()
    pending = b""

    def emit(raw: bytes) -> None:
        line = raw.rstrip(b"\r").decode("utf-8", errors="replace")
        with contextlib.suppress(Exception):
            log.write(f"{time.monotonic() - started:9.2f} {line}\n")
        if on_line is not None:
            with contextlib.suppress(Exception):
                on_line(line)

    for chunk in _chunks(source):
        pending += chunk
        *lines, pending = pending.split(b"\n")
        for raw in lines:
            emit(raw)
    if pending:
        emit(pending)


def _wait(
    proc: subprocess.Popen[bytes],
    started: float,
    timeout_s: float,
    cancel: threading.Event,
    stop_grace_s: float,
) -> tuple[bool, bool]:
    while True:
        try:
            proc.wait(timeout=0.5)
            return False, False
        except subprocess.TimeoutExpired:
            pass
        if cancel.is_set():
            stop_group(proc, stop_grace_s)
            return True, False
        if time.monotonic() - started > timeout_s:
            stop_group(proc, stop_grace_s)
            return False, True


def stop_group(proc: subprocess.Popen[bytes], grace_s: float) -> None:
    """Ctrl-C first (lets the engine release the GPU cleanly), then SIGKILL the group.

    The group is SIGKILLed at the end in any case, so helpers the engine started cannot
    outlive it.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGINT)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace_s)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=KILL_WAIT_S)


def reap_orphan(run_dir: Path, expected_name: str) -> str | None:
    """Kill a process group left by a server that died mid-run, if it is still ours.

    Pids are reused, so the pid is only trusted when the live process is ``expected_name``
    and started when the pid file was written; anything else is left alone.
    """
    pid_file = run_dir / PID_FILE
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        written = pid_file.stat().st_mtime
    except (OSError, ValueError):
        return None
    finally:
        pid_file.unlink(missing_ok=True)
    if pid <= 1:
        return None
    info = process_info(pid)
    if info is None:
        return None
    command, started = info
    # the program is argv[0], or argv[1] for a script run through its interpreter
    names = [Path(token).name for token in command.split()[:2]]
    if expected_name not in names or abs(started - written) > PID_START_SLACK_S:
        return None
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return None
    return f"stopped leftover {expected_name} (pid {pid})"


def process_info(pid: int) -> tuple[str, float] | None:
    """``(command line, start time as epoch seconds)`` of a live process, else None."""
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=,command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = result.stdout.strip()
    if result.returncode != 0 or len(line) <= _LSTART_WIDTH:
        return None
    try:
        started = time.mktime(time.strptime(line[:_LSTART_WIDTH], "%a %b %d %H:%M:%S %Y"))
    except ValueError:
        return None
    return line[_LSTART_WIDTH:].strip(), started
