"""The resident gRPC server: lazy start, reuse, idle stop, stop on a busy peer, stop before a
draft job, restart once then fall back, loopback-only bind, orphan cleanup — against a fake
``gRPCServerCLI-macOS`` (a TCP listener) and the fake ``draw-things-cli``."""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from klein_api.backends.base import GenerationRequest
from klein_api.backends.drawthings import DrawThingsCliBackend, DrawThingsOptions
from klein_api.backends.process import PID_FILE
from klein_api.backends.resident import (
    BIND_ADDRESS,
    SERVER_NAME,
    ResidentConfig,
    ResidentServer,
    port_open,
    server_command,
)
from klein_api.checks import Level
from klein_api.jobs.queue import JobQueue
from klein_api.jobs.store import JobStore
from klein_api.workflows.base import WorkflowRegistry
from tests.fakes import EchoWorkflow, FakeKleinBackend
from tests.test_drawthings_backend import FAKE_CLI, flags_of
from tests.test_drawthings_backend import models_dir as models_dir

FAKE_SERVER = textwrap.dedent(
    r"""
    import json, os, socket, sys
    args = sys.argv[1:]
    calls = os.environ.get("FAKE_GRPC_CALLS")
    if calls:
        with open(calls, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "args": args}) + "\n")
    if os.environ.get("FAKE_GRPC_MODE") == "exit":
        sys.exit(3)
    host, port = args[args.index("-a") + 1], int(args[args.index("-p") + 1])
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(16)
    while True:
        conn, _ = server.accept()
        conn.close()
    """
)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((BIND_ADDRESS, 0))
        return probe.getsockname()[1]


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "opt"
    script(directory / "draw-things-cli", FAKE_CLI)
    script(directory / SERVER_NAME, FAKE_SERVER)
    return directory


@pytest.fixture
def calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    cli_calls, server_calls = tmp_path / "cli.jsonl", tmp_path / "server.jsonl"
    monkeypatch.setenv("FAKE_DT_CALLS", str(cli_calls))
    monkeypatch.setenv("FAKE_GRPC_CALLS", str(server_calls))
    return cli_calls, server_calls


def lines(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def port() -> int:
    return free_port()


def make_backend(
    bin_dir: Path, models: Path, tmp_path: Path, port: int, **options
) -> DrawThingsCliBackend:
    return DrawThingsCliBackend(
        DrawThingsOptions(
            cli=str(bin_dir / "draw-things-cli"),
            models_dir=models,
            stop_grace_s=2,
            grpc_port=port,
            **options,
        ),
        state_dir=tmp_path / "resident",
        poll_seconds=0.05,
    )


@pytest.fixture
def backend(
    bin_dir: Path, models_dir: Path, tmp_path: Path, port: int
) -> Iterator[DrawThingsCliBackend]:
    backend = make_backend(bin_dir, models_dir, tmp_path, port)
    yield backend
    backend.shutdown()


def request(quality: str = "standard", count: int = 1, job: str = "job_X") -> GenerationRequest:
    return GenerationRequest(
        job_id=job, prompt="a tram", quality=quality, aspect_ratio=1.0, seed=3, count=count
    )  # type: ignore[arg-type]


def run(backend: DrawThingsCliBackend, tmp_path: Path, req: GenerationRequest, **kw):
    cancel = kw.pop("cancel", threading.Event())
    return backend.generate(req, tmp_path / req.job_id, timeout_s=60, cancel=cancel, **kw)


# -- bind address ------------------------------------------------------------------------


def test_the_server_only_ever_binds_loopback(
    backend: DrawThingsCliBackend, tmp_path: Path, calls: tuple[Path, Path], port: int
) -> None:
    assert BIND_ADDRESS == "127.0.0.1"
    assert backend.resident is not None
    argv = server_command(backend.resident.config)
    assert argv[1:3] == ["-a", "127.0.0.1"] and argv[3:5] == ["-p", str(port)]
    assert "--no-tls" in argv and argv[argv.index("-w") + 1] == "18"
    assert run(backend, tmp_path, request()).ok
    started = lines(calls[1])[0]["args"]
    assert started[started.index("-a") + 1] == "127.0.0.1"
    # no setting can change it
    for bad in ({"grpc_host": "0.0.0.0"}, {"grpc_address": "0.0.0.0"}):
        with pytest.raises(ValidationError):
            DrawThingsOptions.model_validate(bad)


# -- lifecycle -------------------------------------------------------------------------


def test_lazy_start_and_reuse_across_jobs(
    backend: DrawThingsCliBackend, tmp_path: Path, calls: tuple[Path, Path], port: int
) -> None:
    cli_calls, server_calls = calls
    assert backend.resident is not None and not backend.resident.running()
    assert run(backend, tmp_path, request("draft", job="job_A")).ok
    assert lines(server_calls) == []  # a draft job never starts it
    assert "--local" in lines(cli_calls)[0]
    assert run(backend, tmp_path, request(count=2, job="job_B")).ok
    pid = backend.resident.pid
    assert backend.resident.running() and pid is not None
    assert run(backend, tmp_path, request(job="job_C")).ok
    assert len(lines(server_calls)) == 1 and backend.resident.pid == pid  # reused
    remote = lines(cli_calls)[1:]
    assert len(remote) == 3
    for argv in remote:
        flags = flags_of(argv)
        assert "--remote" in argv and "--no-remote-tls" in argv
        assert (flags["--remote-url"], flags["--remote-port"]) == ("127.0.0.1", str(port))
        assert flags["--model"] == "flux_2_klein_9b_i8x.ckpt"
        assert not {"--local", "--offline", "--no-download-missing"} & set(argv)


def test_an_idle_server_is_stopped(
    bin_dir: Path, models_dir: Path, tmp_path: Path, port: int, calls: tuple[Path, Path]
) -> None:
    backend = make_backend(bin_dir, models_dir, tmp_path, port, resident_idle_seconds=1)
    try:
        assert run(backend, tmp_path, request()).ok
        assert backend.resident is not None and backend.resident.running()
        time.sleep(0.5)
        assert backend.resident.running()
        wait_for(lambda: not backend.resident.running(), timeout=5)  # type: ignore[union-attr]
        assert not port_open(port)
        assert not (tmp_path / "resident" / PID_FILE).exists()
    finally:
        backend.shutdown()


def test_a_busy_peer_stops_an_idle_server_but_not_one_in_use(
    backend: DrawThingsCliBackend, tmp_path: Path, calls: tuple[Path, Path]
) -> None:
    resident = backend.resident
    assert resident is not None
    busy = threading.Event()
    backend.set_peer_check(busy.is_set)
    assert resident.acquire("job_X")
    busy.set()
    time.sleep(0.3)
    assert resident.running()  # a running job is never interrupted
    resident.release()
    wait_for(lambda: not resident.running(), timeout=3)


def test_a_draft_job_stops_the_server_first(
    backend: DrawThingsCliBackend, tmp_path: Path, calls: tuple[Path, Path]
) -> None:
    assert run(backend, tmp_path, request(job="job_A")).ok
    assert backend.resident is not None and backend.resident.running()
    assert run(backend, tmp_path, request("draft", job="job_B")).ok
    assert not backend.resident.running()
    draft = lines(calls[0])[-1]
    assert "--remote" not in draft and flags_of(draft)["--model"] == "flux_2_klein_4b_i8x.ckpt"


def test_a_dead_server_is_restarted_once_then_jobs_fall_back(
    backend: DrawThingsCliBackend,
    tmp_path: Path,
    calls: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cli_calls, server_calls = calls
    resident = backend.resident
    assert resident is not None
    assert run(backend, tmp_path, request(job="job_A")).ok
    os.kill(resident.pid or 0, signal.SIGKILL)
    wait_for(lambda: not resident.running())
    assert run(backend, tmp_path, request(job="job_B")).ok  # restarted
    assert len(lines(server_calls)) == 2 and "--remote" in lines(cli_calls)[-1]
    os.kill(resident.pid or 0, signal.SIGKILL)
    wait_for(lambda: not resident.running())
    monkeypatch.setenv("FAKE_GRPC_MODE", "exit")  # the restart fails
    result = run(backend, tmp_path, request(job="job_C"))
    assert result.ok  # the job still runs, one process per image
    assert len(lines(server_calls)) == 3
    assert "--local" in lines(cli_calls)[-1] and "--remote" not in lines(cli_calls)[-1]
    assert "resident.fallback" in caplog.text


def test_a_foreign_listener_on_the_port_is_never_used(
    backend: DrawThingsCliBackend, tmp_path: Path, calls: tuple[Path, Path], port: int
) -> None:
    with socket.socket() as other:
        other.bind((BIND_ADDRESS, port))
        other.listen(4)
        assert run(backend, tmp_path, request()).ok
    assert lines(calls[1]) == []  # never started, never used
    assert "--remote" not in lines(calls[0])[0]


def test_canceling_a_remote_job_stops_the_server(
    backend: DrawThingsCliBackend,
    tmp_path: Path,
    calls: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_DT_MODE", "hang")
    cancel = threading.Event()
    threading.Timer(0.8, cancel.set).start()
    result = run(backend, tmp_path, request(), cancel=cancel)
    assert result.canceled
    assert "--remote" in lines(calls[0])[0]
    assert backend.resident is not None and not backend.resident.running()


def test_a_failed_remote_job_does_not_keep_the_server(
    backend: DrawThingsCliBackend,
    tmp_path: Path,
    calls: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_DT_MODE", "fail")
    assert not run(backend, tmp_path, request()).ok
    assert backend.resident is not None and not backend.resident.running()


def test_a_leftover_server_is_reaped_by_our_pid_file(
    bin_dir: Path, models_dir: Path, tmp_path: Path, port: int, calls: tuple[Path, Path]
) -> None:
    config = ResidentConfig(
        server_path=bin_dir / SERVER_NAME,
        models_dir=models_dir,
        port=port,
        cache_gb=18,
        idle_seconds=600,
        poll_seconds=60,
        state_dir=tmp_path / "resident",
    )
    crashed = ResidentServer(config)
    assert crashed.acquire("job_A")
    proc = crashed._proc
    assert proc is not None
    # a new klein-api finds the server the crashed one left running
    fresh = ResidentServer(config)
    assert fresh.reap_leftover() == f"stopped leftover {SERVER_NAME} (pid {proc.pid})"
    assert proc.wait(timeout=5) == -signal.SIGKILL
    assert fresh.reap_leftover() is None  # nothing left
    crashed._closed.set()


def test_the_backend_reaps_at_startup_and_stops_at_shutdown(
    backend: DrawThingsCliBackend, tmp_path: Path, calls: tuple[Path, Path]
) -> None:
    assert backend.startup() is None
    assert run(backend, tmp_path, request()).ok
    backend.shutdown()
    assert backend.resident is not None and not backend.resident.running()
    assert not backend.resident.acquire("job_after_shutdown")


def test_resident_off_or_without_state_dir(
    bin_dir: Path, models_dir: Path, tmp_path: Path, port: int, calls: tuple[Path, Path]
) -> None:
    off = make_backend(bin_dir, models_dir, tmp_path, port, resident=False)
    assert off.resident is None and off.startup() is None
    off.set_peer_check(lambda: True)
    off.shutdown()
    assert run(off, tmp_path, request()).ok
    assert lines(calls[1]) == [] and "--local" in lines(calls[0])[0]
    names = {c.name for c in off.health()}
    assert "resident gRPC server" not in names


def test_health_reports_the_server_binary(
    backend: DrawThingsCliBackend, bin_dir: Path, models_dir: Path, tmp_path: Path, port: int
) -> None:
    check = {c.name: c for c in backend.health()}["resident gRPC server"]
    assert check.level is Level.OK and "127.0.0.1" in check.detail
    missing = make_backend(
        bin_dir, models_dir, tmp_path, port, grpc_server_path=str(tmp_path / "none")
    )
    check = {c.name: c for c in missing.health()}["resident gRPC server"]
    assert check.level is Level.WARN and "one process per image" in check.detail
    assert backend.grpc_server_path() == bin_dir / SERVER_NAME


def test_the_queue_drives_startup_and_shutdown(tmp_path: Path) -> None:
    fake = FakeKleinBackend()
    q = JobQueue(
        JobStore(tmp_path),
        WorkflowRegistry([EchoWorkflow()]),
        fake,
        max_waiting=1,
        timeout_factor=2,
        retention_days=7,
    )
    q.start()
    q.stop(timeout_s=5)
    assert fake.lifecycle == ["startup", "shutdown"]
