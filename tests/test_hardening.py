"""Regression tests for failure modes found in review: a worker that must not die, engines
that must not outlive their job, request framing, shutdown semantics, disk space."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from klein_api.api.app import create_app
from klein_api.api.security import host_name
from klein_api.backends import process
from klein_api.backends.process import PID_FILE, ProcessOutcome, child_env, run_process
from klein_api.jobs import queue as queue_module
from klein_api.jobs.models import JobStatus
from klein_api.jobs.queue import InsufficientStorageError, JobQueue
from klein_api.jobs.store import JobStore
from klein_api.workflows.base import WorkflowRegistry
from tests.fakes import EchoParams, EchoWorkflow, FakeKleinBackend


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


@pytest.fixture
def backend() -> FakeKleinBackend:
    return FakeKleinBackend()


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path)


@pytest.fixture
def queue(store: JobStore, backend: FakeKleinBackend) -> Iterator[JobQueue]:
    q = JobQueue(
        store,
        WorkflowRegistry([EchoWorkflow()]),
        backend,
        max_waiting=1,
        timeout_factor=2,
        retention_days=7,
    )
    q.start()
    yield q
    backend.release.set()
    q.stop(timeout_s=5)


# -- the worker survives everything ------------------------------------------------------


def test_worker_survives_a_failing_save(
    queue: JobQueue, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disk full while recording a job's end must not kill the worker (later jobs run)."""
    monkeypatch.setattr(queue_module, "SAVE_ATTEMPTS", 2)
    real_save = store.save

    def failing_save(record):
        if record.status.finished:
            raise OSError(28, "No space left on device")
        return real_save(record)

    monkeypatch.setattr(store, "save", failing_save)
    first, _ = queue.submit(EchoWorkflow(), EchoParams(text="a"))
    # the end could not be written, but the job is not shown running forever
    wait_for(lambda: queue.view(first.id).record.status.finished, timeout=10)
    assert queue.view(first.id).record.status is JobStatus.SUCCEEDED
    monkeypatch.setattr(store, "save", real_save)
    second, _ = queue.submit(EchoWorkflow(), EchoParams(text="b"))
    wait_for(lambda: queue.view(second.id).record.status is JobStatus.SUCCEEDED)
    assert queue.worker_alive


def test_health_reports_a_dead_worker(tmp_path: Path, backend: FakeKleinBackend) -> None:
    q = JobQueue(
        JobStore(tmp_path),
        WorkflowRegistry([EchoWorkflow()]),
        backend,
        max_waiting=1,
        timeout_factor=2,
        retention_days=7,
    )
    app = create_app(q, WorkflowRegistry([EchoWorkflow()]), token=None, max_body_bytes=1 << 20)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/healthz").status_code == 200
        q.stop(timeout_s=5)
        dead = client.get("/v1/health")
        assert dead.status_code == 503 and dead.json()["status"] == "degraded"


def test_shutdown_fails_the_running_job_as_retryable(
    queue: JobQueue, backend: FakeKleinBackend
) -> None:
    """A routine restart is not a client's cancel: report it like a crash would be."""
    backend.block = True
    record, _ = queue.submit(EchoWorkflow(), EchoParams(text="a"))
    assert backend.started.wait(2)
    queue.stop(timeout_s=5)
    stopped = queue.view(record.id).record
    assert stopped.status is JobStatus.FAILED and stopped.error is not None
    assert (stopped.error.code, stopped.error.retryable) == ("server_stopped", True)


def test_pruning_frees_idempotency_keys(queue: JobQueue, store: JobStore) -> None:
    record, _ = queue.submit(EchoWorkflow(), EchoParams(text="a"), "key-1")
    wait_for(lambda: queue.view(record.id).record.status.finished)
    store.delete(record.id)
    queue._store.prune = lambda *_a, **_k: [record.id]  # type: ignore[method-assign]
    queue._prune()
    assert ("echo", "key-1") not in queue._keys


def test_low_disk_space_refuses_new_jobs(
    queue: JobQueue, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "free_bytes", lambda: 10)
    with pytest.raises(InsufficientStorageError):
        queue.submit(EchoWorkflow(), EchoParams(text="a"))
    assert queue.counts() == (0, 0)
    app = create_app(queue, WorkflowRegistry([EchoWorkflow()]), token=None, max_body_bytes=1 << 20)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        full = client.post("/v1/workflows/echo/jobs", json={"text": "x"})
        assert full.status_code == 503
        assert full.json()["error"]["code"] == "insufficient_storage"


# -- engines never outlive their job ---------------------------------------------------------


def test_run_process_stops_the_child_when_supervision_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(process, "_wait", interrupted)
    seen: list[int] = []
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        seen.append(proc.pid)
        return proc

    monkeypatch.setattr(subprocess, "Popen", spy)
    with pytest.raises(KeyboardInterrupt):
        run_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            tmp_path / "log",
            timeout_s=60,
            cancel=threading.Event(),
            stop_grace_s=1,
        )
    with pytest.raises(ProcessLookupError):
        os.kill(seen[0], 0)  # stopped and reaped
    assert not (tmp_path / PID_FILE).exists()


def test_process_outcome_still_running() -> None:
    stuck = ProcessOutcome(returncode=None, duration_s=1.0, canceled=True, timed_out=False)
    assert stuck.still_running
    assert not ProcessOutcome(0, 1.0, canceled=False, timed_out=False).still_running


def test_child_environment_drops_klein_api_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KLEIN_API_TOKEN", "t" * 40)
    monkeypatch.setenv("HOME_TEST_VALUE", "kept")
    env = child_env({"EXTRA": "1"})
    assert "KLEIN_API_TOKEN" not in env and env["HOME_TEST_VALUE"] == "kept"
    assert env["EXTRA"] == "1"


# -- request framing and hostile inputs ----------------------------------------------------


@pytest.fixture
def client(queue: JobQueue) -> Iterator[TestClient]:
    app = create_app(queue, WorkflowRegistry([EchoWorkflow()]), token=None, max_body_bytes=1 << 20)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c


def test_transfer_encoding_is_refused_even_with_a_length(client: TestClient) -> None:
    sent = client.post(
        "/v1/workflows/echo/jobs",
        content=b'{"text":"x"}',
        headers={
            "Content-Type": "application/json",
            "Content-Length": "12",
            "Transfer-Encoding": "chunked",
        },
    )
    assert sent.status_code == 411 and sent.json()["error"]["code"] == "length_required"


@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "text/plain; charset=application/json", "application/x-www-form-urlencoded"],
)
def test_cross_site_simple_posts_are_not_parsed(client: TestClient, content_type: str) -> None:
    sent = client.post(
        "/v1/workflows/echo/jobs", content=b'{"text":"x"}', headers={"Content-Type": content_type}
    )
    assert sent.status_code == 422


@pytest.mark.parametrize("header", ["Via", "X-Forwarded-Proto", "CF-Connecting-IP"])
def test_more_proxy_headers_need_a_token(client: TestClient, header: str) -> None:
    assert client.get("/healthz", headers={header: "1.1 proxy"}).status_code == 403


def test_bracketed_host_must_be_exact() -> None:
    assert host_name("[::1]:8767") == "::1"
    assert host_name("[::1]") == "::1"
    assert host_name("[::1]evil.example") == "[::1]evil.example"
    assert host_name("[::1]:80x") == "[::1]:80x"


@pytest.mark.parametrize("prompt", ["a\x00b", "bell\x07", "esc\x1b[31m", "del\x7f"])
def test_control_characters_in_prompts_are_rejected(prompt: str) -> None:
    from pydantic import ValidationError

    from klein_api.workflows.flux2_klein import Flux2KleinParams

    base = {"output": {"width": 1024, "height": 1024}}
    with pytest.raises(ValidationError, match="control characters"):
        Flux2KleinParams.model_validate(base | {"prompt": prompt})
    ok = Flux2KleinParams.model_validate(base | {"prompt": "line one\n\tline two"})
    assert ok.prompt == "line one\n\tline two"
