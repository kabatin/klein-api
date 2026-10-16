import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from klein_api.jobs.models import JobRecord, JobStatus
from klein_api.jobs.queue import (
    IdempotencyConflictError,
    JobConflictError,
    JobNotFoundError,
    JobQueue,
    QueueFullError,
    redact,
)
from klein_api.jobs.store import JobStore
from klein_api.workflows.base import InvalidParamsError, WorkflowRegistry
from tests.fakes import EchoParams, EchoWorkflow, FakeKleinBackend


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def make_queue(tmp_path: Path, backend: FakeKleinBackend, max_waiting: int = 1) -> JobQueue:
    return JobQueue(
        JobStore(tmp_path),
        WorkflowRegistry([EchoWorkflow()]),
        backend,
        max_waiting=max_waiting,
        timeout_factor=2,
        retention_days=7,
    )


@pytest.fixture
def backend() -> FakeKleinBackend:
    return FakeKleinBackend()


@pytest.fixture
def queue(tmp_path: Path, backend: FakeKleinBackend) -> Iterator[JobQueue]:
    q = make_queue(tmp_path, backend)
    q.start()
    yield q
    backend.release.set()
    q.stop(timeout_s=5)


def status(queue: JobQueue, job_id: str) -> JobStatus:
    return queue.view(job_id).record.status


def test_job_runs_to_success_with_timings(queue: JobQueue) -> None:
    record, created = queue.submit(EchoWorkflow(), EchoParams(text="hello"))
    assert created and record.status is JobStatus.QUEUED
    wait_for(lambda: status(queue, record.id) is JobStatus.SUCCEEDED)
    view = queue.view(record.id)
    assert view.progress == 1.0 and view.record.result == {"length": 5}
    assert queue.output_path(record.id).read_text() == "HELLO"
    timings = view.record.timings
    assert set(timings) == {
        "queue_seconds",
        "coordination_seconds",
        "backend_seconds",
        "sampling_seconds",
        "postprocess_seconds",
        "total_seconds",
    }
    assert view.record.estimate_seconds == 10.0
    assert not (queue._store.job_dir(record.id) / "raw-0.png").exists()


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("fail", "generation_failed"),
        ("timeout", "timeout"),
        ("crash", "internal"),
        ("no_output", "generation_failed"),
    ],
)
def test_failures_are_recorded(
    queue: JobQueue, backend: FakeKleinBackend, mode: str, code: str
) -> None:
    backend.mode = mode
    record, _ = queue.submit(EchoWorkflow(), EchoParams(text="x"))
    wait_for(lambda: status(queue, record.id) is JobStatus.FAILED)
    error = queue.view(record.id).record.error
    assert error is not None and error.code == code and error.retryable
    # paths and internals never reach the client-facing message
    assert "/Users/" not in error.message and str(Path.home()) not in error.message
    assert "exploded" not in error.message
    with pytest.raises(JobConflictError):
        queue.output_path(record.id)


def test_workflow_failure_keeps_its_code(queue: JobQueue) -> None:
    record, _ = queue.submit(EchoWorkflow(), EchoParams(text="x", fail_finalize=True))
    wait_for(lambda: status(queue, record.id) is JobStatus.FAILED)
    view = queue.view(record.id)
    error = view.record.error
    assert error is not None and (error.code, error.retryable) == ("postprocess_failed", False)
    assert "postprocess_seconds" in view.record.timings
    assert view.progress is None


def test_queued_and_running_are_distinct(queue: JobQueue, backend: FakeKleinBackend) -> None:
    backend.block = True
    first, _ = queue.submit(EchoWorkflow(), EchoParams(text="a"))
    assert backend.started.wait(2)
    second, _ = queue.submit(EchoWorkflow(), EchoParams(text="b"))
    running, waiting = queue.view(first.id), queue.view(second.id)
    assert running.record.status is JobStatus.RUNNING and running.queue_position is None
    assert running.progress == 0.5 and running.record.started_at is not None
    assert waiting.record.status is JobStatus.QUEUED and waiting.queue_position == 1
    assert waiting.progress is None and waiting.record.started_at is None
    assert queue.counts() == (1, 1)
    backend.release.set()
    wait_for(lambda: status(queue, second.id) is JobStatus.SUCCEEDED)


def test_capacity_refuses_with_retry_after(queue: JobQueue, backend: FakeKleinBackend) -> None:
    backend.block = True
    queue.submit(EchoWorkflow(), EchoParams(text="a"))
    assert backend.started.wait(2)
    queue.submit(EchoWorkflow(), EchoParams(text="b"))
    with pytest.raises(QueueFullError) as busy:
        queue.submit(EchoWorkflow(), EchoParams(text="c"))
    assert 30 <= busy.value.retry_after_s <= 600
    assert queue.refusal("echo", None) is not None


def test_zero_waiting_slots(tmp_path: Path, backend: FakeKleinBackend) -> None:
    q = make_queue(tmp_path, backend, max_waiting=0)
    q.start()
    try:
        backend.block = True
        q.submit(EchoWorkflow(), EchoParams(text="a"))
        assert backend.started.wait(2)
        with pytest.raises(QueueFullError):
            q.submit(EchoWorkflow(), EchoParams(text="b"))
    finally:
        backend.release.set()
        q.stop(timeout_s=5)


def test_rejected_inputs_release_the_slot(queue: JobQueue, tmp_path: Path) -> None:
    with pytest.raises(InvalidParamsError):
        queue.submit(EchoWorkflow(), EchoParams(text="x", reject=True))
    assert queue.counts() == (0, 0)
    assert not any((tmp_path / "jobs").iterdir())


def test_cancel_queued_running_and_finished(queue: JobQueue, backend: FakeKleinBackend) -> None:
    backend.block = True
    running, _ = queue.submit(EchoWorkflow(), EchoParams(text="a"))
    assert backend.started.wait(2)
    waiting, _ = queue.submit(EchoWorkflow(), EchoParams(text="b"))
    assert queue.cancel(waiting.id).status is JobStatus.CANCELED
    canceled = queue.cancel(running.id)
    assert canceled.status is JobStatus.CANCELED and canceled.cancel_requested
    with pytest.raises(JobConflictError):
        queue.cancel(running.id)
    with pytest.raises(JobNotFoundError):
        queue.cancel("job_missing")
    with pytest.raises(JobConflictError):
        queue.output_path(running.id)
    assert len(backend.calls) == 1  # the waiting job never reached the backend


def test_cancel_is_honoured_even_if_the_engine_finishes(
    queue: JobQueue, backend: FakeKleinBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine without a real stop still runs to the end: its output is not published."""
    monkeypatch.setattr("klein_api.jobs.queue.CANCEL_WAIT_S", 0.3)
    backend.mode = "ignore_cancel"
    backend.block = True
    record, _ = queue.submit(EchoWorkflow(), EchoParams(text="a"))
    assert backend.started.wait(2)
    pending = queue.cancel(record.id)
    assert pending.status is JobStatus.RUNNING and pending.cancel_requested
    backend.release.set()
    wait_for(lambda: status(queue, record.id) is JobStatus.CANCELED)
    assert not list(queue._store.job_dir(record.id).glob("output.*"))
    with pytest.raises(JobConflictError):
        queue.output_path(record.id)


def test_unknown_job(queue: JobQueue) -> None:
    with pytest.raises(JobNotFoundError):
        queue.view("job_nope")
    with pytest.raises(JobNotFoundError):
        queue.output_path("job_nope")


def test_missing_output_file_is_a_conflict(queue: JobQueue) -> None:
    record, _ = queue.submit(EchoWorkflow(), EchoParams(text="x"))
    wait_for(lambda: status(queue, record.id) is JobStatus.SUCCEEDED)
    queue.output_path(record.id).unlink()
    with pytest.raises(JobConflictError, match="gone"):
        queue.output_path(record.id)


def test_restart_requeues_and_fails_running(tmp_path: Path, backend: FakeKleinBackend) -> None:
    store = JobStore(tmp_path)
    queued = store.save(JobRecord(workflow="echo", params=EchoParams(text="q").model_dump()))
    lost = store.save(JobRecord(workflow="echo", params={}).started(10))
    stopping = store.save(JobRecord(workflow="echo", params={}).started(10).with_cancel_requested())
    orphan = store.save(JobRecord(workflow="gone", params={}))
    done = store.save(JobRecord(workflow="echo", params={}).canceled())
    q = make_queue(tmp_path, backend)
    q.start()
    try:
        wait_for(lambda: status(q, queued.id) is JobStatus.SUCCEEDED)
        wait_for(lambda: status(q, orphan.id) is JobStatus.FAILED)
        lost_view = q.view(lost.id).record
        assert lost_view.status is JobStatus.FAILED and lost_view.error is not None
        assert (lost_view.error.code, lost_view.error.retryable) == ("server_restarted", True)
        assert status(q, stopping.id) is JobStatus.CANCELED
        assert status(q, done.id) is JobStatus.CANCELED
        # the backend got the chance to stop what the dead server left running
        assert set(backend.recovered) == {store.job_dir(lost.id), store.job_dir(stopping.id)}
    finally:
        q.stop(timeout_s=5)


def test_idempotency_key_returns_the_same_job(queue: JobQueue, backend: FakeKleinBackend) -> None:
    backend.block = True
    first, created = queue.submit(EchoWorkflow(), EchoParams(text="a"), "gen-1")
    assert backend.started.wait(2)
    again, created_again = queue.submit(EchoWorkflow(), EchoParams(text="a"), "gen-1")
    assert created and not created_again and again.id == first.id
    with pytest.raises(IdempotencyConflictError, match="different params"):
        queue.submit(EchoWorkflow(), EchoParams(text="b"), "gen-1")
    # a known key is never refused as busy, a new one is once the queue is full
    queue.submit(EchoWorkflow(), EchoParams(text="w"), "gen-2")
    assert queue.refusal("echo", "gen-1") is None
    assert queue.refusal("echo", "gen-3") is not None
    assert len(backend.calls) == 1


def test_idempotency_index_survives_restart(tmp_path: Path, backend: FakeKleinBackend) -> None:
    store = JobStore(tmp_path)
    q1 = make_queue(tmp_path, backend)
    first, _ = q1.submit(EchoWorkflow(), EchoParams(text="a"), "k")
    store.save(first.canceled())  # finished, so the restarted queue does not run it
    q2 = make_queue(tmp_path, backend)
    q2.start()
    try:
        again, created = q2.submit(EchoWorkflow(), EchoParams(text="a"), "k")
        assert again.id == first.id and not created
        store.delete(first.id)  # pruned: the key is free again
        fresh, created = q2.submit(EchoWorkflow(), EchoParams(text="a"), "k")
        assert created and fresh.id != first.id
    finally:
        q2.stop(timeout_s=5)


def test_in_flight_key_conflicts(tmp_path: Path, backend: FakeKleinBackend) -> None:
    q = make_queue(tmp_path, backend)
    q._pending_keys.add(("echo", "busy-key"))
    with pytest.raises(IdempotencyConflictError, match="still being processed") as info:
        q.submit(EchoWorkflow(), EchoParams(text="a"), "busy-key")
    assert info.value.retryable and info.value.code == "idempotency_in_flight"


def test_redact_hides_home_directories() -> None:
    home = str(Path.home())
    assert redact(f"cannot open {home}/work/x.mp4") == "cannot open ~/work/x.mp4"
    assert redact("see /Users/alice/a and /home/bob/b") == "see ~/a and ~/b"
    assert redact("no paths here") == "no paths here"
