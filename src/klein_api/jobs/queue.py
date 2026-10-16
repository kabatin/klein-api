"""One GPU slot, FIFO. ``1 running + max_waiting queued``; beyond that submissions are refused
with :class:`QueueFullError` so clients can back off instead of piling up work.

The job at the head of the queue starts only when the other GPU servers on the Mac are idle
(:mod:`klein_api.jobs.coordination`); until then it stays ``queued`` with ``waiting_for``."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from klein_api.backends.base import GenerationResult, KleinBackend
from klein_api.jobs.coordination import Coordinator
from klein_api.jobs.models import JobRecord, JobStatus
from klein_api.jobs.store import JobStore
from klein_api.logs import event
from klein_api.workflows.base import Workflow, WorkflowFailedError, WorkflowRegistry

log = logging.getLogger("klein_api.queue")

TIMEOUT_MARGIN_S = 300.0
PRUNE_INTERVAL_S = 3600.0
RETRY_AFTER_MIN_S = 30
RETRY_AFTER_MAX_S = 600
CANCEL_WAIT_S = 30.0
# A submit needs at least this much free disk in data_dir (raw images + outputs + inputs).
MIN_FREE_BYTES = 2 * 1024**3
# Saving a finished job's record is retried this many times before giving up.
SAVE_ATTEMPTS = 3
# Intermediates removed when a job ends (the deliverables are output-<i>.png).
SCRATCH_FILES = ("prompt.txt", *(f"raw-{i}.png" for i in range(8)))


class QueueFullError(Exception):
    def __init__(self, retry_after_s: int) -> None:
        super().__init__("the GPU slot and the waiting queue are full")
        self.retry_after_s = retry_after_s


class IdempotencyConflictError(Exception):
    """The Idempotency-Key was already used with different params (a client bug)."""

    code = "idempotency_conflict"
    retryable = False


class IdempotencyInFlightError(IdempotencyConflictError):
    """Another request with this key is being accepted right now: retry shortly."""

    code = "idempotency_in_flight"
    retryable = True


class InsufficientStorageError(Exception):
    """data_dir is nearly full: generating now would fail half-way."""


class JobNotFoundError(KeyError):
    pass


class JobConflictError(Exception):
    pass


class OutputIndexError(LookupError):
    """The job succeeded but has no output with this index."""


@dataclass(frozen=True)
class JobView:
    record: JobRecord
    progress: float | None
    queue_position: int | None
    waiting_for: str | None = None


@dataclass
class _Running:
    job_id: str
    started: float
    estimate_s: float
    cancel: threading.Event
    coordination_s: float = 0.0


_USER_DIR = re.compile(r"/(?:Users|home)/[^/\s'\"]+")


def redact(message: str) -> str:
    """Hide the server's home directory in messages that reach clients."""
    return _USER_DIR.sub("~", message.replace(str(Path.home()), "~"))


def fingerprint(workflow_id: str, params: BaseModel) -> str:
    digest = hashlib.sha256(workflow_id.encode())
    digest.update(params.model_dump_json().encode())
    return digest.hexdigest()


class JobQueue:
    def __init__(
        self,
        store: JobStore,
        registry: WorkflowRegistry,
        backend: KleinBackend,
        *,
        max_waiting: int,
        timeout_factor: float,
        retention_days: int,
        coordinator: Coordinator | None = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._backend = backend
        self._max_waiting = max_waiting
        self._timeout_factor = timeout_factor
        self._retention_days = retention_days
        self._coordinator = coordinator if coordinator is not None and coordinator.peers else None
        self._cond = threading.Condition()
        self._waiting: deque[str] = deque()
        self._reserved = 0
        self._running: _Running | None = None
        self._progress: dict[str, float] = {}
        # head-of-queue job id -> the GPU server it is waiting for
        self._waiting_for: dict[str, str] = {}
        # (workflow id, Idempotency-Key) -> job id, and keys whose submit is in flight
        self._keys: dict[tuple[str, str], str] = {}
        self._pending_keys: set[tuple[str, str]] = set()
        self._stopping = False
        self._thread: threading.Thread | None = None
        # finished records that could not be written to disk (e.g. disk full); served from
        # memory so a job is never shown as running forever
        self._unsaved: dict[str, JobRecord] = {}

    # -- lifecycle -----------------------------------------------------------------
    def start(self) -> None:
        requeue, interrupted = self._store.recover()
        for record in interrupted:
            cleanup = self._backend.recover(self._store.job_dir(record.id))
            event(
                log,
                "job.interrupted",
                level=logging.WARNING,
                job_id=record.id,
                workflow=record.workflow,
                state=record.status.value,
                failure_category=record.error.code if record.error else None,
                cleanup=cleanup,
            )
            self._store.discard_inputs(record.id, *SCRATCH_FILES)
        with self._cond:
            self._waiting.extend(record.id for record in requeue)
            self._keys = {
                (record.workflow, record.idempotency_key): record.id
                for record in self._store.all()
                if record.idempotency_key is not None
            }
        self._prune()
        self._thread = threading.Thread(target=self._loop, name="klein-api-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 60.0) -> None:
        with self._cond:
            self._stopping = True
            if self._running is not None:
                self._running.cancel.set()
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    # -- API-facing ----------------------------------------------------------------
    @property
    def max_waiting(self) -> int:
        return self._max_waiting

    @property
    def worker_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def counts(self) -> tuple[int, int]:
        with self._cond:
            return (1 if self._running else 0), len(self._waiting)

    def refusal(self, workflow_id: str, idempotency_key: str | None) -> int | None:
        """``Retry-After`` seconds if a new submit would be refused now, else ``None``.

        Lets the HTTP layer answer 429 before reading a (possibly 30 MB) body. A key that
        already has a job is never refused: resubmitting it just returns that job.
        """
        with self._cond:
            if idempotency_key is not None and (workflow_id, idempotency_key) in self._keys:
                return None
            if self._occupied_locked() >= 1 + self._max_waiting:
                return self._retry_after_locked()
            return None

    def submit(
        self, workflow: Workflow, params: BaseModel, idempotency_key: str | None = None
    ) -> tuple[JobRecord, bool]:
        """Queue a job. Returns ``(record, created)``; ``created`` is False when an
        ``idempotency_key`` resubmission returns the job it already made."""
        key = None if idempotency_key is None else (workflow.id, idempotency_key)
        params_hash = None if key is None else fingerprint(workflow.id, params)
        with self._cond:
            if key is not None:
                existing = self._existing_locked(key, params_hash)
                if existing is not None:
                    return existing, False
                self._pending_keys.add(key)
            if self._occupied_locked() >= 1 + self._max_waiting:
                if key is not None:
                    self._pending_keys.discard(key)
                raise QueueFullError(self._retry_after_locked())
            if self._store.free_bytes() < MIN_FREE_BYTES:
                if key is not None:
                    self._pending_keys.discard(key)
                raise InsufficientStorageError("not enough free disk space for another job")
            self._reserved += 1
        record = JobRecord(
            workflow=workflow.id,
            params={},
            idempotency_key=idempotency_key,
            fingerprint=params_hash,
        )
        job_dir = self._store.job_dir(record.id)
        try:
            stored = workflow.store_inputs(params, job_dir)
            record = self._store.save(record.model_copy(update={"params": stored}))
        except BaseException:
            shutil.rmtree(job_dir, ignore_errors=True)
            with self._cond:
                self._reserved -= 1
                if key is not None:
                    self._pending_keys.discard(key)
            raise
        with self._cond:
            self._reserved -= 1
            if key is not None:
                self._pending_keys.discard(key)
                self._keys[key] = record.id
            self._waiting.append(record.id)
            self._cond.notify_all()
        event(log, "job.queued", job_id=record.id, workflow=workflow.id, state="queued")
        return record, True

    def _occupied_locked(self) -> int:
        return (1 if self._running else 0) + len(self._waiting) + self._reserved

    def _existing_locked(self, key: tuple[str, str], params_hash: str | None) -> JobRecord | None:
        if key in self._pending_keys:
            raise IdempotencyInFlightError(
                "a request with this Idempotency-Key is still being processed; retry shortly"
            )
        job_id = self._keys.get(key)
        if job_id is None:
            return None
        existing = self._store.get(job_id)
        if existing is None:  # pruned after the retention window: the key is free again
            del self._keys[key]
            return None
        if existing.fingerprint != params_hash:
            raise IdempotencyConflictError(
                "this Idempotency-Key was already used with different params"
            )
        return existing

    def view(self, job_id: str) -> JobView:
        record = self._unsaved.get(job_id) or self._store.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        with self._cond:
            position = list(self._waiting).index(job_id) + 1 if job_id in self._waiting else None
            progress = self._progress.get(job_id)
            waiting_for = self._waiting_for.get(job_id)
        if record.status is JobStatus.SUCCEEDED:
            progress = 1.0
        elif record.status is not JobStatus.RUNNING:
            progress = None
        if record.status is not JobStatus.QUEUED:
            position = waiting_for = None
        return JobView(
            record=record, progress=progress, queue_position=position, waiting_for=waiting_for
        )

    def output_path(self, job_id: str, index: int = 0) -> Path:
        record = self._store.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        if record.status is not JobStatus.SUCCEEDED or not record.output_files:
            raise JobConflictError(f"job is {record.status.value}, not succeeded")
        if not 0 <= index < len(record.output_files):
            raise OutputIndexError(f"output {index} not found")
        job_dir = self._store.job_dir(job_id).resolve()
        path = (job_dir / record.output_files[index]).resolve()
        if not path.is_relative_to(job_dir):
            raise JobConflictError("the output file is not available")
        if not path.is_file():
            raise JobConflictError("the output file is gone (retention expired?)")
        return path

    def cancel(self, job_id: str) -> JobRecord:
        """Cancel a job. A running job is stopped through the backend; the returned record is
        still ``running`` (with ``cancel_requested``) if the backend has not stopped yet."""
        with self._cond:
            record = self._store.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            if record.status.finished:
                raise JobConflictError(f"job already {record.status.value}")
            running = self._running
            if running is None or running.job_id != job_id:
                # Not started yet (possibly not even in the deque): the worker skips any
                # record that is no longer QUEUED, so saving CANCELED is enough.
                if job_id in self._waiting:
                    self._waiting.remove(job_id)
                    self._cond.notify_all()  # a coordination wait for it ends now
                canceled = self._store.save(record.canceled())
                self._store.discard_inputs(job_id)
                event(log, "job.canceled", job_id=job_id, workflow=record.workflow)
                return canceled
            self._update(job_id, lambda r: r.with_cancel_requested())
            running.cancel.set()
        event(log, "job.cancel_requested", job_id=job_id, workflow=record.workflow)
        deadline = time.monotonic() + CANCEL_WAIT_S
        while time.monotonic() < deadline:
            current = self._store.get(job_id)
            if current is not None and current.status.finished:
                return current
            time.sleep(0.2)
        current = self._store.get(job_id)
        if current is None:
            raise JobNotFoundError(job_id)
        return current

    # -- worker --------------------------------------------------------------------
    def _retry_after_locked(self) -> int:
        if self._running is None:
            return RETRY_AFTER_MIN_S
        left = self._running.estimate_s - (time.monotonic() - self._running.started)
        return int(min(max(left, RETRY_AFTER_MIN_S), RETRY_AFTER_MAX_S))

    def _next(self) -> _Running | None:
        """Wait until the head job may start (other GPU servers idle), then take it and claim
        the GPU slot in one step, so a cancel can never fall between "left the waiting list"
        and "is the running job". While idle, old jobs are pruned every
        ``PRUNE_INTERVAL_S`` — outside the lock."""
        while True:
            with self._cond:
                if not self._waiting and not self._stopping:
                    self._cond.wait(timeout=PRUNE_INTERVAL_S)
                if self._stopping:
                    return None
                head = self._waiting[0] if self._waiting else None
            if head is None:
                self._prune()
                continue
            began = time.monotonic()
            if not self._coordinate(head):
                continue  # canceled, no longer first, or stopping: look again
            with self._cond:
                if self._stopping:
                    return None
                if not self._waiting or self._waiting[0] != head:
                    continue
                self._waiting.popleft()
                now = time.monotonic()
                coordination = now - began if self._coordinator is not None else 0.0
                running = _Running(head, now, 0.0, threading.Event(), coordination)
                self._running = running
                return running

    def _coordinate(self, job_id: str) -> bool:
        """True once the other GPU servers let ``job_id`` start; False if it should not be
        started now (it was canceled or is no longer first, or the server is stopping)."""
        if self._coordinator is None:
            return True

        def on_waiting(peer: str) -> None:
            with self._cond:
                if job_id not in self._waiting_for:
                    event(log, "job.coordinating", job_id=job_id, waiting_for=peer)
                self._waiting_for[job_id] = peer

        try:
            return self._coordinator.wait_until_idle(
                job_id, sleep=lambda seconds: self._pause(job_id, seconds), on_waiting=on_waiting
            )
        finally:
            with self._cond:
                self._waiting_for.pop(job_id, None)

    def _pause(self, job_id: str, seconds: float) -> bool:
        """Sleep while ``job_id`` stays first in line; False as soon as it is not (or the
        server stops)."""
        deadline = time.monotonic() + seconds
        with self._cond:
            while True:
                if self._stopping or not self._waiting or self._waiting[0] != job_id:
                    return False
                left = deadline - time.monotonic()
                if left <= 0:
                    return True
                self._cond.wait(timeout=left)

    def _prune(self) -> None:
        try:
            removed = set(self._store.prune(self._retention_days))
        except Exception:
            log.exception("pruning old jobs failed")
            return
        if removed:
            with self._cond:
                self._keys = {k: v for k, v in self._keys.items() if v not in removed}

    def _loop(self) -> None:
        """The worker. Nothing a job does may end it: a dead worker would leave every later
        job queued forever while the health check still looked fine."""
        while True:
            try:
                running = self._next()
            except Exception:
                log.exception("worker: could not claim the next job")
                time.sleep(1.0)
                continue
            if running is None:
                return
            try:
                self._run_one(running)
            except Exception:
                log.exception("worker: job %s could not be concluded", running.job_id)
            finally:
                with contextlib.suppress(Exception):
                    self._store.discard_inputs(running.job_id, *SCRATCH_FILES)
                with self._cond:
                    self._running = None
                    self._progress.pop(running.job_id, None)

    def _run_one(self, running: _Running) -> None:
        record = self._store.get(running.job_id)
        if record is None or record.status is not JobStatus.QUEUED:
            return
        workflow = self._registry.get(record.workflow)
        if workflow is None:
            self._finish(
                record.id, lambda r: r.failed("unknown_workflow", r.workflow, retryable=False)
            )
            return
        self._execute(workflow, record, running)

    def _execute(self, workflow: Workflow, record: JobRecord, running: _Running) -> None:
        job_id = record.id
        job_dir = self._store.job_dir(job_id)
        try:
            request = workflow.build_request(job_id, record.params, job_dir)
            plan = self._backend.plan(request)
            running.estimate_s = plan.estimate_seconds
            if running.cancel.is_set():
                self._finish(job_id, lambda r: self._halted(r, {}))
                return
            started = self._update(
                job_id, lambda r: r.started(plan.estimate_seconds, running.coordination_s)
            )
            event(
                log,
                "job.started",
                job_id=job_id,
                workflow=started.workflow,
                state="running",
                queue_seconds=started.timings.get("queue_seconds"),
                coordination_seconds=started.timings.get("coordination_seconds"),
                estimate_seconds=started.estimate_seconds,
                generation=f"{plan.width}x{plan.height}",
                images=plan.count,
            )

            def on_progress(fraction: float) -> None:
                with self._cond:
                    self._progress[job_id] = round(min(max(fraction, 0.0), 0.99), 3)

            raw = self._backend.generate(
                request,
                job_dir,
                timeout_s=plan.estimate_seconds * self._timeout_factor + TIMEOUT_MARGIN_S,
                cancel=running.cancel,
                on_progress=on_progress,
            )
            self._conclude(workflow, started, job_dir, raw, running.cancel)
        except WorkflowFailedError as exc:
            code, message, retryable = exc.code, redact(str(exc)), exc.retryable
            self._finish(job_id, lambda r: r.failed(code, message, retryable=retryable))
        except Exception:
            log.exception("job %s crashed", job_id)
            # details go to the server log only
            message = "internal error (see the server log)"
            self._finish(job_id, lambda r: r.failed("internal", message, retryable=True))

    def _conclude(
        self,
        workflow: Workflow,
        record: JobRecord,
        job_dir: Path,
        raw: GenerationResult,
        cancel: threading.Event,
    ) -> None:
        timings = {"backend_seconds": round(raw.duration_s, 3)}
        if raw.sampling_seconds is not None:
            timings["sampling_seconds"] = round(raw.sampling_seconds, 3)
        if raw.failure_code == "engine_unstoppable":
            event(log, "backend.unstoppable", level=logging.CRITICAL, job_id=record.id)
        if raw.canceled or (cancel.is_set() and raw.failure_code != "engine_unstoppable"):
            # whatever the engine produced is not published
            self._finish(record.id, lambda r: self._halted(r, timings))
            return
        if not raw.ok:
            code = "timeout" if raw.timed_out else raw.failure_code
            message = redact(raw.describe_failure())
            self._finish(
                record.id, lambda r: r.failed(code, message, retryable=True, timings=timings)
            )
            return
        started = time.monotonic()
        try:
            output = workflow.finalize(record.id, record.params, job_dir, raw)
        except WorkflowFailedError as exc:
            timings["postprocess_seconds"] = round(time.monotonic() - started, 3)
            code, message, retryable = exc.code, redact(str(exc)), exc.retryable
            self._finish(
                record.id,
                lambda r: r.failed(code, message, retryable=retryable, timings=timings),
            )
            return
        timings["postprocess_seconds"] = round(time.monotonic() - started, 3)
        if cancel.is_set():
            for file in output.files:
                file.unlink(missing_ok=True)
            self._finish(record.id, lambda r: self._halted(r, timings))
            return
        files = [str(file.relative_to(job_dir)) for file in output.files]
        self._finish(record.id, lambda r: r.succeeded(output.result, files, timings))

    def _halted(self, record: JobRecord, timings: dict[str, float]) -> JobRecord:
        """A job stopped before it finished: canceled if the client asked for it, otherwise
        (the server is shutting down) a retryable failure, like a crash would give."""
        if record.cancel_requested or not self._stopping:
            return record.canceled(timings)
        return record.failed(
            "server_stopped",
            "the server stopped while this job was running",
            retryable=True,
            timings=timings,
        )

    def _update(self, job_id: str, transform: Callable[[JobRecord], JobRecord]) -> JobRecord:
        """Apply a transition to the latest stored record (the API thread writes too)."""
        with self._cond:
            current = self._store.get(job_id)
            if current is None:
                raise JobNotFoundError(job_id)
            return self._store.save(transform(current))

    def _finish(self, job_id: str, transform: Callable[[JobRecord], JobRecord]) -> None:
        """Record a job's end. Never raises: if the record cannot be written (disk full),
        it is retried and finally kept in memory, so the job is not shown running forever."""
        record: JobRecord | None = None
        for attempt in range(SAVE_ATTEMPTS):
            try:
                record = self._update(job_id, transform)
                break
            except JobNotFoundError:
                log.error("job %s vanished before it could be finished", job_id)
                return
            except Exception:
                log.exception("saving the end of job %s failed (attempt %d)", job_id, attempt + 1)
                time.sleep(0.5 * (attempt + 1))
        if record is None:
            with self._cond:
                last = self._store.get(job_id)
                if last is not None:
                    self._unsaved[job_id] = transform(last)
            return
        self._unsaved.pop(job_id, None)
        event(
            log,
            f"job.{record.status.value}",
            level=logging.INFO if record.status is JobStatus.SUCCEEDED else logging.WARNING,
            job_id=record.id,
            workflow=record.workflow,
            state=record.status.value,
            failure_category=record.error.code if record.error else None,
            retryable=record.error.retryable if record.error else None,
            **record.timings,
        )
