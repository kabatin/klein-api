"""Job records. Immutable: every state change produces a new record via ``model_copy``."""

from __future__ import annotations

import secrets
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Crockford base32 without I, L, O, U: readable, URL-safe, sortable by creation time.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_job_id() -> str:
    millis = int(time.time() * 1000)
    stamp = "".join(_ALPHABET[(millis >> (5 * i)) & 31] for i in reversed(range(10)))
    rand = "".join(secrets.choice(_ALPHABET) for _ in range(16))
    return f"job_{stamp}{rand}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    # One "l", like vpipe-api and wan-api: clients parse exactly this value.
    CANCELED = "canceled"

    @property
    def finished(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED)


class JobError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    retryable: bool


class JobRecord(BaseModel):
    """What is persisted per job (``job.json``). Params never contain inline file data."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_job_id)
    workflow: str
    status: JobStatus = JobStatus.QUEUED
    params: dict[str, Any]
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: JobError | None = None
    # deliverables, relative to the job directory, in index order
    output_files: list[str] = Field(default_factory=list)
    # Seconds the backend is expected to need (set when the job starts).
    estimate_seconds: float | None = None
    # A DELETE arrived while the job was running; it is canceled once the backend stops.
    cancel_requested: bool = False
    # queue_seconds, coordination_seconds, backend_seconds, sampling_seconds,
    # postprocess_seconds, total_seconds
    timings: dict[str, float] = Field(default_factory=dict)
    # Idempotency-Key sent with the submit, and a hash of the submitted params
    idempotency_key: str | None = None
    fingerprint: str | None = None

    def started(self, estimate_seconds: float, coordination_seconds: float = 0.0) -> JobRecord:
        """Owns the GPU slot now. Time spent waiting for other GPU servers is reported as
        ``coordination_seconds``, not as queue time."""
        now = utcnow()
        coordination = round(max(coordination_seconds, 0.0), 3)
        queued = max(_seconds(self.created_at, now) - coordination, 0.0)
        return self.model_copy(
            update={
                "status": JobStatus.RUNNING,
                "started_at": now,
                "estimate_seconds": round(estimate_seconds, 1),
                "timings": {
                    "queue_seconds": round(queued, 3),
                    "coordination_seconds": coordination,
                },
            }
        )

    def with_cancel_requested(self) -> JobRecord:
        return self.model_copy(update={"cancel_requested": True})

    def succeeded(
        self, result: dict[str, Any], output_files: list[str], timings: dict[str, float]
    ) -> JobRecord:
        return self._finish(
            JobStatus.SUCCEEDED, timings, result=result, output_files=list(output_files)
        )

    def failed(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        timings: dict[str, float] | None = None,
    ) -> JobRecord:
        error = JobError(code=code, message=message, retryable=retryable)
        return self._finish(JobStatus.FAILED, timings or {}, error=error)

    def canceled(self, timings: dict[str, float] | None = None) -> JobRecord:
        return self._finish(JobStatus.CANCELED, timings or {})

    def _finish(self, status: JobStatus, timings: dict[str, float], **update: Any) -> JobRecord:
        now = utcnow()
        merged = {**self.timings, **timings, "total_seconds": _seconds(self.created_at, now)}
        return self.model_copy(
            update={"status": status, "finished_at": now, "timings": merged, **update}
        )


def _seconds(start: datetime, end: datetime) -> float:
    return round((end - start).total_seconds(), 3)
