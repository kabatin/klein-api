"""Response models (they drive ``/openapi.json``)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from klein_api.jobs.models import JobError, JobStatus
from klein_api.jobs.queue import JobView


class QualityHealth(BaseModel):
    available: bool = Field(description="this tier's model files are installed")
    reason: str | None = Field(default=None, description="why this tier cannot run")


class WorkflowHealth(BaseModel):
    available: bool = Field(description="its engine is installed with its models: jobs accepted")
    reason: str | None = Field(
        default=None,
        description="why it (or one of its quality tiers) cannot run; null when everything can",
    )
    qualities: dict[str, QualityHealth] | None = Field(
        default=None,
        description="per `quality` tier (flux2-klein: draft = 4B, standard = 9B); "
        "null for workflows without tiers",
    )


class HealthResponse(BaseModel):
    """Same shape as vpipe-api's and wan-api's ``/v1/health``, plus ``workflows``."""

    status: str
    version: str
    running: int
    waiting: int
    max_waiting: int
    workflows: dict[str, WorkflowHealth] = Field(
        default_factory=dict, description="per workflow: can it take jobs"
    )


class WorkflowInfo(BaseModel):
    id: str
    description: str
    params_schema: dict[str, Any]
    output_media_type: str


class WorkflowsResponse(BaseModel):
    workflows: list[WorkflowInfo]


class QueueCapacity(BaseModel):
    max_running: int = 1
    max_waiting: int


class Coordination(BaseModel):
    waits_for: list[str] = Field(
        description="other GPU servers on this machine that must be idle before a job starts"
    )


class CapabilitiesResponse(BaseModel):
    version: str
    queue: QueueCapacity
    workflows: list[dict[str, Any]]
    coordination: Coordination


class JobAccepted(BaseModel):
    id: str
    workflow: str
    status: JobStatus
    created_at: datetime


class JobResponse(BaseModel):
    id: str
    workflow: str
    status: JobStatus
    progress: float | None = Field(
        description="0..0.99 over all images while running, 1.0 when succeeded, else null"
    )
    queue_position: int | None = Field(description="1-based position while queued")
    waiting_for: str | None = Field(
        description="while queued first in line: the GPU server this job is waiting for"
    )
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    estimate_seconds: float | None = Field(description="expected generation time, once started")
    cancel_requested: bool = Field(
        description="a cancel was requested and the generation is being stopped"
    )
    timings: dict[str, float] = Field(
        description="seconds: queue, coordination, backend, sampling (if reported), "
        "postprocess, total"
    )
    result: dict[str, Any] | None
    error: JobError | None

    @classmethod
    def from_view(cls, view: JobView) -> JobResponse:
        record = view.record
        return cls(
            id=record.id,
            workflow=record.workflow,
            status=record.status,
            progress=view.progress,
            queue_position=view.queue_position,
            waiting_for=view.waiting_for,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            estimate_seconds=record.estimate_seconds,
            cancel_requested=record.cancel_requested,
            timings=record.timings,
            result=record.result,
            error=record.error,
        )
