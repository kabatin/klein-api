"""FastAPI application. Routes live under ``/v1`` (plus ``/healthz``); one typed POST route per
registered workflow, so validation and the OpenAPI document come from the workflow's model."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Header, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from klein_api import __version__
from klein_api.api.errors import (
    ErrorEnvelope,
    busy_response,
    error_response,
    install_error_handlers,
)
from klein_api.api.schemas import (
    CapabilitiesResponse,
    Coordination,
    HealthResponse,
    JobAccepted,
    JobResponse,
    QueueCapacity,
    WorkflowInfo,
    WorkflowsResponse,
)
from klein_api.api.security import (
    IDEMPOTENCY_HEADER,
    IDEMPOTENCY_KEY_PATTERN,
    BearerAuthMiddleware,
    BodySizeLimitMiddleware,
    BusyGateMiddleware,
    LoopbackOnlyMiddleware,
)
from klein_api.jobs.queue import (
    IdempotencyConflictError,
    InsufficientStorageError,
    JobConflictError,
    JobNotFoundError,
    JobQueue,
    OutputIndexError,
    QueueFullError,
)
from klein_api.workflows.base import InvalidParamsError, Workflow, WorkflowRegistry

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorEnvelope} for code in (401, 403, 404, 409, 411, 413, 422, 429, 503)
}


def _not_found(what: str) -> JSONResponse:
    return error_response(404, "not_found", f"{what} not found")


def _submit_endpoint(queue: JobQueue, workflow: Workflow) -> Callable[..., Any]:
    def submit(params: BaseModel, idempotency_key: str | None = None) -> Any:
        try:
            record, created = queue.submit(workflow, params, idempotency_key)
        except QueueFullError as exc:
            return busy_response(exc.retry_after_s)
        except InsufficientStorageError as exc:
            return error_response(503, "insufficient_storage", str(exc), retryable=True)
        except IdempotencyConflictError as exc:
            return error_response(409, exc.code, str(exc), retryable=exc.retryable)
        except InvalidParamsError as exc:
            return error_response(422, exc.code, str(exc), details=exc.details)
        return JSONResponse(
            status_code=202 if created else 200,
            content=JobAccepted(
                id=record.id,
                workflow=record.workflow,
                status=record.status,
                created_at=record.created_at,
            ).model_dump(mode="json"),
        )

    # Give FastAPI the workflow's own params model so validation and OpenAPI are exact.
    submit.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                "params", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=workflow.params_model
            ),
            inspect.Parameter(
                "idempotency_key",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str | None,
                default=Header(
                    default=None,
                    alias=IDEMPOTENCY_HEADER,
                    pattern=IDEMPOTENCY_KEY_PATTERN,
                    description="Resubmitting the same key with the same params returns the "
                    "same job (200) instead of creating another; different params → 409.",
                ),
            ),
        ]
    )
    return submit


def build_router(
    queue: JobQueue, registry: WorkflowRegistry, waits_for: list[str] | None = None
) -> APIRouter:
    router = APIRouter()

    def health() -> Any:
        running, waiting = queue.counts()
        alive = queue.worker_alive
        body = HealthResponse(
            status="ok" if alive else "degraded",
            version=__version__,
            running=running,
            waiting=waiting,
            max_waiting=queue.max_waiting,
        )
        # a dead worker means nothing will ever run: say so instead of looking healthy
        return body if alive else JSONResponse(status_code=503, content=body.model_dump())

    for path in ("/healthz", "/v1/health"):
        router.add_api_route(
            path,
            health,
            methods=["GET"],
            response_model=HealthResponse,
            responses={503: {"model": HealthResponse, "description": "the worker is not running"}},
        )

    @router.get("/v1/capabilities", response_model=CapabilitiesResponse)
    def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(
            version=__version__,
            queue=QueueCapacity(max_waiting=queue.max_waiting),
            workflows=[w.capabilities() for w in registry],
            coordination=Coordination(waits_for=list(waits_for or [])),
        )

    @router.get("/v1/workflows", response_model=WorkflowsResponse)
    def workflows() -> WorkflowsResponse:
        return WorkflowsResponse(
            workflows=[
                WorkflowInfo(
                    id=w.id,
                    description=w.description,
                    params_schema=w.params_model.model_json_schema(),
                    output_media_type=w.output_media_type,
                )
                for w in registry
            ]
        )

    for workflow in registry:
        router.add_api_route(
            f"/v1/workflows/{workflow.id}/jobs",
            _submit_endpoint(queue, workflow),
            methods=["POST"],
            status_code=202,
            response_model=JobAccepted,
            responses=_ERRORS,
            summary=f"Queue a {workflow.id} job",
            operation_id=f"submit_{workflow.id.replace('-', '_').replace('.', '_')}",
            tags=["workflows"],
        )

    @router.post("/v1/workflows/{workflow_id}/jobs", include_in_schema=False)
    def unknown_workflow(workflow_id: str) -> JSONResponse:
        return _not_found(f"workflow {workflow_id!r}")

    @router.get("/v1/jobs/{job_id}", response_model=JobResponse, responses=_ERRORS)
    def get_job(job_id: str) -> Any:
        try:
            return JobResponse.from_view(queue.view(job_id))
        except JobNotFoundError:
            return _not_found("job")

    @router.get(
        "/v1/jobs/{job_id}/output",
        response_class=FileResponse,
        responses=_ERRORS,
        description="Image `index` (default 0) of a succeeded job. 409 until the job has "
        "succeeded, 404 for an index outside 0 … count-1.",
    )
    def get_output(job_id: str, index: int = Query(default=0)) -> Any:
        try:
            path = queue.output_path(job_id, index)
            record = queue.view(job_id).record
        except JobNotFoundError:
            return _not_found("job")
        except OutputIndexError:
            return _not_found(f"output {index}")
        except JobConflictError as exc:
            return error_response(409, "conflict", str(exc))
        workflow = registry.get(record.workflow)
        media_type = workflow.output_media_type if workflow else "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=f"{job_id}-{index}{path.suffix}")

    @router.delete(
        "/v1/jobs/{job_id}",
        response_model=JobResponse,
        responses={**_ERRORS, 202: {"model": JobResponse}},
        description="200 with status `canceled` once stopped; 202 with `cancel_requested` "
        "while a running generation is still being stopped (poll the job).",
    )
    def cancel_job(job_id: str) -> Any:
        try:
            queue.cancel(job_id)
            view = JobResponse.from_view(queue.view(job_id))
        except JobNotFoundError:
            return _not_found("job")
        except JobConflictError as exc:
            return error_response(409, "conflict", str(exc))
        status = 200 if view.status.finished else 202
        return JSONResponse(status_code=status, content=view.model_dump(mode="json"))

    return router


def create_app(
    queue: JobQueue,
    registry: WorkflowRegistry,
    *,
    token: str | None,
    max_body_bytes: int,
    waits_for: list[str] | None = None,
    manage_queue: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if manage_queue:
            queue.start()
        try:
            yield
        finally:
            if manage_queue:
                queue.stop()

    app = FastAPI(
        title="klein-api",
        version=__version__,
        summary="Run FLUX.2 [klein] text/reference-to-image as HTTP jobs",
        lifespan=lifespan,
    )
    install_error_handlers(app)
    app.include_router(build_router(queue, registry, waits_for))
    # Starlette runs the last-added middleware first:
    # auth (or loopback-only without a token), size check, busy gate, route.
    app.add_middleware(BusyGateMiddleware, queue=queue)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_body_bytes)
    if token is not None:
        app.add_middleware(BearerAuthMiddleware, token=token)
    else:
        app.add_middleware(LoopbackOnlyMiddleware)
    return app
