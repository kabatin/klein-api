"""The workflow contract: a closed, server-registered recipe exposed to clients.

A workflow owns the public side of a generation: the params schema (validated at submit
time), turning params into a backend-neutral :class:`GenerationRequest`, and turning the
backend's raw images into the deliverables it promises (exact size, verified). Inline inputs
(base64 images) are written into the job directory by ``store_inputs`` so persisted records
hold only JSON-safe values and can be re-run after a restart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from klein_api.backends.base import GenerationRequest, GenerationResult


class InvalidParamsError(ValueError):
    """Params passed schema validation but cannot be used (e.g. an undecodable image)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_params",
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class WorkflowFailedError(RuntimeError):
    """Post-processing or output validation failed; ``retryable`` hints whether to retry."""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class WorkflowOutput:
    # the deliverables, in index order, inside the job directory
    files: tuple[Path, ...]
    result: dict[str, Any]


class Workflow(ABC):
    id: ClassVar[str]
    description: ClassVar[str]
    output_media_type: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def store_inputs(self, params: BaseModel, job_dir: Path) -> dict[str, Any]:
        """Persist inline inputs; return the JSON-safe params stored on the job record."""

    @abstractmethod
    def build_request(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path
    ) -> GenerationRequest: ...

    @abstractmethod
    def finalize(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path, raw: GenerationResult
    ) -> WorkflowOutput:
        """Turn the backend's raw output into the promised deliverables, or raise."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Client-facing limits (sizes, tiers, inputs) — no engine details."""

    @abstractmethod
    def smoke_params(self) -> dict[str, Any]:
        """Cheapest valid params, used by ``klein-api doctor --smoke``."""


class WorkflowRegistry:
    def __init__(self, workflows: Iterable[Workflow]) -> None:
        self._by_id: dict[str, Workflow] = {}
        for workflow in workflows:
            if workflow.id in self._by_id:
                raise ValueError(f"duplicate workflow id {workflow.id!r}")
            self._by_id[workflow.id] = workflow

    def get(self, workflow_id: str) -> Workflow | None:
        return self._by_id.get(workflow_id)

    def __iter__(self) -> Iterator[Workflow]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)
