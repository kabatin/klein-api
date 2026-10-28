"""In-process stand-ins: a scriptable klein backend and a trivial text workflow."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field

from klein_api.backends.base import (
    GenerationPlan,
    GenerationRequest,
    GenerationResult,
    KleinBackend,
    ModelId,
    ProgressCallback,
)
from klein_api.checks import Check, Level
from klein_api.workflows.base import (
    InvalidParamsError,
    Workflow,
    WorkflowFailedError,
    WorkflowOutput,
)
from tests.conftest import make_png


class FakeKleinBackend(KleinBackend):
    """``mode`` picks the outcome:

    ok (writes one image per requested image: a real PNG when ``png`` is set, else the
    prompt as text), fail, timeout, crash (raises), no_output, short (one image fewer),
    malformed (not a PNG), rgba (PNGs with alpha), ignore_cancel (finishes normally even when
    canceled, like an engine without real cancel).
    ``block`` keeps a run going until ``release`` (or cancel) is set.
    """

    name: ClassVar[str] = "fake"
    supports_cancel: ClassVar[bool] = True

    def __init__(
        self, *, png: bool = False, size: tuple[int, int] = (448, 256), max_refs: int = 0
    ) -> None:
        self.mode = "ok"
        self.png = png
        self.size = size
        self.max_refs = max_refs
        self.block = False
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls: list[GenerationRequest] = []
        self.recovered: list[Path] = []
        self.healthy = True
        self.lifecycle: list[str] = []
        self.peer_check: Callable[[], bool] | None = None
        # model -> why it cannot run (absent: it can)
        self.problems: dict[ModelId, str] = {}

    @property
    def max_reference_images(self) -> int:
        return self.max_refs

    def typical_seconds(self, model: ModelId) -> float:
        return {"klein-4b": 15.0, "klein-9b": 26.0, "z-image-turbo": 34.0}[model]

    def model_problem(self, model: ModelId) -> str | None:
        return self.problems.get(model)

    def plan(self, request: GenerationRequest) -> GenerationPlan:
        return GenerationPlan(self.size[0], self.size[1], request.count, estimate_seconds=10.0)

    def health(self) -> list[Check]:
        level = Level.OK if self.healthy else Level.FAIL
        return [Check("fake backend", level, "ready" if self.healthy else "model missing")]

    def recover(self, run_dir: Path) -> str | None:
        self.recovered.append(run_dir)
        return "nothing to stop"

    def startup(self) -> str | None:
        self.lifecycle.append("startup")
        return "stopped a leftover engine server"

    def shutdown(self) -> None:
        self.lifecycle.append("shutdown")

    def set_peer_check(self, check: Callable[[], bool] | None) -> None:
        self.peer_check = check

    def generate(
        self,
        request: GenerationRequest,
        run_dir: Path,
        *,
        timeout_s: float,
        cancel: threading.Event,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        self.calls.append(request)
        plan = self.plan(request)
        self.started.set()
        if on_progress is not None:
            on_progress(0.5)
        if self.block:
            while not self.release.is_set() and not (
                cancel.is_set() and self.mode != "ignore_cancel"
            ):
                self.release.wait(0.02)
        if cancel.is_set() and self.mode != "ignore_cancel":
            return GenerationResult(plan=plan, duration_s=0.1, canceled=True)
        if self.mode == "crash":
            raise RuntimeError("backend exploded at /Users/someone/secret")
        if self.mode == "timeout":
            return GenerationResult(plan=plan, duration_s=9.0, timed_out=True)
        if self.mode == "fail":
            return GenerationResult(
                plan=plan, duration_s=1.0, failure=f"engine failed reading {Path.home()}/x"
            )
        if self.mode == "no_output":
            return GenerationResult(plan=plan, duration_s=1.0)
        run_dir.mkdir(parents=True, exist_ok=True)
        count = request.count - 1 if self.mode == "short" else request.count
        outputs = tuple(self._write(run_dir / f"raw-{i}.png", request) for i in range(count))
        return GenerationResult(plan=plan, duration_s=1.0, outputs=outputs, sampling_seconds=0.8)

    def _write(self, path: Path, request: GenerationRequest) -> Path:
        if self.mode == "malformed":
            path.write_bytes(b"\x89PNG\r\n\x1a\n this is not really a png")
        elif self.png:
            mode = "RGBA" if self.mode == "rgba" else "RGB"
            make_png(path, self.size, "blue", mode=mode)
        else:
            path.write_text(request.prompt)
        return path


class EchoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    fail_finalize: bool = False
    reject: bool = False


class EchoWorkflow(Workflow):
    """Upper-cases the backend's text output: exercises the queue without images."""

    id: ClassVar[str] = "echo"
    description: ClassVar[str] = "writes the text to a file"
    output_media_type: ClassVar[str] = "text/plain"
    params_model: ClassVar[type[BaseModel]] = EchoParams

    def store_inputs(self, params: BaseModel, job_dir: Path) -> dict[str, Any]:
        p = cast(EchoParams, params)
        if p.reject:
            raise InvalidParamsError(
                "rejected on purpose", details=[{"loc": ["body", "reject"], "msg": "no"}]
            )
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "inputs").mkdir(exist_ok=True)
        return p.model_dump()

    def build_request(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path
    ) -> GenerationRequest:
        return GenerationRequest(
            job_id=job_id,
            prompt=params["text"],
            quality="draft",
            aspect_ratio=16 / 9,
            seed=1,
            count=1,
        )

    def finalize(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path, raw: GenerationResult
    ) -> WorkflowOutput:
        if params["fail_finalize"]:
            raise WorkflowFailedError("postprocess_failed", "boom", retryable=False)
        out = job_dir / "output-0.txt"
        out.write_text(raw.outputs[0].read_text().upper())
        return WorkflowOutput(files=(out,), result={"length": len(params["text"])})

    def capabilities(self) -> dict[str, Any]:
        return {"id": self.id}

    def smoke_params(self) -> dict[str, Any]:
        return {"text": "hi"}
