"""Workflow ``z-image-turbo``: Z Image Turbo (Tongyi-MAI, Apache-2.0) text to image.

Same delivery as ``flux2-klein`` — generated at a ~1 MP size in multiples of 64 closest to the
requested aspect ratio, then cover-cropped to the exact size as verified 8-bit RGB PNG — but
text only: one model, no quality tiers, no reference images. Steps, guidance and sampler are
the model's recommended settings (distilled), set by the backend, never by the client.

Measured on the target Mac: 33 s per image (56 s for the very first run). Peak memory and the
non-square sizes are not measured yet; the size table is the one verified with FLUX.2 [klein].
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from klein_api.backends.base import GenerationRequest, GenerationResult, ModelId
from klein_api.workflows.base import Workflow, WorkflowOutput
from klein_api.workflows.flux2_klein import (
    ASPECT_DEN,
    ASPECT_NUM,
    MAX_COUNT,
    MAX_PROMPT_CHARS,
    MAX_SEED,
    MAX_SIDE,
    MIN_SIDE,
    OUTPUT_MEDIA_TYPE,
    ModelProblem,
    OutputSize,
    clean_prompt,
    deliver_pngs,
)

WORKFLOW_ID = "z-image-turbo"
MODEL: ModelId = "z-image-turbo"
DEFAULT_TYPICAL_SECONDS = 34.0


class ZImageTurboParams(BaseModel):
    """POST body for ``z-image-turbo``."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    output: OutputSize
    seed: int | None = Field(
        default=None, ge=0, le=MAX_SEED, description="image i uses seed + i; random if omitted"
    )
    count: int = Field(default=1, ge=1, le=MAX_COUNT, description="images, run one after another")

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: str) -> str:
        return clean_prompt(value)


class ZImageTurboWorkflow(Workflow):
    id = WORKFLOW_ID
    description = "Z Image Turbo: text to image, delivered as PNG at the exact requested size"
    output_media_type = OUTPUT_MEDIA_TYPE
    params_model = ZImageTurboParams

    def __init__(
        self,
        *,
        typical_seconds: float = DEFAULT_TYPICAL_SECONDS,
        model_problem: ModelProblem | None = None,
    ) -> None:
        self.typical_seconds = typical_seconds
        self._model_problem = model_problem or (lambda _: None)

    def unavailable_reason(self, params: BaseModel | None = None) -> str | None:
        return self._model_problem(MODEL)

    def store_inputs(self, params: BaseModel, job_dir: Path) -> dict[str, Any]:
        p = cast(ZImageTurboParams, params)
        stored = p.model_dump()
        stored["seed"] = p.seed if p.seed is not None else secrets.randbelow(MAX_SEED + 1)
        return stored

    def build_request(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path
    ) -> GenerationRequest:
        out = params["output"]
        return GenerationRequest(
            job_id=job_id,
            prompt=params["prompt"],
            quality="standard",  # unused: the model has one recipe
            aspect_ratio=out["width"] / out["height"],
            seed=params["seed"],
            count=params["count"],
            model=MODEL,
        )

    def finalize(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path, raw: GenerationResult
    ) -> WorkflowOutput:
        return deliver_pngs(job_id, params, job_dir, raw, {})

    def capabilities(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "output": {
                "media_type": self.output_media_type,
                "width": {"min": MIN_SIDE, "max": MAX_SIDE, "multiple_of": 2},
                "height": {"min": MIN_SIDE, "max": MAX_SIDE, "multiple_of": 2},
                "aspect_ratio": {
                    "min": f"{ASPECT_DEN}:{ASPECT_NUM}",
                    "max": f"{ASPECT_NUM}:{ASPECT_DEN}",
                },
                "fit": "cover (generated at a supported size, then centre-cropped and resized)",
            },
            "count": {"min": 1, "max": MAX_COUNT},
            "seed": True,
            "reference_images": {"max": 0},
            "typical_seconds": round(self.typical_seconds),
        }

    def smoke_params(self) -> dict[str, Any]:
        return {
            "prompt": "A small wooden boat on a calm lake at dawn, soft mist, photograph.",
            "output": {"width": 1024, "height": 1024},
            "seed": 1,
            "count": 1,
        }
