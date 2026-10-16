"""Workflow ``flux2-klein``: FLUX.2 [klein] text (and optionally reference images) to image.

The client asks for an exact output size and 1-4 images. The backend generates each image at
a supported ~1 MP size closest to the requested aspect ratio; this workflow then delivers
exactly what was asked: cover + centre crop (lanczos) to ``width x height``, 8-bit RGB PNG
without alpha — and decodes every file again before the job succeeds.
"""

from __future__ import annotations

import base64
import binascii
import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from klein_api.backends.base import GenerationRequest, GenerationResult, Quality
from klein_api.media import (
    ALLOWED_IMAGE_FORMATS,
    MAX_IMAGE_PIXELS,
    MAX_REFERENCE_SIDE,
    ImageInfo,
    MediaError,
    cover_crop_png,
    inspect_png,
    normalize_image,
)
from klein_api.workflows.base import (
    InvalidParamsError,
    Workflow,
    WorkflowFailedError,
    WorkflowOutput,
)

WORKFLOW_ID = "flux2-klein"
OUTPUT_MEDIA_TYPE = "image/png"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PROMPT_CHARS = 4000
MIN_SIDE = 256
MAX_SIDE = 2048
# aspect ratio bounds as integers: 9:21 <= width:height <= 21:9
ASPECT_NUM, ASPECT_DEN = 21, 9
MAX_COUNT = 4
MAX_REFERENCES = 4
MAX_SEED = 2**31 - 1

IMAGE_MEDIA_TYPES = sorted(set(ALLOWED_IMAGE_FORMATS.values()))

# C0 control characters and DEL; tab and newline are ordinary prompt text.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: str = Field(description="base64-encoded image", json_schema_extra={"format": "base64"})
    media_type: Literal["image/png", "image/jpeg", "image/webp"]

    @field_validator("data")
    @classmethod
    def _decodable(cls, value: str) -> str:
        # cheap length check before decoding: 4 base64 chars carry 3 bytes
        if len(value) > (MAX_IMAGE_BYTES // 3 + 1) * 4:
            raise ValueError(f"image larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("data is not valid base64") from exc
        if not raw:
            raise ValueError("data is empty")
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"image larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
        return value

    def decoded(self) -> bytes:
        return base64.b64decode(self.data, validate=True)


class OutputSize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(ge=MIN_SIDE, le=MAX_SIDE, description="even number of pixels")
    height: int = Field(ge=MIN_SIDE, le=MAX_SIDE, description="even number of pixels")

    @model_validator(mode="after")
    def _shape(self) -> OutputSize:
        if self.width % 2 or self.height % 2:
            raise ValueError("width and height must be even")
        too_wide = ASPECT_DEN * self.width > ASPECT_NUM * self.height
        too_tall = ASPECT_NUM * self.width < ASPECT_DEN * self.height
        if too_wide or too_tall:
            raise ValueError("aspect ratio must be between 9:21 and 21:9")
        return self


class Flux2KleinParams(BaseModel):
    """POST body for ``flux2-klein``."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    output: OutputSize
    quality: Quality = Field(
        default="standard",
        description="`draft` for fast iteration, `standard` for final candidates",
    )
    seed: int | None = Field(
        default=None, ge=0, le=MAX_SEED, description="image i uses seed + i; random if omitted"
    )
    count: int = Field(default=1, ge=1, le=MAX_COUNT, description="images, run one after another")
    reference_images: list[ImageInput] = Field(
        default_factory=list,
        max_length=MAX_REFERENCES,
        description="subjects, wardrobe, places or style to keep; the first is the main one. "
        "At most `reference_images.max` from /v1/capabilities (0: send none)",
    )

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: str) -> str:
        if _CONTROL.search(value):
            raise ValueError("prompt contains control characters")
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt is blank")
        return stripped


class Flux2KleinWorkflow(Workflow):
    id = WORKFLOW_ID
    description = (
        "FLUX.2 [klein]: text (and optional reference images) to image, delivered as PNG at "
        "the exact requested size"
    )
    output_media_type = OUTPUT_MEDIA_TYPE
    params_model = Flux2KleinParams

    def __init__(
        self, *, max_reference_images: int = 0, typical_seconds: Mapping[str, float] | None = None
    ) -> None:
        self.max_reference_images = max_reference_images
        self.typical_seconds = dict(typical_seconds or {"draft": 15.0, "standard": 26.0})

    def store_inputs(self, params: BaseModel, job_dir: Path) -> dict[str, Any]:
        p = cast(Flux2KleinParams, params)
        if len(p.reference_images) > self.max_reference_images:
            allowed = self.max_reference_images
            raise InvalidParamsError(
                f"this server accepts at most {allowed} reference image(s)",
                details=[
                    {
                        "loc": ["body", "reference_images"],
                        "msg": f"at most {allowed} reference image(s) (see /v1/capabilities)",
                    }
                ],
            )
        stored = p.model_dump(exclude={"reference_images"})
        stored["seed"] = p.seed if p.seed is not None else secrets.randbelow(MAX_SEED + 1)
        references: list[str] = []
        for index, image in enumerate(p.reference_images):
            try:
                png = normalize_image(
                    image.decoded(), image.media_type, max_side=MAX_REFERENCE_SIDE
                )
            except MediaError as exc:
                raise InvalidParamsError(
                    f"reference_images[{index}]: {exc}",
                    details=[{"loc": ["body", "reference_images", str(index)], "msg": str(exc)}],
                ) from exc
            inputs = job_dir / "inputs"
            inputs.mkdir(parents=True, exist_ok=True)
            path = inputs / f"reference-{index}.png"
            path.write_bytes(png)
            references.append(str(path.relative_to(job_dir)))
        stored["reference_images"] = references
        return stored

    def build_request(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path
    ) -> GenerationRequest:
        out = params["output"]
        root = job_dir.resolve()
        references: list[Path] = []
        for relative in params.get("reference_images", []):
            path = (job_dir / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise WorkflowFailedError(
                    "input_missing", "a stored reference image is gone", retryable=False
                )
            references.append(path)
        return GenerationRequest(
            job_id=job_id,
            prompt=params["prompt"],
            quality=params["quality"],
            aspect_ratio=out["width"] / out["height"],
            seed=params["seed"],
            count=params["count"],
            references=tuple(references),
        )

    def finalize(
        self, job_id: str, params: Mapping[str, Any], job_dir: Path, raw: GenerationResult
    ) -> WorkflowOutput:
        width, height = params["output"]["width"], params["output"]["height"]
        count = params["count"]
        if len(raw.outputs) != count:
            raise WorkflowFailedError(
                "invalid_output",
                f"the engine delivered {len(raw.outputs)} of {count} images",
                retryable=True,
            )
        finals: list[Path] = []
        try:
            source = inspect_png(raw.outputs[0])
            for index, produced in enumerate(raw.outputs):
                final = job_dir / f"output-{index}.png"
                finals.append(final)
                cover_crop_png(
                    produced, final, width, height, comment=f"klein-job:{job_id}:{index}"
                )
                problem = _output_problem(inspect_png(final), width, height)
                if problem is not None:
                    raise WorkflowFailedError("invalid_output", problem, retryable=True)
        except (MediaError, WorkflowFailedError) as exc:
            for final in finals:
                final.unlink(missing_ok=True)
            if isinstance(exc, WorkflowFailedError):
                raise
            raise WorkflowFailedError("invalid_output", str(exc), retryable=True) from exc
        for produced in raw.outputs:
            produced.unlink(missing_ok=True)
        seed = params["seed"]
        return WorkflowOutput(
            files=tuple(finals),
            result={
                "outputs": [
                    {
                        "index": index,
                        "media_type": self.output_media_type,
                        "width": width,
                        "height": height,
                        "seed_used": seed + index,
                    }
                    for index in range(count)
                ],
                "seed_used": seed,
                "details": {
                    "generation": {
                        "width": source.width,
                        "height": source.height,
                        "quality": params["quality"],
                        "references": len(params.get("reference_images", [])),
                    }
                },
            },
        )

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
            "quality": ["draft", "standard"],
            "seed": True,
            "reference_images": {
                "max": self.max_reference_images,
                "media_types": IMAGE_MEDIA_TYPES,
                "max_bytes": MAX_IMAGE_BYTES,
                "max_pixels": MAX_IMAGE_PIXELS,
            },
            "typical_seconds": {
                quality: round(self.typical_seconds[quality]) for quality in ("draft", "standard")
            },
        }

    def smoke_params(self) -> dict[str, Any]:
        return {
            "prompt": "A small wooden boat on a calm lake at dawn, soft mist, photograph.",
            "output": {"width": 1024, "height": 1024},
            "quality": "draft",
            "seed": 1,
            "count": 1,
        }


def _output_problem(info: ImageInfo, width: int, height: int) -> str | None:
    if info.format != "PNG":
        return f"output is {info.format}, not PNG"
    if info.mode != "RGB":
        return f"output mode is {info.mode}, not 8-bit RGB"
    if (info.width, info.height) != (width, height):
        return f"output is {info.width}x{info.height}, expected {width}x{height}"
    return None
