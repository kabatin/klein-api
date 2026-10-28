"""Draw Things backend: runs ``draw-things-cli generate`` headless, one process per image.

Facts this module relies on (draw-things-cli v26.1011.0 on the target Mac, 2026-10-16):

- ``--local --offline --no-download-missing --disable-preview`` always: without ``--local`` a
  saved cloud key may move the job to the cloud; without ``--no-download-missing`` a missing
  file is downloaded mid-job;
- width and height must be multiples of 64 (see :mod:`klein_api.backends.sizes`);
- FLUX.2 [klein] is distilled: 4 steps, CFG 1 (never raised), passed explicitly;
- Z Image Turbo is distilled too: its recommended settings (community-models
  ``configs/z-image-turbo-1.0``: 8 steps, CFG 1, UniPC Trailing, shift 3) are passed explicitly.
  The CLI would only find them online or in a cached catalog; offline on a fresh Mac it falls
  back to its generic defaults (no per-model override for Z Image);
- ``--seed`` is deterministic and ``--output`` writes a PNG;
- ``--image`` (repeatable, the first is the main one) is a FLUX.2 reference at the default
  strength 1.0, centre-cropped by the CLI to the output aspect. ``--strength`` is never
  passed: below 1.0 it turns into img2img and returns roughly the input;
- one image per run with ``seed + i``: ``batchCount`` is ignored and ``batchSize=2`` is slower
  than two runs.

Output line formats are parsed in :mod:`klein_api.backends.drawthings_output`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from klein_api.backends.base import (
    GenerationPlan,
    GenerationRequest,
    GenerationResult,
    KleinBackend,
    ModelId,
    ProgressCallback,
)
from klein_api.backends.drawthings_run import JobRun, raw_path
from klein_api.backends.process import reap_orphan
from klein_api.backends.resident import (
    BIND_ADDRESS,
    SERVER_NAME,
    ResidentConfig,
    ResidentServer,
)
from klein_api.backends.sizes import generation_size, reference_factor, size_factor
from klein_api.checks import Check, Level
from klein_api.logs import event
from klein_api.settings import Settings, SettingsError

log = logging.getLogger("klein_api.drawthings")

DEFAULT_CLI = "~/.local/opt/draw-things-cli/v26.1011.0/draw-things-cli"
DEFAULT_MODELS_DIR = Path("~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models")
MAX_REFERENCE_IMAGES = 4
DEFAULT_REFERENCE_IMAGES = 2  # 3-4 work in principle but are unmeasured
ENGINE_FLAGS = ("--local", "--offline", "--no-download-missing", "--disable-preview")


@dataclass(frozen=True)
class Recipe:
    """Engine settings behind a model id. Never exposed to clients."""

    label: str  # the doctor check's name
    model: str
    # files the model needs besides itself (checked by `doctor` / `serve`); a quantised text
    # encoder's .ckpt is only an index (0.4 MB): its weights are in the -tensordata file
    dependencies: tuple[str, ...]
    steps: int
    cfg: float
    # per 1024x1024 image, cold process (model load included): measured 25.9 s (9B), 14 s
    # (4B) and 33 s (Z Image Turbo; 56 s for the very first run), rounded up
    seconds_per_image: float
    # more of the model's recommended settings, as ``--config-json`` (JSGenerationConfiguration)
    config_json: str | None = None

    @property
    def files(self) -> tuple[str, ...]:
        return (self.model, *self.dependencies)


VAE = "flux_2_vae_f16.ckpt"
RECIPES: dict[ModelId, Recipe] = {
    "klein-9b": Recipe(
        label="FLUX.2 [klein] standard model",
        model="flux_2_klein_9b_i8x.ckpt",
        dependencies=("qwen_3_8b_q8p.ckpt", "qwen_3_8b_q8p.ckpt-tensordata", VAE),
        steps=4,
        cfg=1.0,
        seconds_per_image=26.0,
    ),
    "klein-4b": Recipe(
        label="FLUX.2 [klein] draft model",
        model="flux_2_klein_4b_i8x.ckpt",
        dependencies=("qwen_3_4b_q8p.ckpt", "qwen_3_4b_q8p.ckpt-tensordata", VAE),
        steps=4,
        cfg=1.0,
        seconds_per_image=15.0,
    ),
    # text encoder and VAE from Draw Things' model definition (ModelZoo.swift at v26.1011.0,
    # "Z Image Turbo 1.0 (8-bit S)"). Whether its text encoder also leaves a -tensordata file
    # is unchecked, so only the .ckpt names are required.
    "z-image-turbo": Recipe(
        label="Z Image Turbo model",
        model="z_image_turbo_1.0_i8x.ckpt",
        dependencies=("qwen_3_vl_4b_instruct_q8p.ckpt", "flux_1_vae_f16.ckpt"),
        steps=8,
        cfg=1.0,
        seconds_per_image=34.0,
        # sampler 17 = UniPCTrailing (SamplerType in DataModels/config.fbs)
        config_json=json.dumps(
            {"sampler": 17, "shift": 3, "resolutionDependentShift": False, "seedMode": 2},
            separators=(",", ":"),
        ),
    ),
}
# Size timings are only measured with the 9B model; the others are assumed to scale the same way.
SIZE_REFERENCE_MODEL = "flux_2_klein_9b_i8x.ckpt"


class DrawThingsOptions(BaseModel):
    """``[backends.drawthings]`` in the config file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cli: str = DEFAULT_CLI
    models_dir: Path | None = None
    max_reference_images: int = Field(
        default=DEFAULT_REFERENCE_IMAGES, ge=0, le=MAX_REFERENCE_IMAGES
    )
    stop_grace_s: float = Field(default=15.0, ge=1.0, le=120.0)
    # keep a gRPC server with the 9B loaded between standard jobs (see resident.py)
    resident: bool = True
    resident_idle_seconds: float = Field(default=120.0, ge=1.0, le=3600.0)
    grpc_server_path: str | None = None  # default: next to the CLI
    grpc_port: int = Field(default=7859, ge=1024, le=65535)
    resident_cache_gb: int = Field(default=18, ge=1, le=256)
    # the client's port flag is inferred, not read from `generate --help`: configurable
    remote_port_flag: str = Field(default="--remote-port", pattern=r"^--[a-z][a-z-]*$")


class DrawThingsCliBackend(KleinBackend):
    name: ClassVar[str] = "drawthings"
    supports_cancel: ClassVar[bool] = True  # the CLI process is killed; the GPU is released

    def __init__(
        self,
        options: DrawThingsOptions | None = None,
        *,
        state_dir: Path | None = None,
        poll_seconds: float = 5.0,
    ) -> None:
        """``state_dir`` enables the resident server (with ``options.resident``)."""
        self.options = options or DrawThingsOptions()
        self.resident: ResidentServer | None = None
        if self.options.resident and state_dir is not None:
            self.resident = ResidentServer(
                ResidentConfig(
                    server_path=self.grpc_server_path(),
                    models_dir=self.models_dir,
                    port=self.options.grpc_port,
                    cache_gb=self.options.resident_cache_gb,
                    idle_seconds=self.options.resident_idle_seconds,
                    poll_seconds=poll_seconds,
                    state_dir=state_dir,
                    stop_grace_s=self.options.stop_grace_s,
                )
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> DrawThingsCliBackend:
        try:
            options = DrawThingsOptions.model_validate(dict(settings.backend_options()))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            raise SettingsError(f"invalid [backends.drawthings] options: {problems}") from exc
        return cls(
            options,
            state_dir=settings.data_dir / "resident",
            poll_seconds=settings.coordination_poll_seconds,
        )

    @property
    def max_reference_images(self) -> int:
        return self.options.max_reference_images

    # -- planning --------------------------------------------------------------------
    def typical_seconds(self, model: ModelId) -> float:
        return RECIPES[model].seconds_per_image

    def plan(self, request: GenerationRequest) -> GenerationPlan:
        width, height = generation_size(request.aspect_ratio)
        per_image = (
            RECIPES[request.model_id].seconds_per_image
            * size_factor((width, height), SIZE_REFERENCE_MODEL)
            * reference_factor(len(request.references))
        )
        estimate = per_image * request.count
        return GenerationPlan(width, height, request.count, estimate_seconds=round(estimate, 1))

    # -- health ----------------------------------------------------------------------
    @property
    def models_dir(self) -> Path:
        return (self.options.models_dir or DEFAULT_MODELS_DIR).expanduser()

    def cli_path(self) -> str | None:
        return shutil.which(str(Path(self.options.cli).expanduser()))

    def grpc_server_path(self) -> Path:
        if self.options.grpc_server_path is not None:
            return Path(self.options.grpc_server_path).expanduser()
        cli = self.cli_path() or str(Path(self.options.cli).expanduser())
        return Path(cli).parent / SERVER_NAME

    def health(self) -> list[Check]:
        cli = self.cli_path()
        checks = [
            Check(
                "draw-things-cli",
                Level.OK if cli else Level.FAIL,
                _describe_cli(cli) if cli else f"{self.options.cli} not found or not executable",
            )
        ]
        missing = {model: self.missing_files(model) for model in RECIPES}
        # one installed model is enough to serve; the others are optional
        level = Level.WARN if any(not files for files in missing.values()) else Level.FAIL
        for model, recipe in RECIPES.items():
            absent = missing[model]
            checks.append(
                Check(
                    recipe.label,
                    level if absent else Level.OK,
                    f"missing in {self.models_dir}: {', '.join(absent)} "
                    f"(klein-api setup --models {model})"
                    if absent
                    else f"{recipe.model} (+ {', '.join(recipe.dependencies)})",
                )
            )
        refs = self.options.max_reference_images
        checks.append(
            Check(
                "reference images",
                Level.OK,
                f"up to {refs} (FLUX.2 references)"
                if refs
                else "disabled (max_reference_images = 0): requests with references get 422",
            )
        )
        if self.options.resident:
            server = self.grpc_server_path()
            usable = os.access(server, os.X_OK)
            checks.append(
                Check(
                    "resident gRPC server",
                    Level.OK if usable else Level.WARN,
                    f"{server} (port {self.options.grpc_port}, 127.0.0.1)"
                    if usable
                    else f"{server} not found: standard jobs run one process per image "
                    "(set resident = false to silence this)",
                )
            )
        return checks

    def missing_files(self, model: ModelId) -> list[str]:
        return [name for name in RECIPES[model].files if not (self.models_dir / name).is_file()]

    def model_problem(self, model: ModelId) -> str | None:
        if self.cli_path() is None:
            return "draw-things-cli is not installed"
        missing = self.missing_files(model)
        return f"model files missing: {', '.join(missing)}" if missing else None

    # -- generation ------------------------------------------------------------------
    def command(
        self,
        cli: str,
        request: GenerationRequest,
        plan: GenerationPlan,
        run_dir: Path,
        index: int,
        remote_port: int | None = None,
    ) -> list[str]:
        """argv for one image; with ``remote_port`` it runs on the resident server."""
        recipe = RECIPES[request.model_id]
        references: list[str] = []
        for path in request.references:
            references += ["--image", str(path)]  # never --strength: < 1 means img2img
        sampling = ["--steps", str(recipe.steps), "--cfg", f"{recipe.cfg:g}"]
        if recipe.config_json is not None:
            sampling += ["--config-json", recipe.config_json]
        return [
            cli,
            "generate",
            "--model",
            recipe.model,
            "--models-dir",
            str(self.models_dir),
            # prompts go through files: no argv length limits, not visible in `ps`
            "--prompt-file",
            str(run_dir / "prompt.txt"),
            "--width",
            str(plan.width),
            "--height",
            str(plan.height),
            *sampling,
            "--seed",
            str(request.seed + index),
            *references,
            "--output",
            str(raw_path(run_dir, index)),
            *(ENGINE_FLAGS if remote_port is None else self._remote_flags(remote_port)),
        ]

    def _remote_flags(self, port: int) -> list[str]:
        # never --local / --offline here: the server (ours, on loopback) has the models
        return [
            "--remote",
            "--remote-url",
            BIND_ADDRESS,
            self.options.remote_port_flag,
            str(port),
            "--no-remote-tls",
            "--disable-preview",
        ]

    def generate(
        self,
        request: GenerationRequest,
        run_dir: Path,
        *,
        timeout_s: float,
        cancel: threading.Event,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        plan = self.plan(request)
        if len(request.references) > self.max_reference_images:  # the workflow refuses these
            return GenerationResult(
                plan=plan,
                duration_s=0.0,
                failure="this engine is not configured for reference images",
                failure_code="reference_images_unsupported",
            )
        cli = self.cli_path()
        if cli is None:
            return GenerationResult(
                plan=plan,
                duration_s=0.0,
                failure=f"{self.options.cli} is not installed",
                failure_code="backend_unavailable",
            )
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "prompt.txt").write_text(request.prompt, encoding="utf-8")
        except OSError as exc:
            return GenerationResult(
                plan=plan, duration_s=0.0, failure=f"cannot prepare inputs: {exc}"
            )
        port = self._resident_port(request)
        result: GenerationResult | None = None
        try:
            result = JobRun(self, cli, request, plan, run_dir, on_progress, port).all(
                timeout_s, cancel
            )
            return result
        finally:
            if port is not None and self.resident is not None:
                # a canceled client may leave the server generating: never reuse it then
                ok = result is not None and result.ok
                self.resident.release(discard=not ok, job_id=request.job_id)

    def _resident_port(self, request: GenerationRequest) -> int | None:
        """The resident server's port for a 9B job, else None (one process per image). Any
        other model first stops the server: it must not share the memory with the 9B."""
        if self.resident is None:
            return None
        if request.model_id != "klein-9b":
            if self.resident.running():
                self.resident.stop(f"a {request.model_id} job runs without it")
            return None
        if self.resident.acquire(request.job_id):
            return self.options.grpc_port
        event(
            log,
            "resident.fallback",
            level=logging.WARNING,
            job_id=request.job_id,
            reason="the resident server is not available: one process per image",
        )
        return None

    def recover(self, run_dir: Path) -> str | None:
        return reap_orphan(run_dir, Path(self.options.cli).name)

    def startup(self) -> str | None:
        return self.resident.reap_leftover() if self.resident is not None else None

    def shutdown(self) -> None:
        if self.resident is not None:
            self.resident.shutdown()

    def set_peer_check(self, check: Callable[[], bool] | None) -> None:
        if self.resident is not None:
            self.resident.set_peer_check(check)


def _describe_cli(path: str) -> str:
    """The release is part of the install path (``.../draw-things-cli/v26.1011.0/...``)."""
    match = re.search(r"/draw-things-cli/(v?[0-9][^/]*)/", str(Path(path).resolve()))
    return f"{path} ({match.group(1)})" if match else path
