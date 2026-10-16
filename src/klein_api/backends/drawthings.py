"""Draw Things backend: runs ``draw-things-cli generate`` headless, one process per image.

Facts this module relies on (draw-things-cli v26.1011.0 on the target Mac, 2026-10-16):

- ``--local --offline --no-download-missing --disable-preview`` always: without ``--local`` a
  saved cloud key may move the job to the cloud; without ``--no-download-missing`` a missing
  file is downloaded mid-job;
- width and height must be multiples of 64 (see :mod:`klein_api.backends.sizes`);
- FLUX.2 [klein] is distilled: 4 steps, CFG 1 (never raised), passed explicitly;
- ``--seed`` is deterministic and ``--output`` writes a PNG;
- ``--image`` (repeatable, the first is the main one) is a FLUX.2 reference at the default
  strength 1.0, centre-cropped by the CLI to the output aspect. ``--strength`` is never
  passed: below 1.0 it turns into img2img and returns roughly the input;
- one image per run with ``seed + i``: ``batchCount`` is ignored and ``batchSize=2`` is slower
  than two runs.

Output line formats are parsed in :mod:`klein_api.backends.drawthings_output`.
"""

from __future__ import annotations

import re
import shutil
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from klein_api.backends.base import (
    GenerationPlan,
    GenerationRequest,
    GenerationResult,
    KleinBackend,
    ProgressCallback,
    Quality,
)
from klein_api.backends.drawthings_output import OutputWatch
from klein_api.backends.process import ProcessOutcome, reap_orphan, run_process
from klein_api.backends.sizes import generation_size, reference_factor, size_factor
from klein_api.checks import Check, Level
from klein_api.settings import Settings, SettingsError

DEFAULT_CLI = "~/.local/opt/draw-things-cli/v26.1011.0/draw-things-cli"
DEFAULT_MODELS_DIR = Path("~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models")
MAX_REFERENCE_IMAGES = 4
DEFAULT_REFERENCE_IMAGES = 2  # 3-4 work in principle but are unmeasured
ENGINE_FLAGS = ("--local", "--offline", "--no-download-missing", "--disable-preview")


@dataclass(frozen=True)
class Recipe:
    """Engine settings behind a quality tier. Never exposed to clients."""

    model: str
    # files the model needs besides itself (checked by `doctor` / `serve`); a quantised text
    # encoder's .ckpt is only an index (0.4 MB): its weights are in the -tensordata file
    dependencies: tuple[str, ...]
    steps: int
    cfg: float
    # per 1024x1024 image, cold process (model load included): measured 25.9 s (9B) and
    # 14 s (4B), rounded up
    seconds_per_image: float


VAE = "flux_2_vae_f16.ckpt"
RECIPES: dict[Quality, Recipe] = {
    "standard": Recipe(
        model="flux_2_klein_9b_i8x.ckpt",
        dependencies=("qwen_3_8b_q8p.ckpt", "qwen_3_8b_q8p.ckpt-tensordata", VAE),
        steps=4,
        cfg=1.0,
        seconds_per_image=26.0,
    ),
    "draft": Recipe(
        model="flux_2_klein_4b_i8x.ckpt",
        dependencies=("qwen_3_4b_q8p.ckpt", "qwen_3_4b_q8p.ckpt-tensordata", VAE),
        steps=4,
        cfg=1.0,
        seconds_per_image=15.0,
    ),
}
# Size timings are only measured with the 9B model; the 4B is assumed to scale the same way.
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


class DrawThingsCliBackend(KleinBackend):
    name: ClassVar[str] = "drawthings"
    supports_cancel: ClassVar[bool] = True  # the CLI process is killed; the GPU is released

    def __init__(self, options: DrawThingsOptions | None = None) -> None:
        self.options = options or DrawThingsOptions()

    @classmethod
    def from_settings(cls, settings: Settings) -> DrawThingsCliBackend:
        try:
            options = DrawThingsOptions.model_validate(dict(settings.backend_options()))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            raise SettingsError(f"invalid [backends.drawthings] options: {problems}") from exc
        return cls(options)

    @property
    def max_reference_images(self) -> int:
        return self.options.max_reference_images

    # -- planning --------------------------------------------------------------------
    def typical_seconds(self, quality: Quality) -> float:
        return RECIPES[quality].seconds_per_image

    def plan(self, request: GenerationRequest) -> GenerationPlan:
        width, height = generation_size(request.aspect_ratio)
        per_image = (
            RECIPES[request.quality].seconds_per_image
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

    def health(self) -> list[Check]:
        cli = self.cli_path()
        checks = [
            Check(
                "draw-things-cli",
                Level.OK if cli else Level.FAIL,
                _describe_cli(cli) if cli else f"{self.options.cli} not found or not executable",
            )
        ]
        for quality, recipe in RECIPES.items():
            missing = [
                name
                for name in (recipe.model, *recipe.dependencies)
                if not (self.models_dir / name).is_file()
            ]
            checks.append(
                Check(
                    f"FLUX.2 [klein] {quality} model",
                    Level.FAIL if missing else Level.OK,
                    f"missing in {self.models_dir}: {', '.join(missing)}"
                    if missing
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
        return checks

    # -- generation ------------------------------------------------------------------
    def command(
        self, cli: str, request: GenerationRequest, plan: GenerationPlan, run_dir: Path, index: int
    ) -> list[str]:
        recipe = RECIPES[request.quality]
        references: list[str] = []
        for path in request.references:
            references += ["--image", str(path)]  # never --strength: < 1 means img2img
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
            "--steps",
            str(recipe.steps),
            "--cfg",
            f"{recipe.cfg:g}",
            "--seed",
            str(request.seed + index),
            *references,
            "--output",
            str(raw_path(run_dir, index)),
            *ENGINE_FLAGS,
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
        return _Run(self, cli, request, plan, run_dir, on_progress).all(timeout_s, cancel)

    def recover(self, run_dir: Path) -> str | None:
        return reap_orphan(run_dir, Path(self.options.cli).name)


def raw_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"raw-{index}.png"


class _Run:
    """One job: ``count`` engine processes, one after another, sharing a time budget."""

    def __init__(
        self,
        backend: DrawThingsCliBackend,
        cli: str,
        request: GenerationRequest,
        plan: GenerationPlan,
        run_dir: Path,
        on_progress: ProgressCallback | None,
    ) -> None:
        self._backend = backend
        self._cli = cli
        self._request = request
        self._plan = plan
        self._run_dir = run_dir
        self._on_progress = on_progress
        self._sampling: list[float] = []

    def all(self, timeout_s: float, cancel: threading.Event) -> GenerationResult:
        started = time.monotonic()
        outputs: list[Path] = []
        for index in range(self._plan.count):
            remaining = timeout_s - (time.monotonic() - started)
            failed = self._one(index, max(remaining, 0.0), cancel)
            if isinstance(failed, GenerationResult):
                for path in outputs:
                    path.unlink(missing_ok=True)
                return replace(failed, duration_s=time.monotonic() - started)
            outputs.append(failed)
        return GenerationResult(
            plan=self._plan,
            duration_s=time.monotonic() - started,
            outputs=tuple(outputs),
            sampling_seconds=sum(self._sampling) if self._sampling else None,
            log_path=self._run_dir / f"backend-{self._plan.count - 1}.log",
        )

    def _report(self, index: int, fraction: float) -> None:
        if self._on_progress is not None:
            self._on_progress((index + fraction) / self._plan.count)

    def _one(
        self, index: int, timeout_s: float, cancel: threading.Event
    ) -> Path | GenerationResult:
        raw = raw_path(self._run_dir, index)
        raw.unlink(missing_ok=True)  # a stale file must never pass for a fresh one
        log_path = self._run_dir / f"backend-{index}.log"
        watch = OutputWatch(lambda fraction: self._report(index, fraction))
        argv = self._backend.command(self._cli, self._request, self._plan, self._run_dir, index)
        try:
            outcome = run_process(
                argv,
                self._run_dir,
                log_path,
                timeout_s=timeout_s,
                cancel=cancel,
                on_line=watch.line,
                stop_grace_s=self._backend.options.stop_grace_s,
                # piped stdout is block-buffered: progress would only arrive at exit
                use_pty=True,
            )
        except (OSError, ValueError) as exc:  # e.g. missing binary, NUL byte in an argument
            return self._failed(0.0, log_path, failure=f"cannot start draw-things-cli: {exc}")
        if watch.sampling_seconds is not None:
            self._sampling.append(watch.sampling_seconds)
        return self._conclude(index, raw, outcome, watch, log_path)

    def _conclude(
        self, index: int, raw: Path, outcome: ProcessOutcome, watch: OutputWatch, log_path: Path
    ) -> Path | GenerationResult:
        """Exit 0 alone proves nothing: no ``Error:`` line and a fresh, non-empty file too."""
        if outcome.still_running:
            return self._failed(
                outcome.duration_s,
                log_path,
                failure="draw-things-cli did not exit after SIGKILL",
                failure_code="engine_unstoppable",
            )
        if outcome.canceled or outcome.timed_out:
            raw.unlink(missing_ok=True)  # possibly half-written
            return self._failed(
                outcome.duration_s,
                log_path,
                canceled=outcome.canceled,
                timed_out=outcome.timed_out,
            )
        if outcome.returncode != 0 or watch.error is not None:
            raw.unlink(missing_ok=True)
            reason = watch.error or f"draw-things-cli exited with code {outcome.returncode}"
            return self._failed(outcome.duration_s, log_path, failure=f"image {index}: {reason}")
        written = self._written(raw, watch)
        if written is None:
            return self._failed(
                outcome.duration_s,
                log_path,
                failure=f"image {index}: draw-things-cli wrote no image",
            )
        self._report(index, 1.0)
        return written

    def _written(self, raw: Path, watch: OutputWatch) -> Path | None:
        """The file at ``--output``; else the ``Wrote:`` path if it is inside the run dir (in
        case the CLI adjusts the name)."""
        candidates = [raw]
        if watch.wrote is not None:
            reported = (self._run_dir / watch.wrote).resolve()
            if reported.is_relative_to(self._run_dir.resolve()) and reported != raw.resolve():
                candidates.append(reported)
        for path in candidates:
            if path.is_file() and path.stat().st_size > 0:
                if path != raw:
                    path.replace(raw)
                return raw
        return None

    def _failed(
        self,
        duration_s: float,
        log_path: Path,
        *,
        failure: str | None = None,
        failure_code: str = "generation_failed",
        canceled: bool = False,
        timed_out: bool = False,
    ) -> GenerationResult:
        return GenerationResult(
            plan=self._plan,
            duration_s=duration_s,
            canceled=canceled,
            timed_out=timed_out,
            failure=failure,
            failure_code=failure_code,
            sampling_seconds=sum(self._sampling) if self._sampling else None,
            log_path=log_path,
        )


def _describe_cli(path: str) -> str:
    """The release is part of the install path (``.../draw-things-cli/v26.1011.0/...``)."""
    match = re.search(r"/draw-things-cli/(v?[0-9][^/]*)/", str(Path(path).resolve()))
    return f"{path} ({match.group(1)})" if match else path
