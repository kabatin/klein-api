"""Everything ``klein-api setup`` installs, pinned in one place.

Engine: draw-things-cli and gRPCServerCLI-macOS from the GitHub release v26.1011.0 of
drawthingsai/draw-things-community (Homebrew's 1.20260430.0 lacks ``--local`` and the
FLUX.2 [klein] models). URLs, sizes and SHA-256 are GitHub's own asset metadata
(``gh api repos/drawthingsai/draw-things-community/releases/tags/v26.1011.0``: ``digest``).

Models: fetched by the CLI itself (``draw-things-cli models ensure --model FILE``, which also
fetches the model's registered text encoder and VAE) from static.libnnc.org into Draw Things'
Models folder. Sizes are the files' Content-Length there (HEAD, 2026-10-28). A quantised text
encoder's ``-tensordata`` is not downloaded: the CLI splits it out of the ``.ckpt`` (so it
counts 0 here; the ``.ckpt``'s size covers it).
"""

from __future__ import annotations

from dataclasses import dataclass

from klein_api.backends.base import ModelId
from klein_api.backends.drawthings import RECIPES

CLI_VERSION = "v26.1011.0"
_RELEASE = f"https://github.com/drawthingsai/draw-things-community/releases/download/{CLI_VERSION}"


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    sha256: str
    size: int


CLI_ASSET = Asset(
    name="draw-things-cli",
    url=f"{_RELEASE}/draw-things-cli",
    sha256="4ac2ba3c0ddb832b4193c41f9527118543cabe7f6415e9871019f15041e76525",
    size=268_399_776,
)
GRPC_ASSET = Asset(
    name="gRPCServerCLI-macOS",
    url=f"{_RELEASE}/gRPCServerCLI-macOS",
    sha256="9de7d830a9efdcb7ba60d3621cd0dc5fe854c6f97b80e00fe8f34d556cc700c0",
    size=253_459_456,
)
ENGINE_LICENCE = "draw-things-cli: GPL-3.0 (run as a separate program)"

FILE_BYTES: dict[str, int] = {
    "flux_2_klein_9b_i8x.ckpt": 9_180_635_136,
    "qwen_3_8b_q8p.ckpt": 8_426_487_808,
    "qwen_3_8b_q8p.ckpt-tensordata": 0,
    "flux_2_vae_f16.ckpt": 168_534_016,
    "flux_2_klein_4b_i8x.ckpt": 3_926_003_712,
    "qwen_3_4b_q8p.ckpt": 4_535_328_768,
    "qwen_3_4b_q8p.ckpt-tensordata": 0,
    "z_image_turbo_1.0_i8x.ckpt": 6_188_580_864,
    "qwen_3_vl_4b_instruct_q8p.ckpt": 4_535_328_768,
    "flux_1_vae_f16.ckpt": 167_870_464,
}


@dataclass(frozen=True)
class Component:
    id: ModelId
    title: str
    licence: str
    # below this the model does not fit (None: no measured bound)
    min_ram_gb: int | None
    # what is known about speed and memory on the target Mac (M5, 32 GB)
    measured: str

    @property
    def files(self) -> tuple[str, ...]:
        return RECIPES[self.id].files

    @property
    def model_file(self) -> str:
        return RECIPES[self.id].model


COMPONENTS: dict[ModelId, Component] = {
    "klein-9b": Component(
        id="klein-9b",
        title="FLUX.2 [klein] 9B (flux2-klein standard)",
        licence="FLUX.2 [klein] 9B weights: non-commercial licence (generated images may be used)",
        min_ram_gb=16,
        measured="25.9 s per 1024x1024 image, peak 10.4 GB",
    ),
    "klein-4b": Component(
        id="klein-4b",
        title="FLUX.2 [klein] 4B (flux2-klein draft)",
        licence="FLUX.2 [klein] 4B weights: Apache-2.0",
        min_ram_gb=8,
        measured="14.1 s per 1024x1024 image, peak 4.9 GB",
    ),
    "z-image-turbo": Component(
        id="z-image-turbo",
        title="Z Image Turbo (z-image-turbo workflow)",
        licence="Z Image Turbo weights: Apache-2.0",
        min_ram_gb=None,
        measured="33 s per image (56 s the very first time); memory not measured",
    ),
}
COMPONENT_IDS: tuple[ModelId, ...] = tuple(COMPONENTS)
