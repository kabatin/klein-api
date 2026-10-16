"""Pillow helpers: validate uploads, cover-crop engine output, verify deliverables."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, PngImagePlugin, UnidentifiedImageError

# MPO = multi-picture JPEG written by many cameras/phones; its first frame is a plain JPEG.
ALLOWED_IMAGE_FORMATS = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "MPO": "image/jpeg",
    "WEBP": "image/webp",
}
# Decoders that ever see untrusted bytes. MPO has no decoder of its own: Pillow's JPEG
# decoder returns MPO images, and naming "MPO" here makes Pillow raise KeyError.
DECODERS = ("PNG", "JPEG", "WEBP")
MAX_IMAGE_PIXELS = 50_000_000
MIN_IMAGE_SIDE = 64
# References handed to the engine are scaled down to this longest side (aspect kept): the
# engine does not need 50 MP, and memory is shared with the model.
MAX_REFERENCE_SIDE = 2048


class MediaError(RuntimeError):
    """An image is not usable (message is client-safe: no paths, no object reprs)."""


@dataclass(frozen=True)
class ImageInfo:
    format: str
    mode: str
    width: int
    height: int


def normalize_image(data: bytes, media_type: str, *, max_side: int | None = None) -> bytes:
    """Validate an uploaded image and re-encode it as an upright RGB PNG.

    The size is checked from the header before any pixel is decoded (decompression bombs),
    and only the PNG/JPEG/WebP decoders ever see the untrusted bytes.
    """
    try:
        with Image.open(io.BytesIO(data), formats=DECODERS) as img:
            actual = ALLOWED_IMAGE_FORMATS.get(img.format or "")
            if actual is None:
                raise MediaError(f"unsupported image format {img.format!r}")
            if actual != media_type:
                raise MediaError(f"media_type {media_type!r} does not match the data ({actual})")
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise MediaError(f"image too large ({img.width}x{img.height})")
            if min(img.width, img.height) < MIN_IMAGE_SIDE:
                raise MediaError(f"image too small ({img.width}x{img.height})")
            upright = ImageOps.exif_transpose(img).convert("RGB")
    except MediaError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise MediaError(f"cannot decode image ({type(exc).__name__})") from exc
    except Exception as exc:  # Pillow plugins raise ValueError/OSError/SyntaxError/...
        raise MediaError(f"cannot decode image ({type(exc).__name__})") from exc
    if max_side is not None and max(upright.size) > max_side:
        upright.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    upright.save(out, format="PNG")
    return out.getvalue()


def cover_crop_png(src: Path, dst: Path, width: int, height: int, *, comment: str) -> None:
    """Scale the engine's PNG to cover ``width``x``height`` and crop the centre (lanczos).

    The result is 8-bit RGB without alpha. ``comment`` goes into a tEXt chunk, which makes
    every output byte-unique (a consumer that de-duplicates by checksum never mistakes a new
    image for an old one).
    """
    try:
        with Image.open(src, formats=["PNG"]) as img:
            fitted = ImageOps.fit(
                img.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS
            )
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise MediaError(
            f"the engine's image is not a readable PNG ({type(exc).__name__})"
        ) from exc
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", comment)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.part")
    try:
        fitted.save(tmp, format="PNG", pnginfo=info)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def inspect_png(path: Path) -> ImageInfo:
    """Fully decode ``path`` (a truncated file fails here) and report what it is."""
    try:
        with Image.open(path, formats=["PNG"]) as img:
            img.load()
            return ImageInfo(img.format or "", img.mode, img.width, img.height)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise MediaError(f"{path.name} is not a decodable PNG ({type(exc).__name__})") from exc
