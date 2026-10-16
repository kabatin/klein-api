import io
from pathlib import Path

import pytest
from PIL import Image

from klein_api.media import MediaError, cover_crop_png, inspect_png, normalize_image
from tests.conftest import make_png


def test_normalize_png(tmp_path: Path) -> None:
    out = normalize_image(make_png(tmp_path / "a.png", (96, 64)), "image/png")
    with Image.open(io.BytesIO(out)) as img:
        assert (img.format, img.mode, img.size) == ("PNG", "RGB", (96, 64))


def test_normalize_applies_exif_orientation() -> None:
    img = Image.new("RGB", (160, 80), "blue")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° CW on display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    out = normalize_image(buf.getvalue(), "image/jpeg")
    with Image.open(io.BytesIO(out)) as upright:
        assert upright.size == (80, 160)


def test_normalize_webp_and_rgba() -> None:
    buf = io.BytesIO()
    Image.new("RGBA", (100, 100), (0, 0, 255, 128)).save(buf, format="WEBP")
    with Image.open(io.BytesIO(normalize_image(buf.getvalue(), "image/webp"))) as img:
        assert img.mode == "RGB"


def test_normalize_caps_the_longest_side(tmp_path: Path) -> None:
    data = make_png(tmp_path / "big.png", (3000, 1500))
    with Image.open(io.BytesIO(normalize_image(data, "image/png", max_side=1000))) as img:
        assert img.size == (1000, 500)
    with Image.open(io.BytesIO(normalize_image(data, "image/png"))) as img:
        assert img.size == (3000, 1500)


def test_normalize_rejects_mismatch_garbage_and_tiny(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="does not match"):
        normalize_image(make_png(tmp_path / "a.png"), "image/webp")
    with pytest.raises(MediaError, match="cannot decode"):
        normalize_image(b"not an image", "image/png")
    with pytest.raises(MediaError, match="too small"):
        normalize_image(make_png(tmp_path / "t.png", (32, 32)), "image/png")
    gif = io.BytesIO()
    Image.new("RGB", (64, 64)).save(gif, format="GIF")
    with pytest.raises(MediaError, match="cannot decode"):  # GIF decoder is never tried
        normalize_image(gif.getvalue(), "image/png")


def test_cover_crop_keeps_aspect(tmp_path: Path) -> None:
    src = tmp_path / "s.png"
    img = Image.new("RGB", (400, 400), "black")
    img.paste(Image.new("RGB", (400, 100), "white"), (0, 150))  # white band in the middle
    img.save(src)
    dst = tmp_path / "out" / "d.png"
    cover_crop_png(src, dst, 256, 128, comment="klein-job:x:0")
    with Image.open(dst) as out:
        assert out.size == (256, 128) and out.mode == "RGB"
        # centre crop of a square to 2:1 keeps the middle band, scaled, not squashed
        assert out.getpixel((128, 64)) == (255, 255, 255)
        assert out.getpixel((128, 2)) == (0, 0, 0)
        assert out.info["Comment"] == "klein-job:x:0"
    assert not list(dst.parent.glob(".*.part"))


def test_cover_crop_upscales_and_drops_alpha(tmp_path: Path) -> None:
    src = tmp_path / "s.png"
    make_png(src, (1344, 768), "red", mode="RGBA")
    dst = tmp_path / "d.png"
    cover_crop_png(src, dst, 2048, 878, comment="c")
    info = inspect_png(dst)
    assert (info.format, info.mode, info.width, info.height) == ("PNG", "RGB", 2048, 878)


def test_engine_output_that_is_not_a_png(tmp_path: Path) -> None:
    src = tmp_path / "raw.png"
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="JPEG")
    src.write_bytes(buf.getvalue())  # a JPEG named .png
    with pytest.raises(MediaError, match="not a readable PNG"):
        cover_crop_png(src, tmp_path / "d.png", 64, 64, comment="c")
    assert not (tmp_path / "d.png").exists()


def test_inspect_png_refuses_truncated_files(tmp_path: Path) -> None:
    good = make_png(tmp_path / "g.png", (300, 300), "green")
    cut = tmp_path / "cut.png"
    cut.write_bytes(good[: len(good) // 2])
    with pytest.raises(MediaError, match="not a decodable PNG"):
        inspect_png(cut)
    with pytest.raises(MediaError):
        inspect_png(tmp_path / "missing.png")
    assert inspect_png(tmp_path / "g.png").width == 300
