"""The assumed draw-things-cli line formats (to be re-checked against the real CLI)."""

from pathlib import Path

from klein_api.backends.drawthings_output import MAX_ERROR_CHARS, OutputWatch
from klein_api.backends.sizes import SIZES, VERIFIED_SIZES, generation_size


def feed(*lines: str) -> tuple[OutputWatch, list[float]]:
    steps: list[float] = []
    watch = OutputWatch(steps.append)
    for line in lines:
        watch.line(line)
    return watch, steps


def test_progress_lines_with_ansi_codes() -> None:
    _, steps = feed(
        "\x1b[1A\x1b[KStarting... [ ] 0  %",
        "\x1b[1A\x1b[KSampling... 1 / 4 [█   ] 25 %",
        "\x1b[1A\x1b[KSampling... 4 / 4 [████] 85 %",
        "Sampling... 9 / 4",  # never above 1
        "Sampling... 1 / 0",  # nonsense is ignored
    )
    assert steps == [0.25, 1.0, 1.0]


def test_carriage_return_redraws_use_the_last_value() -> None:
    _, steps = feed("Sampling... 1 / 4\rSampling... 2 / 4\r")
    assert steps == [0.5]


def test_footer_error_and_wrote() -> None:
    watch, _ = feed(
        "Generation timing:",
        "  Sampling step time (4 step(s)): avg 2.75 s, median 2.70 s",
        "Error: " + "x" * 2000,
        "Wrote: /tmp/run/raw-0.png  ",
        "something unknown",
        "",
    )
    assert watch.sampling_seconds == 11.0
    assert watch.error is not None and len(watch.error) == MAX_ERROR_CHARS
    assert watch.wrote == Path("/tmp/run/raw-0.png")


def test_without_a_callback() -> None:
    watch = OutputWatch()
    watch.line("Sampling... 2 / 4")
    assert watch.error is None and watch.wrote is None and watch.sampling_seconds is None


def test_generation_sizes() -> None:
    assert set(SIZES) >= VERIFIED_SIZES
    for width, height in SIZES:
        assert width % 64 == 0 and height % 64 == 0
        assert 0.85e6 < width * height < 1.1e6
    assert generation_size(1.0) == (1024, 1024)
    assert generation_size(16 / 9) == (1344, 768)
    assert generation_size(9 / 16) == (768, 1344)
    assert generation_size(4 / 3) == (1152, 896)
    assert generation_size(21 / 9) == (1536, 640)
    assert generation_size(9 / 21) == (640, 1536)
    try:
        generation_size(0)
    except ValueError:
        pass
    else:
        raise AssertionError("a zero ratio must be refused")
