"""The assumed draw-things-cli line formats (to be re-checked against the real CLI)."""

from pathlib import Path

from klein_api.backends.drawthings_output import MAX_ERROR_CHARS, OutputWatch, scrub_paths
from klein_api.backends.sizes import (
    MEASURED_SECONDS,
    SIZES,
    VERIFIED_SIZES,
    generation_size,
    size_factor,
)


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
    assert set(watch.error) == {"x"}
    assert watch.wrote == Path("/tmp/run/raw-0.png")


def test_without_a_callback() -> None:
    watch = OutputWatch()
    watch.line("Sampling... 2 / 4")
    assert watch.error is None and watch.wrote is None and watch.sampling_seconds is None


def test_generation_sizes() -> None:
    assert VERIFIED_SIZES["flux_2_klein_9b_i8x.ckpt"] == {
        (1024, 1024),
        (1344, 768),
        (768, 1344),
        (1536, 640),
    }
    for verified in VERIFIED_SIZES.values():
        assert verified <= set(SIZES)
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


# -- fixtures recorded from draw-things-cli v26.1011.0 (FLUX.2 [klein] 9B, pty) -----------

SUCCESS_RUN = [
    "\x1b[1A\x1b[KStarting... [ ] 0  %\r",
    "\x1b[1A\x1b[KSampling... 1 / 4 [ ] 5  %\r",
    "\x1b[1A\x1b[KSampling... 2 / 4 [ ] 25 %\r",
    "\x1b[1A\x1b[KSampling... 3 / 4 [ ] 45 %\r",
    "\x1b[1A\x1b[KSampling... 4 / 4 [ ] 65 %\r",
    "\x1b[1A\x1b[KFinishing... [\u2588] 82 %\r",
    "\x1b[1A\x1b[KGenerated [\u2588] 100%\r",
    "Wrote: /Users/someone/.local/share/klein-api/jobs/job_X/raw-0.png\r",
    "Generation timing:\r",
    "  Sampling step time (4 step(s)): avg 3.66 s, median 3.89 s\r",
]


def test_recorded_success_run() -> None:
    watch, steps = feed(*SUCCESS_RUN)
    assert steps == [0.25, 0.5, 0.75, 1.0]
    assert watch.error is None
    assert watch.wrote == Path("/Users/someone/.local/share/klein-api/jobs/job_X/raw-0.png")
    assert watch.sampling_seconds == 4 * 3.66


def test_recorded_unknown_model_exit_64() -> None:
    watch, steps = feed(
        "Available models:",
        "  flux_2_klein_9b_i8x.ckpt   FLUX.2 [klein] 9B",
        "  flux_2_klein_4b_i8x.ckpt   FLUX.2 [klein] 4B",
        "Error: Could not resolve --model 'flux_2_klein_nope.ckpt'.",
        "Usage: draw-things-cli generate [--model <model>] ...",
        "  See 'draw-things-cli generate --help' for more information.",
    )
    assert steps == []
    assert watch.error == "Could not resolve --model 'flux_2_klein_nope.ckpt'."


def test_recorded_bad_size_exit_1() -> None:
    watch, _ = feed("Error: Image dimensions must be multiples of 64, got 1000x1024")
    assert watch.error == "Image dimensions must be multiples of 64, got 1000x1024"


def test_recorded_missing_image_hides_the_path() -> None:
    watch, _ = feed(
        "Error: Input image path does not exist: "
        "/Users/someone/.local/share/klein-api/jobs/job_X/inputs/reference-0.png"
    )
    assert watch.error == "Input image path does not exist: <path>"


def test_the_first_error_is_kept() -> None:
    watch, _ = feed("Error: first cause", "Error: follow-up")
    assert watch.error == "first cause"


def test_scrub_paths() -> None:
    assert scrub_paths("open /private/tmp/x.png failed") == "open <path> failed"
    assert scrub_paths("see ~/Models/a.ckpt") == "see <path>"
    assert scrub_paths("ratio 3/4 and a.b/c") == "ratio 3/4 and a.b/c"


def test_size_factor_is_conservative() -> None:
    model = "flux_2_klein_9b_i8x.ckpt"
    times = MEASURED_SECONDS[model]
    for size, seconds in times.items():
        assert size_factor(size, model) >= seconds / times[(1024, 1024)]
    assert size_factor((1408, 704), model) == 35.1 / 25.9
    assert size_factor((704, 1408), model) == 35.1 / 25.9
    assert size_factor((1280, 832), model) == 30.4 / 25.9
    # nothing measured is as skewed for the 4B: the most skewed measured size is used
    assert size_factor((1536, 640), "flux_2_klein_4b_i8x.ckpt") == 1.0
