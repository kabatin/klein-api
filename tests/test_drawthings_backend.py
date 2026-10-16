"""Adapter contract test: DrawThingsCliBackend against a fake ``draw-things-cli`` that prints
the line format assumed in ``drawthings_output`` (ANSI cursor codes, step counter, footer)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from klein_api.backends.base import GenerationRequest
from klein_api.backends.drawthings import (
    DEFAULT_CLI,
    ENGINE_FLAGS,
    RECIPES,
    DrawThingsCliBackend,
    DrawThingsOptions,
    _describe_cli,
)
from klein_api.backends.process import PID_FILE, reap_orphan
from klein_api.checks import Level
from klein_api.settings import SettingsError, load_settings
from tests.conftest import make_png

# FAKE_DT_MODE: ok | fail | fail_second | error_exit0 | exit0_no_file | hang | slow |
#               wrote_elsewhere | wrote_outside | redraw
FAKE_CLI = textwrap.dedent(
    r"""
    import json, os, sys, time
    from PIL import Image
    args = sys.argv[1:]
    log = os.environ.get("FAKE_DT_CALLS")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(args) + "\n")
    def opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default
    mode = os.environ.get("FAKE_DT_MODE", "ok")
    if not sys.stdout.isatty():  # like the real CLI: a pipe gets everything at exit
        sys.stdout = open(sys.stdout.fileno(), "w", buffering=1 << 20, closefd=False)
        _print = print
        print = lambda *a, **k: _print(*a, **{**k, "flush": False})
    print("Models directory: /Users/someone/Library/Containers/x/Models\n", flush=True)
    w, h = int(opt("--width")), int(opt("--height"))
    if w % 64 or h % 64:
        print(f"Error: Image dimensions must be multiples of 64, got {w}x{h}", flush=True)
        sys.exit(1)
    out = opt("--output")
    if mode == "missing_image":
        print(f"Error: Input image path does not exist: {os.path.dirname(out)}/inputs/r.png")
        sys.exit(1)
    if mode == "fail" or (mode == "fail_second" and out.endswith("raw-1.png")):
        print("Error: Generation failed (no tensors returned)", flush=True)
        sys.exit(1)
    steps = int(opt("--steps", "4"))
    up = "\x1b[1A\x1b[K"
    print(up + "Starting... [ ] 0  %", flush=True)
    if mode == "hang":
        while True:
            time.sleep(0.1)
    for i in range(1, steps + 1):
        line = up + f"Sampling... {i} / {steps} [ ] {5 + 80 * i // steps:<3}%"
        if mode == "redraw":
            line = f"\rSampling... {max(i - 1, 0)} / {steps}\r" + line
        print(line, flush=True)
        if mode == "slow":
            time.sleep(0.25)
    if mode == "error_exit0":
        print("Error: model file is corrupt", flush=True)
        sys.exit(0)
    if mode == "wrote_elsewhere":
        out = out.replace(".png", "_1.png")
    if mode == "wrote_outside":
        out = os.path.join(os.path.dirname(os.path.dirname(out)), "elsewhere.png")
    if mode != "exit0_no_file":
        Image.new("RGB", (w, h), "purple").save(out, format="PNG")
        print(up + "Generated [█] 100%", flush=True)
        print(f"Wrote: {out}", flush=True)
    print("Generation timing:", flush=True)
    print("  Total generation time (including model loading): 3.21 s", flush=True)
    print(f"  Sampling step time ({steps} step(s)): avg 0.50 s, median 0.48 s", flush=True)
    """
)


@pytest.fixture
def fake_cli(tmp_path: Path) -> Path:
    script = tmp_path / "bin" / "draw-things-cli"
    script.parent.mkdir()
    script.write_text(f"#!{sys.executable}\n{FAKE_CLI}", encoding="utf-8")
    script.chmod(0o755)
    return script


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    root = tmp_path / "Models"
    root.mkdir()
    for recipe in RECIPES.values():
        for name in (recipe.model, *recipe.dependencies):
            (root / name).write_bytes(b"x")
    return root


def make_backend(cli: Path, models: Path, **options) -> DrawThingsCliBackend:
    return DrawThingsCliBackend(
        DrawThingsOptions(cli=str(cli), models_dir=models, stop_grace_s=2, **options)
    )


@pytest.fixture
def backend(fake_cli: Path, models_dir: Path) -> DrawThingsCliBackend:
    return make_backend(fake_cli, models_dir)


@pytest.fixture
def calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_DT_CALLS", str(path))
    return path


def request(**overrides) -> GenerationRequest:
    values = {
        "job_id": "job_X",
        "prompt": "a tram in the rain",
        "quality": "standard",
        "aspect_ratio": 16 / 9,
        "seed": 7,
        "count": 1,
        "references": (),
    }
    return GenerationRequest(**(values | overrides))


def flags_of(argv: list[str]) -> dict[str, str]:
    return {
        arg: argv[i + 1]
        for i, arg in enumerate(argv[:-1])
        if arg.startswith("--") and not argv[i + 1].startswith("--")
    }


def test_plans_use_one_megapixel_multiples_of_64(backend: DrawThingsCliBackend) -> None:
    for ratio in (21 / 9, 16 / 9, 4 / 3, 1.0, 3 / 4, 9 / 16, 9 / 21):
        plan = backend.plan(request(aspect_ratio=ratio, count=3))
        assert plan.width % 64 == 0 and plan.height % 64 == 0
        assert 0.85e6 < plan.width * plan.height < 1.1e6
        assert plan.count == 3 and plan.estimate_seconds > 0
    assert (backend.plan(request(aspect_ratio=16 / 9)).width, 768) == (1344, 768)
    assert backend.plan(request(aspect_ratio=1.0)).width == 1024
    standard = backend.plan(request(aspect_ratio=1.0, count=2)).estimate_seconds
    draft = backend.plan(request(aspect_ratio=1.0, count=2, quality="draft")).estimate_seconds
    assert (standard, draft) == (52.0, 30.0)
    assert backend.typical_seconds("standard") == 26.0


def test_estimates_never_under_predict_measured_runs(backend: DrawThingsCliBackend) -> None:
    """9B measurements on the target Mac: wide sizes are slower than their pixel count, and
    references multiply the time (1 ref 50.3 s, 2 refs 72.9 s at 1024x1024)."""
    measured = {16 / 9: 30.4, 9 / 16: 26.0, 21 / 9: 35.1, 1.0: 25.9}
    for ratio, seconds in measured.items():
        assert backend.plan(request(aspect_ratio=ratio)).estimate_seconds >= seconds
    # unmeasured skewed sizes are at least as slow as the next more skewed measured one
    assert backend.plan(request(aspect_ratio=2.0)).estimate_seconds >= 35.1
    assert backend.plan(request(aspect_ratio=9 / 21)).estimate_seconds >= 35.1
    ref = Path("r.png")
    one = backend.plan(request(aspect_ratio=1.0, references=(ref,))).estimate_seconds
    two = backend.plan(request(aspect_ratio=1.0, references=(ref, ref))).estimate_seconds
    four = backend.plan(request(aspect_ratio=1.0, references=(ref,) * 4)).estimate_seconds
    assert one >= 50.3 and two >= 72.9 and four > two * 1.5


def test_generate_passes_every_setting_explicitly(
    backend: DrawThingsCliBackend, tmp_path: Path, calls: Path
) -> None:
    progress: list[float] = []
    run_dir = tmp_path / "job"
    result = backend.generate(
        request(), run_dir, timeout_s=60, cancel=threading.Event(), on_progress=progress.append
    )
    assert result.ok, result.describe_failure()
    assert result.outputs == (run_dir / "raw-0.png",)
    assert result.sampling_seconds == pytest.approx(4 * 0.5)
    assert progress == [0.25, 0.5, 0.75, 1.0, 1.0]
    argv = json.loads(calls.read_text().splitlines()[0])
    flags = flags_of(argv)
    assert argv[0] == "generate"
    assert flags["--model"] == "flux_2_klein_9b_i8x.ckpt"
    assert (flags["--width"], flags["--height"]) == ("1344", "768")
    assert (flags["--steps"], flags["--cfg"], flags["--seed"]) == ("4", "1", "7")
    assert flags["--output"] == str(run_dir / "raw-0.png")
    for flag in ENGINE_FLAGS:
        assert flag in argv
    assert {"--local", "--offline", "--no-download-missing", "--disable-preview"} <= set(argv)
    assert "a tram in the rain" not in argv  # the prompt goes through a file
    assert (run_dir / "prompt.txt").read_text() == "a tram in the rain"
    assert "--image" not in argv and "--strength" not in argv
    assert not (run_dir / PID_FILE).exists()
    assert "Sampling" in (run_dir / "backend-0.log").read_text()


def test_count_runs_one_process_per_image_with_seed_plus_i(
    backend: DrawThingsCliBackend, tmp_path: Path, calls: Path
) -> None:
    progress: list[float] = []
    result = backend.generate(
        request(count=3, quality="draft"),
        tmp_path / "job",
        timeout_s=60,
        cancel=threading.Event(),
        on_progress=progress.append,
    )
    assert result.ok and len(result.outputs) == 3
    runs = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [flags_of(argv)["--seed"] for argv in runs] == ["7", "8", "9"]
    assert {flags_of(argv)["--model"] for argv in runs} == {"flux_2_klein_4b_i8x.ckpt"}
    assert progress == sorted(progress) and progress[-1] == 1.0
    assert progress[3] == pytest.approx(1 / 3)  # image 0 done = a third of the job
    assert result.sampling_seconds == pytest.approx(3 * 2.0)


def test_references_are_passed_as_images(
    fake_cli: Path, models_dir: Path, tmp_path: Path, calls: Path
) -> None:
    backend = make_backend(fake_cli, models_dir, max_reference_images=2)
    refs = (tmp_path / "a.png", tmp_path / "b.png")
    for ref in refs:
        make_png(ref, (256, 256))
    result = backend.generate(
        request(references=refs), tmp_path / "job", timeout_s=60, cancel=threading.Event()
    )
    assert result.ok
    argv = json.loads(calls.read_text().splitlines()[0])
    images = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--image"]
    assert images == [str(refs[0]), str(refs[1])]  # order kept: the first is the main one
    # strength stays at the CLI's 1.0: anything lower turns references into img2img
    assert "--strength" not in argv
    assert backend.max_reference_images == 2


def test_references_beyond_the_setting_never_reach_the_engine(
    fake_cli: Path, models_dir: Path, tmp_path: Path, calls: Path
) -> None:
    backend = make_backend(fake_cli, models_dir, max_reference_images=0)
    ref = tmp_path / "a.png"
    make_png(ref)
    result = backend.generate(
        request(references=(ref,)), tmp_path / "job", timeout_s=60, cancel=threading.Event()
    )
    assert not result.ok and result.failure_code == "reference_images_unsupported"
    assert not calls.exists()


def test_progress_arrives_while_the_engine_runs(
    backend: DrawThingsCliBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pipe would be block-buffered; the backend gives the CLI a pty so steps stream live."""
    monkeypatch.setenv("FAKE_DT_MODE", "slow")
    seen: list[tuple[float, float]] = []
    started = time.monotonic()
    result = backend.generate(
        request(),
        tmp_path / "job",
        timeout_s=60,
        cancel=threading.Event(),
        on_progress=lambda f: seen.append((time.monotonic() - started, f)),
    )
    assert result.ok
    assert seen[-1][1] == 1.0
    assert seen[-2][0] - seen[0][0] > 0.5  # spread over the run, not delivered at exit


def test_redrawn_progress_lines_use_the_last_value(
    backend: DrawThingsCliBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_DT_MODE", "redraw")
    progress: list[float] = []
    result = backend.generate(
        request(),
        tmp_path / "job",
        timeout_s=60,
        cancel=threading.Event(),
        on_progress=progress.append,
    )
    assert result.ok and progress[:4] == [0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize(
    ("mode", "needle"),
    [
        ("fail", "no tensors returned"),
        ("error_exit0", "model file is corrupt"),
        ("exit0_no_file", "wrote no image"),
        ("wrote_outside", "wrote no image"),
        ("missing_image", "Input image path does not exist: <path>"),
    ],
)
def test_failures(
    backend: DrawThingsCliBackend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    needle: str,
) -> None:
    monkeypatch.setenv("FAKE_DT_MODE", mode)
    result = backend.generate(request(), tmp_path / "job", timeout_s=60, cancel=threading.Event())
    assert not result.ok and needle in result.describe_failure()
    assert result.outputs == () and result.failure_code == "generation_failed"
    assert result.log_path == tmp_path / "job" / "backend-0.log"
    assert str(tmp_path) not in result.describe_failure()  # no paths for clients


def test_a_failing_second_image_discards_the_first(
    backend: DrawThingsCliBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_DT_MODE", "fail_second")
    run_dir = tmp_path / "job"
    result = backend.generate(request(count=2), run_dir, timeout_s=60, cancel=threading.Event())
    assert not result.ok and "image 1" in result.describe_failure()
    assert not list(run_dir.glob("raw-*.png"))


def test_a_renamed_output_inside_the_run_dir_is_accepted(
    backend: DrawThingsCliBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_DT_MODE", "wrote_elsewhere")
    run_dir = tmp_path / "job"
    result = backend.generate(request(), run_dir, timeout_s=60, cancel=threading.Event())
    assert result.ok and result.outputs == (run_dir / "raw-0.png",)
    assert (run_dir / "raw-0.png").is_file() and not (run_dir / "raw-0_1.png").exists()


def test_stale_output_never_counts(
    backend: DrawThingsCliBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_DT_MODE", "exit0_no_file")
    run_dir = tmp_path / "job"
    run_dir.mkdir()
    (run_dir / "raw-0.png").write_bytes(b"old image")
    result = backend.generate(request(), run_dir, timeout_s=60, cancel=threading.Event())
    assert not result.ok


def test_cancel_kills_the_process(
    backend: DrawThingsCliBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_DT_MODE", "hang")
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    result = backend.generate(request(count=2), tmp_path / "job", timeout_s=60, cancel=cancel)
    assert result.canceled and not result.ok and result.describe_failure() == "canceled"
    assert time.monotonic() - started < 10


def test_timeout_kills_the_process(
    backend: DrawThingsCliBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_DT_MODE", "hang")
    result = backend.generate(request(), tmp_path / "job", timeout_s=0.5, cancel=threading.Event())
    assert result.timed_out and "timed out" in result.describe_failure()


def test_health(backend: DrawThingsCliBackend, models_dir: Path) -> None:
    checks = {c.name: c for c in backend.health()}
    assert checks["draw-things-cli"].level is Level.OK
    assert checks["FLUX.2 [klein] standard model"].level is Level.OK
    assert checks["FLUX.2 [klein] draft model"].level is Level.OK
    assert "up to 2" in checks["reference images"].detail
    (models_dir / "qwen_3_8b_q8p.ckpt-tensordata").unlink()
    (models_dir / "qwen_3_4b_q8p.ckpt").unlink()
    checks = {c.name: c for c in backend.health()}
    model = checks["FLUX.2 [klein] standard model"]
    assert model.level is Level.FAIL and model.detail.endswith("qwen_3_8b_q8p.ckpt-tensordata")
    draft = checks["FLUX.2 [klein] draft model"]
    assert draft.level is Level.FAIL and "qwen_3_4b_q8p.ckpt" in draft.detail


def test_missing_cli(models_dir: Path, tmp_path: Path) -> None:
    backend = DrawThingsCliBackend(
        DrawThingsOptions(cli=str(tmp_path / "nope" / "draw-things-cli"), models_dir=models_dir)
    )
    assert {c.name: c for c in backend.health()}["draw-things-cli"].level is Level.FAIL
    result = backend.generate(request(), tmp_path / "job", timeout_s=5, cancel=threading.Event())
    assert result.failure_code == "backend_unavailable" and not result.ok


def test_default_paths_are_expanded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cli = tmp_path / ".local/opt/draw-things-cli/v26.1011.0/draw-things-cli"
    cli.parent.mkdir(parents=True)
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)
    backend = DrawThingsCliBackend()
    assert backend.options.cli == DEFAULT_CLI
    assert backend.cli_path() == str(cli)
    assert (
        backend.models_dir
        == tmp_path / "Library/Containers/com.liuliu.draw-things/Data/Documents/Models"
    )
    assert _describe_cli(str(cli)).endswith("(v26.1011.0)")
    assert _describe_cli("/usr/bin/true") == "/usr/bin/true"


def test_options_from_settings(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text('[backends.drawthings]\nmax_reference_images = 3\nmodels_dir = "/m"\n')
    backend = DrawThingsCliBackend.from_settings(load_settings(env={}, config_path=cfg))
    assert backend.max_reference_images == 3 and backend.models_dir == Path("/m")
    for bad in ("max_reference_images = 5", 'model = "/etc/passwd"', "reference_strength = 2"):
        cfg.write_text(f"[backends.drawthings]\n{bad}\n")
        with pytest.raises(SettingsError, match=r"backends\.drawthings"):
            DrawThingsCliBackend.from_settings(load_settings(env={}, config_path=cfg))


def test_recover_reaps_only_the_cli(backend: DrawThingsCliBackend, tmp_path: Path) -> None:
    assert backend.recover(tmp_path) is None  # no pid file


def test_reap_orphan_only_kills_our_own_process(tmp_path: Path) -> None:
    sleeper = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        (tmp_path / PID_FILE).write_text(str(sleeper.pid))
        assert reap_orphan(tmp_path, "draw-things-cli") is None  # wrong name: left alone
        assert sleeper.poll() is None
        (tmp_path / PID_FILE).write_text(str(sleeper.pid))
        assert reap_orphan(tmp_path, "sleep") is not None
        sleeper.wait(timeout=5)
        assert sleeper.returncode == -signal.SIGKILL
    finally:
        if sleeper.poll() is None:
            os.kill(sleeper.pid, signal.SIGKILL)
    assert reap_orphan(tmp_path, "sleep") is None  # no pid file
