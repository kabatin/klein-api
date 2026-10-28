"""``setup``'s model step: the exact ``models ensure`` argv, the silence watchdog, and the
file check afterwards — with a fake CLI run, never a real download."""

import threading
from pathlib import Path

import pytest

from klein_api.backends.process import run_process
from klein_api.setup.errors import SetupError
from klein_api.setup.models import ModelEnsurer, ensure_argv, missing_files
from klein_api.setup.pins import COMPONENTS

CLI = Path("/opt/dt/draw-things-cli")


def test_ensure_argv_is_the_documented_form(tmp_path: Path) -> None:
    argv = ensure_argv(CLI, COMPONENTS["z-image-turbo"], tmp_path)
    assert argv == [
        str(CLI),
        "models",
        "ensure",
        "--model",
        "z_image_turbo_1.0_i8x.ckpt",
        "--models-dir",
        str(tmp_path),
    ]


class FakeRun:
    """Scripted `models ensure` runs: each entry is "ok" (creates the files), "fail"
    (exit 1), "stall" (silent until canceled) or "partial" (exit 0, one file short)."""

    def __init__(self, models_dir: Path, script: list[str]) -> None:
        self.models_dir = models_dir
        self.script = list(script)
        self.argvs: list[list[str]] = []

    def __call__(self, argv, log_path, cancel: threading.Event, on_line, on_activity):
        self.argvs.append(list(argv))
        mode = self.script.pop(0)
        component = next(c for c in COMPONENTS.values() if c.model_file == argv[4])
        if mode == "stall":
            cancel.wait(5)
            return -2
        if mode == "fail":
            on_line("Error: network")
            return 1
        files = component.files[:-1] if mode == "partial" else component.files
        for name in files:
            on_activity()
            (self.models_dir / name).write_bytes(b"x")
        on_line(f"Model ready: {component.model_file}")
        return 0


def ensurer(tmp_path: Path, run: FakeRun, **kw) -> ModelEnsurer:
    return ModelEnsurer(CLI, run.models_dir, tmp_path / "work", run=run, **kw)


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    path = tmp_path / "Models"
    path.mkdir()
    return path


def test_a_successful_ensure_leaves_every_file(tmp_path: Path, models_dir: Path) -> None:
    run = FakeRun(models_dir, ["ok"])
    ensurer(tmp_path, run).ensure(COMPONENTS["klein-4b"])
    assert missing_files(COMPONENTS["klein-4b"], models_dir) == []
    assert len(run.argvs) == 1


def test_a_failed_run_is_retried_and_resumes(tmp_path: Path, models_dir: Path) -> None:
    messages: list[str] = []
    run = FakeRun(models_dir, ["fail", "ok"])
    ensurer(tmp_path, run, emit=messages.append).ensure(COMPONENTS["klein-9b"])
    assert len(run.argvs) == 2
    assert any("exited with 1; resuming (1/" in m for m in messages)


def test_a_silent_cli_is_stopped_and_started_again(tmp_path: Path, models_dir: Path) -> None:
    messages: list[str] = []
    run = FakeRun(models_dir, ["stall", "ok"])
    ensurer(tmp_path, run, emit=messages.append, stall_s=0.2, poll_s=0.05).ensure(
        COMPONENTS["z-image-turbo"]
    )
    assert len(run.argvs) == 2 and any("no progress; resuming" in m for m in messages)


def test_gives_up_after_the_restarts(tmp_path: Path, models_dir: Path) -> None:
    run = FakeRun(models_dir, ["fail", "fail", "fail"])
    with pytest.raises(SetupError, match="did not finish after 2 restarts"):
        ensurer(tmp_path, run, max_restarts=2).ensure(COMPONENTS["klein-4b"])


def test_a_file_still_missing_after_success_fails(tmp_path: Path, models_dir: Path) -> None:
    run = FakeRun(models_dir, ["partial"])
    with pytest.raises(SetupError, match="still missing after `models ensure`"):
        ensurer(tmp_path, run).ensure(COMPONENTS["klein-9b"])


def test_run_process_reports_partial_output_as_activity(tmp_path: Path) -> None:
    """The watchdog relies on this: \\r progress without a newline still counts."""
    chunks: list[int] = []
    lines: list[str] = []
    outcome = run_process(
        ["/bin/sh", "-c", "printf '\\r10%%'; sleep 0.2; printf '\\r20%%\\n'"],
        tmp_path,
        tmp_path / "log",
        timeout_s=10,
        cancel=threading.Event(),
        on_line=lines.append,
        on_output=lambda: chunks.append(1),
    )
    assert outcome.returncode == 0
    assert len(chunks) >= 2 and lines == ["\r10%\r20%"]
