import logging
import subprocess
from pathlib import Path

import pytest

from klein_api import cli, doctor
from klein_api.checks import Check, Level
from klein_api.doctor import check_disk, check_macos, check_memory, check_power, run_doctor
from klein_api.jobs.store import JobStore
from klein_api.logs import JsonFormatter, event
from klein_api.settings import Settings, load_settings
from klein_api.workflows import build_registry
from tests.fakes import FakeKleinBackend

TOKEN = "test-token-0123456789-abcdefghijklmnop"


def fake_run(outputs: dict[str, str]):
    def run(cmd: list[str], timeout: float = 20):
        for key, stdout in outputs.items():
            if key in cmd[0]:
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        return None

    return run


@pytest.fixture
def healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI also runs on Linux: pretend to be an Apple Silicon Mac
    monkeypatch.setattr(doctor, "check_platform", lambda: Check("platform", Level.OK, ""))
    monkeypatch.setattr(doctor, "check_macos", lambda: Check("macOS", Level.OK, "26.5"))
    monkeypatch.setattr(
        doctor,
        "_run",
        fake_run(
            {
                "pmset": "Now drawing from 'AC Power'",
                "memory_pressure": "System-wide memory free percentage: 64%",
            }
        ),
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings(
        env={"KLEIN_API_DATA_DIR": str(tmp_path / "data")}, config_path=tmp_path / "none.toml"
    )


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeKleinBackend:
    monkeypatch.setenv("KLEIN_API_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("KLEIN_API_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("KLEIN_API_TOKEN", TOKEN)
    backend = FakeKleinBackend()
    monkeypatch.setattr(cli, "build_backend", lambda _settings: backend)
    return backend


# -- doctor ----------------------------------------------------------------------------


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


def test_doctor_reports_backend_health(settings: Settings, healthy: None) -> None:
    emitted: list[Check] = []
    backend = FakeKleinBackend()
    checks = run_doctor(settings, backend, build_registry(backend), emit=emitted.append)
    assert emitted == checks
    names = by_name(checks)
    assert {"python", "fake backend", "disk", "power", "memory"} <= set(names)
    assert names["fake backend"].level is Level.OK
    backend.healthy = False
    assert by_name(run_doctor(settings, backend, build_registry(backend)))["fake backend"].failed


def test_macos_version_check() -> None:
    assert check_macos("26.5").level is Level.OK
    assert check_macos("13.6").level is Level.FAIL
    assert check_macos("").level is Level.WARN


def test_disk_check_walks_up_to_an_existing_dir(tmp_path: Path) -> None:
    check = check_disk(tmp_path / "not" / "yet" / "there")
    assert str(tmp_path) in check.detail


def test_power_and_memory_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor,
        "_run",
        fake_run(
            {
                "pmset": "Now drawing from 'Battery Power'",
                "memory_pressure": "System-wide memory free percentage: 5%",
            }
        ),
    )
    assert check_power().level is Level.WARN
    assert check_memory().level is Level.WARN
    monkeypatch.setattr(doctor, "_run", fake_run({}))
    assert check_power().level is Level.WARN and check_memory().level is Level.WARN


def test_peer_checks_never_fail(
    settings: Settings, healthy: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klein_api.jobs.coordination import PeerHealth
    from klein_api.settings import PeerServer

    peer = PeerServer(name="vpipe-api", health_url="http://127.0.0.1:1/v1/health")
    with_peer = settings.model_copy(update={"coordinate_with": (peer,)})
    backend = FakeKleinBackend()
    check = by_name(run_doctor(with_peer, backend, build_registry(backend)))
    assert check["coordinate with vpipe-api"].level is Level.WARN
    assert "count as idle" in check["coordinate with vpipe-api"].detail

    def busy(_peer: PeerServer) -> PeerHealth:
        return PeerHealth(running=1, waiting=2)

    monkeypatch.setattr(doctor, "probe_health", busy)
    assert doctor.check_peer(peer).detail == "busy (running 1, waiting 2)"


def test_smoke_runs_a_short_job(settings: Settings, healthy: None) -> None:
    backend = FakeKleinBackend(png=True, size=(1024, 1024))
    checks = by_name(run_doctor(settings, backend, build_registry(backend), smoke=True))
    smoke = checks["smoke flux2-klein"]
    assert smoke.level is Level.OK and "1024x1024 PNG" in smoke.detail
    assert backend.calls[0].quality == "draft" and backend.calls[0].count == 1
    assert not any((settings.data_dir / "smoke").iterdir())
    assert checks["smoke z-image-turbo"].level is Level.OK
    assert backend.calls[1].model_id == "z-image-turbo"


def test_smoke_skips_a_workflow_whose_model_is_missing(settings: Settings, healthy: None) -> None:
    backend = FakeKleinBackend(png=True, size=(1024, 1024))
    backend.problems = {"z-image-turbo": "model files missing: z_image_turbo_1.0_i8x.ckpt"}
    checks = by_name(run_doctor(settings, backend, build_registry(backend), smoke=True))
    skipped = checks["smoke z-image-turbo"]
    assert skipped.level is Level.WARN and "z_image_turbo_1.0_i8x.ckpt" in skipped.detail
    assert checks["smoke flux2-klein"].level is Level.OK and len(backend.calls) == 1


def test_smoke_failure_is_reported(settings: Settings, healthy: None) -> None:
    backend = FakeKleinBackend()
    backend.mode = "fail"
    smoke = by_name(run_doctor(settings, backend, build_registry(backend), smoke=True))[
        "smoke flux2-klein"
    ]
    assert smoke.failed and "engine failed" in smoke.detail


def test_smoke_is_skipped_when_checks_fail(settings: Settings, healthy: None) -> None:
    backend = FakeKleinBackend()
    backend.healthy = False
    checks = run_doctor(settings, backend, build_registry(backend), smoke=True)
    assert not backend.calls and "smoke flux2-klein" not in by_name(checks)


# -- CLI -------------------------------------------------------------------------------


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "klein-api 0.1.1" in capsys.readouterr().out


def test_workflows_and_config_hide_the_token(
    env: FakeKleinBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["workflows"]) == 0
    assert "flux2-klein" in capsys.readouterr().out
    assert cli.main(["config"]) == 0
    out = capsys.readouterr().out
    assert "token = ********" in out and TOKEN not in out and "port = 8767" in out


def test_doctor_exit_codes(
    env: FakeKleinBackend, healthy: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["doctor"]) == 0
    assert "all good" in capsys.readouterr().out
    env.healthy = False
    assert cli.main(["doctor"]) == 1


def test_settings_errors_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("KLEIN_API_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("KLEIN_API_HOST", "0.0.0.0")
    monkeypatch.delenv("KLEIN_API_TOKEN", raising=False)
    assert cli.main(["config"]) == 1
    assert "KLEIN_API_TOKEN" in capsys.readouterr().err


def test_serve_wires_uvicorn(env: FakeKleinBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_run(app, **kwargs):
        seen.update(kwargs, app=app)

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert cli.main(["serve", "--port", "9999"]) == 0
    assert seen["port"] == 9999 and seen["workers"] == 1 and seen["host"] == "127.0.0.1"
    assert seen["log_config"] is None


def test_serve_refuses_remote_bind_without_token(
    env: FakeKleinBackend, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("KLEIN_API_TOKEN")
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("must not start"))
    assert cli.main(["serve", "--host", "0.0.0.0"]) == 1
    assert "KLEIN_API_TOKEN" in capsys.readouterr().err


def test_serve_fails_early_when_the_backend_cannot_generate(
    env: FakeKleinBackend, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env.healthy = False
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("must not start"))
    assert cli.main(["serve"]) == 1
    err = capsys.readouterr().err
    assert "cannot generate" in err and "model missing" in err


def test_serve_refuses_second_instance(
    env: FakeKleinBackend, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    holder = JobStore(tmp_path / "data")
    holder.acquire_instance_lock()
    try:
        assert cli.main(["serve"]) == 1
        assert "already" in capsys.readouterr().err
    finally:
        holder.release_instance_lock()


# -- structured logs ---------------------------------------------------------------------


def test_json_log_lines() -> None:
    import json

    records: list[logging.LogRecord] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("klein_api.test")
    logger.addHandler(Keep())
    logger.setLevel(logging.INFO)
    event(logger, "job.succeeded", job_id="job_1", total_seconds=3.5)
    line = json.loads(JsonFormatter().format(records[0]))
    assert line["event"] == "job.succeeded" and line["job_id"] == "job_1"
    assert line["total_seconds"] == 3.5 and line["level"] == "info"


def test_build_backend_is_draw_things(tmp_path: Path) -> None:
    from klein_api.backends import build_backend
    from klein_api.backends.drawthings import DrawThingsCliBackend

    base = {"KLEIN_API_DATA_DIR": str(tmp_path)}
    default = build_backend(load_settings(env=base, config_path=tmp_path / "x"))
    assert isinstance(default, DrawThingsCliBackend) and default.max_reference_images == 2


def test_smoke_refuses_while_a_server_holds_the_data_dir(settings: Settings, healthy: None) -> None:
    holder = JobStore(settings.data_dir)
    holder.acquire_instance_lock()
    try:
        backend = FakeKleinBackend()
        smoke = by_name(run_doctor(settings, backend, build_registry(backend), smoke=True))[
            "smoke flux2-klein"
        ]
        assert smoke.failed and "server is running" in smoke.detail and not backend.calls
    finally:
        holder.release_instance_lock()


@pytest.mark.parametrize("host", ["::", "0.0.0.0", "192.168.0.2", "mac.local"])
def test_serve_host_override_needs_a_token(
    env: FakeKleinBackend,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    host: str,
) -> None:
    monkeypatch.delenv("KLEIN_API_TOKEN")
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("must not start"))
    assert cli.main(["serve", "--host", host]) == 1
    assert "KLEIN_API_TOKEN" in capsys.readouterr().err


def test_serve_and_config_with_peers(
    env: FakeKleinBackend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "peers.toml"
    cfg.write_text(
        '[[coordinate_with]]\nname = "vpipe-api"\n'
        'health_url = "http://127.0.0.1:8765/v1/health"\ntoken = "peer-secret-value"\n'
    )
    cfg.chmod(0o600)
    monkeypatch.setenv("KLEIN_API_CONFIG", str(cfg))
    seen: dict = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(kw, app=app))
    assert cli.main(["serve"]) == 0
    out = capsys.readouterr().out
    assert "waits for: vpipe-api" in out and "peer-secret-value" not in out
    assert env.peer_check is not None  # the engine is told when a peer gets busy
    assert cli.main(["config"]) == 0
    out = capsys.readouterr().out
    assert "name = vpipe-api" in out and "token = ********" in out
    assert "peer-secret-value" not in out
