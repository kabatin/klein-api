"""``klein-api setup`` end to end with fakes: no network, no real Models folder, no launchctl.

Each test gets its own config file pointing ``models_dir`` and ``data_dir`` into tmp_path."""

import hashlib
import json
import plistlib
import threading
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from klein_api.cli import main
from klein_api.setup import command, engine
from klein_api.setup.command import Deps, SetupArgs, run_setup
from klein_api.setup.machine import Machine
from klein_api.setup.pins import COMPONENT_IDS, COMPONENTS, FILE_BYTES, Asset

CLI_BYTES = b"cli" * 50
GRPC_BYTES = b"grpc" * 50
MAC = Machine(arm64=True, macos="26.0", ram_gb=32, free_bytes=10**12, tools={"launchctl": True})


@pytest.fixture(autouse=True)
def small_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for attr, name, content in (
        ("CLI_ASSET", "draw-things-cli", CLI_BYTES),
        ("GRPC_ASSET", "gRPCServerCLI-macOS", GRPC_BYTES),
    ):
        digest = hashlib.sha256(content).hexdigest()
        url = f"https://x.invalid/{name}"
        monkeypatch.setattr(engine, attr, Asset(name, url, digest, len(content)))
    monkeypatch.setattr(engine, "DEFAULT_CLI", str(tmp_path / "default" / "draw-things-cli"))


class World:
    """The fake machine: what got downloaded, ensured and launched."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.models = tmp_path / "Models"
        self.config = tmp_path / "config" / "config.toml"
        self.config.parent.mkdir()
        self.config.write_text(
            f'data_dir = "{tmp_path / "data"}"\n'
            f'[backends.drawthings]\nmodels_dir = "{self.models}"\n'
        )
        self.downloads: list[str] = []
        self.ensured: list[str] = []
        self.launchctl: list[list[str]] = []
        self.messages: list[str] = []
        self.answers: list[str] = []
        self.health = 200

    def download(self, url: str, part: Path, size: int, emit) -> None:
        self.downloads.append(url)
        part.write_bytes(CLI_BYTES if url.endswith("/draw-things-cli") else GRPC_BYTES)

    def run_ensure(self, argv, log_path, cancel: threading.Event, on_line, on_activity) -> int:
        component = next(c for c in COMPONENTS.values() if c.model_file == argv[4])
        self.ensured.append(component.id)
        assert argv[argv.index("--models-dir") + 1] == str(self.models)
        self.models.mkdir(exist_ok=True)
        for name in component.files:
            (self.models / name).write_bytes(b"x")
        return 0

    def deps(self, **overrides) -> Deps:
        base = Deps(
            download=self.download,
            run_ensure=self.run_ensure,
            launchctl=lambda cmd: self.launchctl.append(list(cmd)) or 0,
            which=lambda name: str(self.tmp / "bin" / name),
            http_get=lambda url: self.health,
            machine=lambda path: MAC,
            is_tty=lambda: False,
            ask=lambda prompt: self.answers.pop(0),
            env={},
            home=self.tmp / "home",
            uid=501,
            emit=self.messages.append,
            health_wait_s=0,
        )
        return replace(base, **overrides)

    def args(self, *models: str, **overrides) -> SetupArgs:
        base = SetupArgs(
            models=tuple(models or COMPONENT_IDS),  # type: ignore[arg-type]
            config_path=self.config,
            config_is_default=False,
            prefix=self.tmp / "prefix",
            yes=True,
        )
        return replace(base, **overrides)

    def install_models(self, *ids: str) -> None:
        self.models.mkdir(exist_ok=True)
        for model in ids:
            for name in COMPONENTS[model].files:  # type: ignore[index]
                (self.models / name).write_bytes(b"x")


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def rows(report: dict) -> dict[str, dict]:
    return {row["id"]: row for row in report["components"]}


def tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def test_check_reports_sizes_and_changes_nothing(world: World) -> None:
    before = tree(world.tmp)
    code, report = run_setup(world.args(check=True), world.deps())
    assert code == 0 and report["action"] == "check" and report["ok"]
    assert tree(world.tmp) == before and not world.downloads and not world.ensured
    by_id = rows(report)
    assert all(row["status"] == "missing" for row in by_id.values())
    engine_bytes = len(CLI_BYTES) + len(GRPC_BYTES)
    klein_9b = sum(FILE_BYTES[n] for n in COMPONENTS["klein-9b"].files)
    assert by_id["klein-9b"]["downloadBytes"] == klein_9b + engine_bytes
    # the shared VAE and the engine are counted once
    vae = FILE_BYTES["flux_2_vae_f16.ckpt"]
    klein_4b = sum(FILE_BYTES[n] for n in COMPONENTS["klein-4b"].files)
    assert by_id["klein-4b"]["downloadBytes"] == klein_4b - vae
    assert by_id["z-image-turbo"]["downloadBytes"] == 10_891_780_096
    assert report["machine"]["ramGb"] == 32 and report["service"]["installed"] is False
    assert report["url"] == "http://127.0.0.1:8767" and report["tokenSet"] is False


def test_install_downloads_ensures_and_writes_the_config(world: World) -> None:
    code, report = run_setup(world.args("klein-4b", "z-image-turbo"), world.deps())
    assert code == 0 and report["ok"], report
    assert {r["status"] for r in report["components"]} == {"installed"}
    assert world.ensured == ["klein-4b", "z-image-turbo"] and len(world.downloads) == 2
    config = tomllib.loads(world.config.read_text())
    cli = world.tmp / "prefix" / "draw-things-cli" / "v26.1011.0" / "draw-things-cli"
    assert config["backends"]["drawthings"] == {"models_dir": str(world.models), "cli": str(cli)}
    assert cli.read_bytes() == CLI_BYTES
    # idempotent: a second run downloads nothing and reports `already`
    code, again = run_setup(world.args("klein-4b", "z-image-turbo"), world.deps())
    assert code == 0 and {r["status"] for r in again["components"]} == {"already"}
    assert world.ensured == ["klein-4b", "z-image-turbo"] and len(world.downloads) == 2


def test_without_yes_and_without_a_terminal_nothing_happens(world: World) -> None:
    code, report = run_setup(world.args("klein-4b", yes=False), world.deps())
    assert code == 3 and not world.downloads and not world.ensured
    fix = rows(report)["klein-4b"]["fix"]
    assert fix["command"] == "klein-api setup --models klein-4b --yes"
    assert fix["needsHuman"] is False


def test_a_terminal_is_asked_once_with_the_licences(world: World) -> None:
    world.answers = ["n"]
    code, _ = run_setup(world.args(yes=False), world.deps(is_tty=lambda: True))
    assert code == 3 and not world.downloads
    assert any("non-commercial" in m for m in world.messages)
    world.answers = ["y"]
    code, report = run_setup(world.args("klein-4b", yes=False), world.deps(is_tty=lambda: True))
    assert code == 0 and rows(report)["klein-4b"]["status"] == "installed"


def test_nothing_to_download_needs_no_confirmation(world: World) -> None:
    run_setup(world.args("klein-4b"), world.deps())
    code, report = run_setup(world.args("klein-4b", yes=False), world.deps())
    assert code == 0 and rows(report)["klein-4b"]["status"] == "already"


def test_too_little_disk_fails_before_downloading(world: World) -> None:
    small = replace(MAC, free_bytes=5_000_000_000)
    code, report = run_setup(world.args("klein-9b"), world.deps(machine=lambda _: small))
    row = rows(report)["klein-9b"]
    assert code == 1 and row["status"] == "failed" and "free disk space" in row["fix"]["hint"]
    assert not world.downloads and not report["ok"]


def test_unsupported_machines_and_memory(world: World) -> None:
    intel = replace(MAC, arm64=False)
    _, report = run_setup(world.args(check=True), world.deps(machine=lambda _: intel))
    assert {r["status"] for r in report["components"]} == {"unsupported"}
    small = replace(MAC, ram_gb=8)
    _, report = run_setup(world.args(check=True), world.deps(machine=lambda _: small))
    by_id = rows(report)
    assert by_id["klein-9b"]["status"] == "unsupported" and "16 GB" in by_id["klein-9b"]["message"]
    assert by_id["klein-4b"]["status"] == "missing"


def test_a_hand_configured_old_cli_needs_a_human(world: World) -> None:
    old = world.tmp / "brew" / "draw-things-cli"
    old.parent.mkdir()
    old.write_bytes(b"1.20260430.0")
    world.config.write_text(world.config.read_text() + f'cli = "{old}"\n')
    code, report = run_setup(world.args("klein-4b"), world.deps())
    row = rows(report)["klein-4b"]
    assert code == 3 and row["status"] == "needs_human" and row["fix"]["needsHuman"]
    assert str(old) in row["fix"]["hint"] and not world.downloads
    assert old.read_bytes() == b"1.20260430.0"


def test_a_failed_ensure_is_reported(world: World) -> None:
    def broken(argv, log_path, cancel, on_line, on_activity) -> int:
        return 1

    code, report = run_setup(world.args("klein-4b"), world.deps(run_ensure=broken))
    row = rows(report)["klein-4b"]
    assert code == 1 and row["status"] == "failed" and "did not finish" in row["message"]


def test_service_writes_the_agent_waits_for_health_and_asks_for_the_dialog(
    world: World,
) -> None:
    code, report = run_setup(
        world.args("klein-4b", service=True, label_suffix="test", port=8799), world.deps()
    )
    assert code == 3 and report["ok"]
    assert report["service"] == {
        "installed": True,
        "label": "com.github.kabatin.klein-api.test",
        "healthy": True,
    }
    row = rows(report)["klein-4b"]
    assert row["status"] == "needs_human" and "approve the dialog" in row["fix"]["hint"]
    plist_path = world.tmp / "home/Library/LaunchAgents/com.github.kabatin.klein-api.test.plist"
    agent = plistlib.loads(plist_path.read_bytes())
    assert agent["ProgramArguments"] == [str((world.tmp / "bin/klein-api").resolve()), "serve"]
    assert agent["EnvironmentVariables"]["KLEIN_API_CONFIG"] == str(world.config)
    assert agent["KeepAlive"] == {"SuccessfulExit": False} and agent["RunAtLoad"] is True
    assert agent["StandardErrorPath"].endswith("Library/Logs/klein-api.test.log")
    assert world.launchctl == [
        ["launchctl", "bootout", "gui/501/com.github.kabatin.klein-api.test"],
        ["launchctl", "bootstrap", "gui/501", str(plist_path)],
    ]
    assert report["url"] == "http://127.0.0.1:8799"


def test_a_service_that_never_answers_fails(world: World) -> None:
    world.health = 0
    code, report = run_setup(world.args("klein-4b", service=True), world.deps())
    assert code == 1 and report["service"]["healthy"] is False and "no answer" in report["error"]


def test_lan_sets_a_token_that_is_never_reported(world: World) -> None:
    code, report = run_setup(world.args("klein-4b", lan=True), world.deps())
    token = tomllib.loads(world.config.read_text())["token"]
    assert code == 0 and report["lan"] and report["tokenSet"]
    text = json.dumps(report) + "\n".join(world.messages)
    assert token not in text


def test_cli_rejects_unknown_ids_with_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["setup", "--models", "klein-9b,flux-pro"])
    assert exc.value.code == 2
    assert "valid: klein-9b, klein-4b, z-image-turbo" in capsys.readouterr().err


def test_cli_json_is_one_object_on_the_last_stdout_line(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original = command.Deps
    monkeypatch.setattr(
        command, "Deps", lambda: replace(original(), machine=lambda p: MAC, http_get=lambda u: None)
    )
    code = main(
        ["--config", str(world.config), "setup", "--check", "--json", "--models", "klein-4b"]
    )
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 0 and len(out) == 1
    report = json.loads(out[-1])
    assert report["components"][0]["id"] == "klein-4b" and report["action"] == "check"


def test_cli_human_output(world: World, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    original = command.Deps
    monkeypatch.setattr(
        command, "Deps", lambda: replace(original(), machine=lambda p: MAC, http_get=lambda u: None)
    )
    assert main(["--config", str(world.config), "setup", "--check"]) == 0
    out = capsys.readouterr().out
    assert "klein-9b" in out and "missing" in out and "config:" in out


def test_machine_support_rules(tmp_path: Path) -> None:
    assert MAC.unsupported() is None
    assert "macOS 14.0+" in (replace(MAC, macos="13.6").unsupported() or "")
    assert replace(MAC, macos="").unsupported() == "needs an Apple Silicon Mac"
    facts = command.machine_facts(tmp_path / "missing" / "Models")
    assert facts.free_bytes > 0 and set(facts.as_json()) == {
        "arm64",
        "macos",
        "ramGb",
        "freeBytes",
        "tools",
    }
