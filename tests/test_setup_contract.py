"""``klein-api setup`` rules shared by all five servers: exit codes and ``ok``, statuses,
argument-time validation, JSON on usage errors, launchd reloads, prompts on stderr."""

import io
import json
import plistlib
import sys
import tomllib
from dataclasses import replace

import pytest

from klein_api.cli import main
from klein_api.setup import command
from klein_api.setup.command import run_setup
from tests.setup_world import MAC, World, rows

# never the real home: these tests also run setup through the CLI
pytestmark = pytest.mark.usefixtures("isolated_home")

REPORT_KEYS = {
    "server",
    "version",
    "ok",
    "action",
    "url",
    "lan",
    "tokenSet",
    "configPath",
    "service",
    "machine",
    "components",
}


def test_exit_3_is_not_ok(world: World) -> None:
    code, report = run_setup(world.args("klein-4b", yes=False), world.deps())
    assert code == 3 and report["ok"] is False


def test_check_reports_present_components_as_installed(world: World) -> None:
    world.install_models("klein-4b")
    run_setup(world.args("klein-4b"), world.deps())  # installs the engine
    _, report = run_setup(world.args("klein-4b", check=True), world.deps())
    assert rows(report)["klein-4b"]["status"] == "installed"
    _, again = run_setup(world.args("klein-4b"), world.deps())
    assert rows(again)["klein-4b"]["status"] == "already"


def test_unsupported_in_the_default_selection_does_not_fail(world: World) -> None:
    small = replace(MAC, ram_gb=8)  # too little for the 9B only
    deps = world.deps(machine=lambda _: small)
    code, report = run_setup(world.args(), deps)
    assert code == 0 and report["ok"], report
    assert rows(report)["klein-9b"]["status"] == "unsupported"
    assert rows(report)["klein-4b"]["status"] == "installed"


def test_an_explicitly_requested_unsupported_component_fails(world: World) -> None:
    small = replace(MAC, ram_gb=8)
    for check in (True, False):
        code, report = run_setup(
            world.args("klein-9b", explicit=True, check=check),
            world.deps(machine=lambda _: small),
        )
        assert code == 1 and report["ok"] is False


def run_cli(argv: list[str], capsys) -> tuple[int, str, str]:
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


@pytest.mark.parametrize(
    "argv",
    [
        ["setup", "--json", "--models", "nope"],
        ["setup", "--json", "--port", "70000"],
        ["setup", "--json", "--no-such-flag"],
        ["setup", "--json", "--label-suffix", "bad suffix!", "--port", "8800"],
        ["setup", "--json", "--label-suffix", "test"],  # needs --port
        ["setup", "--json", "--coordinate-with", "wan-api=ftp://x/health"],
    ],
)
def test_usage_errors_with_json_still_print_one_report(argv: list[str], capsys) -> None:
    code, out, err = run_cli(argv, capsys)
    lines = out.strip().splitlines()
    assert code == 2 and len(lines) == 1, (out, err)
    report = json.loads(lines[0])
    assert set(report) >= REPORT_KEYS and report["ok"] is False and report["error"]
    assert report["components"] == []


def test_usage_errors_without_json_go_to_stderr(capsys) -> None:
    code, out, err = run_cli(["setup", "--label-suffix", "x"], capsys)
    assert code == 2 and out == "" and "--port" in err


def test_a_weak_lan_token_is_refused_before_anything_runs(
    world: World, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KLEIN_API_TOKEN", "short-secret")
    code, out, _ = run_cli(["--config", str(world.config), "setup", "--json", "--lan"], capsys)
    report = json.loads(out.strip().splitlines()[-1])
    assert code == 2 and "KLEIN_API_TOKEN" in report["error"]
    assert "short-secret" not in out


def test_a_bad_peer_token_is_refused_at_argument_time(
    world: World, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KLEIN_API_COORD_TOKEN_WAN_API", "has space")
    argv = ["--config", str(world.config), "setup", "--json", "--coordinate-with"]
    code, out, _ = run_cli([*argv, "wan-api=http://127.0.0.1:8766/v1/health"], capsys)
    assert code == 2 and "has space" not in out


def test_an_existing_peer_gets_its_new_url_and_token(world: World) -> None:
    world.config.write_text(
        world.config.read_text()
        + '[[coordinate_with]]\nname = "wan-api"\nhealth_url = "http://old/v1/health"\n'
    )
    peers = (("wan-api", "http://127.0.0.1:8766/v1/health"),)
    env = {"KLEIN_API_COORD_TOKEN_WAN_API": "new-token"}
    run_setup(world.args("klein-4b", coordinate_with=peers), world.deps(env=env))
    config = tomllib.loads(world.config.read_text())
    assert config["coordinate_with"] == [
        {"name": "wan-api", "health_url": "http://127.0.0.1:8766/v1/health", "token": "new-token"}
    ]


def test_setup_runs_on_a_config_the_server_would_reject(world: World) -> None:
    world.config.write_text(
        world.config.read_text().replace("data_dir", "max_waiting = 999\ndata_dir")
    )
    code, report = run_setup(world.args("klein-4b"), world.deps())
    assert code == 0 and report["ok"], report
    config = tomllib.loads(world.config.read_text())
    assert config["max_waiting"] == 999 and "cli" in config["backends"]["drawthings"]


def test_an_unparseable_config_is_a_json_failure(world: World, capsys) -> None:
    world.config.write_text("port = = 1\n")
    code, out, _ = run_cli(["--config", str(world.config), "setup", "--json", "--check"], capsys)
    report = json.loads(out.strip().splitlines()[-1])
    assert code == 1 and report["ok"] is False and "cannot read config" in report["error"]


def test_a_symlinked_config_is_written_through_to_its_target(world: World) -> None:
    target = world.tmp / "dotfiles" / "klein.toml"
    target.parent.mkdir()
    target.write_text(world.config.read_text())
    world.config.unlink()
    world.config.symlink_to(target)
    run_setup(world.args("klein-4b"), world.deps())
    assert world.config.is_symlink()
    assert "cli" in tomllib.loads(target.read_text())["backends"]["drawthings"]


def test_reload_waits_for_bootout_and_retries_a_busy_bootstrap(world: World) -> None:
    world.print_rc = [0, 0, 113]  # still loaded twice after bootout, then gone
    world.bootstrap_rc = [5, 37, 0]  # EIO, EALREADY, then it works
    code, report = run_setup(world.args("klein-4b", service=True), world.deps())
    assert code == 3 and report["service"]["healthy"], report
    verbs = [cmd[1] for cmd in world.launchctl]
    assert verbs == ["bootout", "print", "print", "print", "bootstrap", "bootstrap", "bootstrap"]


def test_bootstrap_gives_up_on_other_errors(world: World) -> None:
    world.bootstrap_rc = [1]
    code, report = run_setup(world.args("klein-4b", service=True), world.deps())
    assert code == 1 and "bootstrap" in report["error"]


def test_an_unchanged_service_is_not_a_human_step_again(world: World) -> None:
    args = world.args("klein-4b", service=True)
    assert run_setup(args, world.deps())[0] == 3
    code, report = run_setup(args, world.deps())
    assert code == 0 and report["ok"] and rows(report)["klein-4b"]["status"] == "already"
    assert any("unchanged" in m for m in world.messages)
    plist = world.tmp / "home/Library/LaunchAgents/com.github.kabatin.klein-api.plist"
    assert plistlib.loads(plist.read_bytes())["Label"] == "com.github.kabatin.klein-api"
    assert not list(plist.parent.glob("*.tmp"))


def test_the_confirmation_prompt_goes_to_stderr(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    assert command.ask_on_stderr("Continue? [y/N] ") == "y"
    out, err = capsys.readouterr()
    assert out == "" and "Continue?" in err
