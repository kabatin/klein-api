"""``setup``'s config writer: TOML emitter round trip, merge rules, LAN token, file modes."""

import stat
import tomllib
from pathlib import Path

import pytest

from klein_api.settings import load_settings
from klein_api.setup.config_write import (
    LAN_HOST,
    ConfigRequest,
    coord_token_env,
    dumps,
    write_config,
)
from klein_api.setup.errors import SetupError

TOKEN = "t" * 40


def test_the_emitter_round_trips_the_subset_used() -> None:
    data = {
        "port": 8767,
        "host": "127.0.0.1",
        "job_timeout_factor": 2.5,
        "resident": False,
        "names": ["a", 'b "quoted"', "ü\n"],
        "odd key": "x\\y\t",
        "backends": {"drawthings": {"cli": "~/x/draw-things-cli", "max_reference_images": 2}},
        "coordinate_with": [
            {"name": "vpipe-api", "health_url": "http://127.0.0.1:8765/v1/health"},
            {"name": "wan-api", "health_url": "http://127.0.0.1:8766/v1/health", "token": "p"},
        ],
    }
    assert tomllib.loads(dumps(data)) == data


@pytest.mark.parametrize("value", [float("inf"), {1, 2}, [1, 2], None])
def test_the_emitter_refuses_what_it_cannot_write(value: object) -> None:
    with pytest.raises(TypeError):
        dumps({"x": value})


def test_adds_engine_keys_and_keeps_everything_else(tmp_path: Path) -> None:
    path = tmp_path / "klein-api" / "config.toml"
    path.parent.mkdir()
    path.write_text(
        'port = 9000\nretention_days = 3\n[backends.drawthings]\ncli = "/opt/mine"\n'
        "resident = false\n"
    )
    data, changed = write_config(
        path, ConfigRequest(cli=Path("/new/cli"), models_dir="~/Models"), env={}
    )
    assert changed
    # an existing value wins; only the missing key is added
    assert data["backends"]["drawthings"] == {
        "cli": "/opt/mine",
        "resident": False,
        "models_dir": "~/Models",
    }
    assert (data["port"], data["retention_days"]) == (9000, 3)
    assert tomllib.loads(path.read_text()) == data
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # nothing new: the file is not rewritten
    assert write_config(path, ConfigRequest(models_dir="~/Models"), env={}) == (data, False)


def test_a_new_config_is_private_and_readable_by_the_server(tmp_path: Path) -> None:
    path = tmp_path / "fresh" / "config.toml"
    write_config(path, ConfigRequest(cli=Path("/x/cli"), port=8800), env={})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    settings = load_settings(env={}, config_path=path)
    assert settings.port == 8800 and settings.backend_options()["cli"] == "/x/cli"


def test_nothing_to_add_writes_no_file(tmp_path: Path) -> None:
    path = tmp_path / "none" / "config.toml"
    assert write_config(path, ConfigRequest(), env={}) == ({}, False)
    assert not path.exists()


def test_lan_generates_a_token_once_and_keeps_it(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    data, _ = write_config(path, ConfigRequest(lan=True), env={})
    assert data["host"] == LAN_HOST and len(data["token"]) >= 32
    again, changed = write_config(path, ConfigRequest(lan=True), env={})
    assert again["token"] == data["token"] and not changed
    assert load_settings(env={}, config_path=path).token is not None


def test_lan_takes_a_valid_token_from_the_environment(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    data, _ = write_config(path, ConfigRequest(lan=True), env={"KLEIN_API_TOKEN": TOKEN})
    assert data["token"] == TOKEN


def test_a_weak_token_from_the_environment_is_refused_without_echoing_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    with pytest.raises(SetupError) as exc:
        write_config(path, ConfigRequest(lan=True), env={"KLEIN_API_TOKEN": "short-secret"})
    assert "short-secret" not in str(exc.value)
    assert not path.exists()


def test_coordinate_with_adds_peers_once_with_their_token(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[[coordinate_with]]\nname = "vpipe-api"\nhealth_url = "http://a/v1/health"\n')
    peers = (("vpipe-api", "http://b/v1/health"), ("wan-api", "http://127.0.0.1:8766/v1/health"))
    env = {coord_token_env("wan-api"): "peer-token"}
    data, _ = write_config(path, ConfigRequest(coordinate_with=peers), env=env)
    assert data["coordinate_with"] == [
        {"name": "vpipe-api", "health_url": "http://a/v1/health"},
        {"name": "wan-api", "health_url": "http://127.0.0.1:8766/v1/health", "token": "peer-token"},
    ]
    assert coord_token_env("wan-api") == "KLEIN_API_COORD_TOKEN_WAN_API"
    settings = load_settings(env={}, config_path=path)
    assert [p.name for p in settings.coordinate_with] == ["vpipe-api", "wan-api"]


def test_a_bad_peer_url_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SetupError, match="ftp-peer"):
        write_config(
            tmp_path / "c.toml",
            ConfigRequest(coordinate_with=(("ftp-peer", "ftp://x/health"),)),
            env={},
        )


def test_a_broken_config_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("port = = 1")
    with pytest.raises(SetupError, match="cannot read config file"):
        write_config(path, ConfigRequest(port=1), env={})
