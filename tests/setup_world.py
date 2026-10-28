"""A fake machine for ``klein-api setup`` tests: no network, no real Models folder, no
launchctl. Each test gets its own config file pointing ``models_dir`` and ``data_dir`` into
tmp_path."""

import hashlib
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from klein_api.setup import engine
from klein_api.setup.command import Deps, SetupArgs
from klein_api.setup.machine import Machine
from klein_api.setup.pins import COMPONENT_IDS, COMPONENTS, Asset

CLI_BYTES = b"cli" * 50
GRPC_BYTES = b"grpc" * 50
MAC = Machine(arm64=True, macos="26.0", ram_gb=32, free_bytes=10**12, tools={"launchctl": True})


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
        # scripted exit codes: `launchctl print` (113 = not loaded) and `bootstrap`
        self.print_rc: list[int] = []
        self.bootstrap_rc: list[int] = []
        self.messages: list[str] = []
        self.answers: list[str] = []
        self.health = 200
        self.tokens: list[str | None] = []

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

    def run_launchctl(self, cmd) -> int:
        self.launchctl.append(list(cmd))
        if cmd[1] == "print":
            return self.print_rc.pop(0) if self.print_rc else 113
        if cmd[1] == "bootstrap":
            return self.bootstrap_rc.pop(0) if self.bootstrap_rc else 0
        return 0

    def http_get(self, url: str, token: str | None) -> int:
        self.tokens.append(token)
        return self.health

    def deps(self, **overrides) -> Deps:
        base = Deps(
            download=self.download,
            run_ensure=self.run_ensure,
            launchctl=self.run_launchctl,
            which=lambda name: str(self.tmp / "bin" / name),
            http_get=self.http_get,
            machine=lambda path: MAC,
            is_tty=lambda: False,
            ask=lambda prompt: self.answers.pop(0),
            env={},
            home=self.tmp / "home",
            uid=501,
            emit=self.messages.append,
            health_wait_s=0,
            launchd_poll_s=0,
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


def rows(report: dict) -> dict[str, dict]:
    return {row["id"]: row for row in report["components"]}


def tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def use_small_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tiny fake release assets, and a default CLI path inside tmp_path."""
    for attr, name, content in (
        ("CLI_ASSET", "draw-things-cli", CLI_BYTES),
        ("GRPC_ASSET", "gRPCServerCLI-macOS", GRPC_BYTES),
    ):
        digest = hashlib.sha256(content).hexdigest()
        url = f"https://x.invalid/{name}"
        monkeypatch.setattr(engine, attr, Asset(name, url, digest, len(content)))
    monkeypatch.setattr(engine, "DEFAULT_CLI", str(tmp_path / "default" / "draw-things-cli"))
