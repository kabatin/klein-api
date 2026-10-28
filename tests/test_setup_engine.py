"""``setup``'s engine step with small fake release assets: detection by SHA-256, install,
refusing to replace a hand-configured CLI, and the resumable HTTP download (no network)."""

import hashlib
import io
from pathlib import Path

import pytest

from klein_api.setup import engine
from klein_api.setup.engine import (
    EngineState,
    http_download,
    inspect_engine,
    install_engine,
    prefix_cli,
)
from klein_api.setup.errors import SetupError
from klein_api.setup.pins import CLI_VERSION, Asset

CLI_BYTES = b"cli-binary" * 10
GRPC_BYTES = b"grpc-binary" * 10


def asset(name: str, content: bytes) -> Asset:
    return Asset(
        name, f"https://example.invalid/{name}", hashlib.sha256(content).hexdigest(), len(content)
    )


@pytest.fixture(autouse=True)
def small_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(engine, "CLI_ASSET", asset("draw-things-cli", CLI_BYTES))
    monkeypatch.setattr(engine, "GRPC_ASSET", asset("gRPCServerCLI-macOS", GRPC_BYTES))
    monkeypatch.setattr(engine, "DEFAULT_CLI", str(tmp_path / "default" / "draw-things-cli"))


def fake_download(served: dict[str, bytes], calls: list[str]):
    def download(url: str, part: Path, size: int, emit) -> None:
        calls.append(url)
        part.write_bytes(served[url.rsplit("/", 1)[1]])

    return download


def test_nothing_installed_means_the_prefix_and_its_download_size(tmp_path: Path) -> None:
    state = inspect_engine(None, tmp_path / "prefix")
    assert state.cli == tmp_path / "prefix" / "draw-things-cli" / CLI_VERSION / "draw-things-cli"
    assert not state.ready and state.download_bytes == len(CLI_BYTES) + len(GRPC_BYTES)


def test_install_verifies_and_places_both_binaries(tmp_path: Path) -> None:
    calls: list[str] = []
    served = {"draw-things-cli": CLI_BYTES, "gRPCServerCLI-macOS": GRPC_BYTES}
    state = inspect_engine(None, tmp_path / "prefix")
    done = install_engine(state, download=fake_download(served, calls), emit=lambda _: None)
    assert done.ready and done.grpc_ok and len(calls) == 2
    assert done.cli.read_bytes() == CLI_BYTES and done.cli.stat().st_mode & 0o111
    assert done.grpc.read_bytes() == GRPC_BYTES
    assert not list(done.cli.parent.glob("*.part"))
    # found again on the next run: nothing to download
    again = inspect_engine(None, tmp_path / "prefix")
    assert again.ready and again.grpc_ok and again.download_bytes == 0


def test_a_tampered_download_is_deleted_and_fails(tmp_path: Path) -> None:
    served = {"draw-things-cli": b"evil", "gRPCServerCLI-macOS": GRPC_BYTES}
    state = inspect_engine(None, tmp_path / "prefix")
    with pytest.raises(SetupError, match="SHA-256"):
        install_engine(state, download=fake_download(served, []), emit=lambda _: None)
    assert not state.cli.exists() and not list(state.cli.parent.glob("*.part"))


def test_the_default_location_is_used_when_it_holds_the_pinned_release(tmp_path: Path) -> None:
    default = Path(engine.DEFAULT_CLI)
    default.parent.mkdir(parents=True)
    default.write_bytes(CLI_BYTES)
    state = inspect_engine(None, tmp_path / "prefix")
    assert state.cli == default and state.ready and not state.grpc_ok
    assert state.download_bytes == len(GRPC_BYTES)
    # an older CLI there (e.g. another release) is not ours: install into the prefix
    default.write_bytes(b"older release")
    assert inspect_engine(None, tmp_path / "prefix").cli == prefix_cli(tmp_path / "prefix")


@pytest.mark.parametrize("content", [None, b"homebrew 1.20260430.0"])
def test_a_configured_cli_that_is_not_the_pin_needs_a_human(
    tmp_path: Path, content: bytes | None
) -> None:
    configured = tmp_path / "opt" / "draw-things-cli"
    if content is not None:
        configured.parent.mkdir()
        configured.write_bytes(content)
    state = inspect_engine(str(configured), tmp_path / "prefix")
    assert state.blocked is not None and str(configured) in state.blocked
    assert not state.ready and state.download_bytes == 0 and state.configured
    with pytest.raises(SetupError):
        install_engine(state, download=fake_download({}, []), emit=lambda _: None)
    if content is not None:
        assert configured.read_bytes() == content  # never replaced


def test_a_configured_pinned_cli_is_ready(tmp_path: Path) -> None:
    configured = tmp_path / "opt" / "draw-things-cli"
    configured.parent.mkdir()
    configured.write_bytes(CLI_BYTES)
    state = inspect_engine(str(configured), tmp_path / "prefix")
    assert state == EngineState(configured, True, False, None, True)


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, status: int) -> None:
        super().__init__(data)
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_http_download_continues_a_partial_file_with_a_range_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = bytes(range(256)) * 8
    ranges: list[str | None] = []

    def urlopen(request, timeout):
        rng = request.get_header("Range")
        ranges.append(rng)
        if rng is None:
            raise TimeoutError("read timed out")  # nothing arrived
        start = int(rng.removeprefix("bytes=").rstrip("-"))
        return FakeResponse(content[start:], 206)

    part = tmp_path / "x.part"
    part.write_bytes(content[:100])  # left by an interrupted run
    monkeypatch.setattr(engine.urllib.request, "urlopen", urlopen)
    messages: list[str] = []
    http_download("https://example.invalid/x", part, len(content), messages.append)
    assert part.read_bytes() == content and ranges == ["bytes=100-"]


def test_http_download_restarts_when_the_server_ignores_the_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"abcdef" * 10
    attempts: list[int] = []

    def urlopen(request, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("connection reset")
        return FakeResponse(content, 200)

    part = tmp_path / "x.part"
    part.write_bytes(b"abc")
    monkeypatch.setattr(engine.urllib.request, "urlopen", urlopen)
    messages: list[str] = []
    http_download("https://example.invalid/x", part, len(content), messages.append)
    assert part.read_bytes() == content and "resuming (1/5)" in messages[0]


def test_http_download_gives_up_after_its_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def urlopen(request, timeout):
        raise OSError("no route to host")

    monkeypatch.setattr(engine.urllib.request, "urlopen", urlopen)
    with pytest.raises(SetupError, match="no route to host"):
        http_download("https://example.invalid/x", tmp_path / "x.part", 10, lambda _: None)
