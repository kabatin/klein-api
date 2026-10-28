"""The engine: find or install the pinned draw-things-cli (and gRPCServerCLI-macOS beside it).

An engine counts as installed only when the file's SHA-256 is the pinned release's: an older
CLI (e.g. Homebrew's) runs, but without ``--local`` and the models this server uses.
A CLI path set by hand in the config file is never replaced: when it is not the pinned
release, setup stops and says what to do.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from klein_api.backends.drawthings import DEFAULT_CLI
from klein_api.backends.resident import SERVER_NAME
from klein_api.setup.errors import Emit, SetupError
from klein_api.setup.pins import CLI_ASSET, CLI_VERSION, GRPC_ASSET, Asset

# (url, destination .part file, expected size, emit) -> None; resumes an existing .part
Download = Callable[[str, Path, int, Emit], None]
DOWNLOAD_ATTEMPTS = 5
READ_TIMEOUT_S = 60.0
_CHUNK = 1 << 20


@dataclass(frozen=True)
class EngineState:
    cli: Path  # where the CLI is (or will be installed)
    cli_ok: bool
    grpc_ok: bool
    # set: the configured CLI is not ours to replace; a human has to decide
    blocked: str | None = None
    # the config file names this CLI already (nothing to add there)
    configured: bool = False

    @property
    def grpc(self) -> Path:
        return self.cli.parent / SERVER_NAME

    @property
    def ready(self) -> bool:
        return self.cli_ok and self.blocked is None

    @property
    def download_bytes(self) -> int:
        if self.blocked is not None:
            return 0
        return (0 if self.cli_ok else CLI_ASSET.size) + (0 if self.grpc_ok else GRPC_ASSET.size)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches(path: Path, asset: Asset) -> bool:
    try:
        return path.stat().st_size == asset.size and sha256_of(path) == asset.sha256
    except OSError:
        return False


def prefix_cli(prefix: Path) -> Path:
    return prefix / "draw-things-cli" / CLI_VERSION / CLI_ASSET.name


def inspect_engine(configured_cli: str | None, prefix: Path) -> EngineState:
    """Where the engine is and whether it is the pinned release. Reads files only."""
    if configured_cli is not None:
        cli = Path(configured_cli).expanduser()
        if _matches(cli, CLI_ASSET):
            return EngineState(
                cli, True, _matches(cli.parent / SERVER_NAME, GRPC_ASSET), None, True
            )
        state = "is missing" if not cli.exists() else f"is not draw-things-cli {CLI_VERSION}"
        return EngineState(
            cli,
            False,
            False,
            f"the config file's [backends.drawthings] cli ({cli}) {state}; remove that line "
            f"and rerun setup to install {CLI_VERSION}, or put {CLI_VERSION} there",
            True,
        )
    for cli in (Path(DEFAULT_CLI).expanduser(), prefix_cli(prefix)):
        if _matches(cli, CLI_ASSET):
            return EngineState(cli, True, _matches(cli.parent / SERVER_NAME, GRPC_ASSET))
    return EngineState(prefix_cli(prefix), False, False)


def install_engine(state: EngineState, *, download: Download, emit: Emit) -> EngineState:
    if state.blocked is not None:
        raise SetupError(state.blocked)
    if not state.cli_ok:
        _install(CLI_ASSET, state.cli, download, emit)
    if not state.grpc_ok:
        _install(GRPC_ASSET, state.grpc, download, emit)
    return EngineState(state.cli, True, True, None, state.configured)


def _install(asset: Asset, target: Path, download: Download, emit: Emit) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    emit(f"downloading {asset.name} {CLI_VERSION} ({asset.size / 1e6:.0f} MB)")
    download(asset.url, part, asset.size, emit)
    actual = sha256_of(part)
    if actual != asset.sha256:
        part.unlink(missing_ok=True)
        raise SetupError(
            f"{asset.name} download has SHA-256 {actual}, expected {asset.sha256}; "
            "deleted it — rerun setup"
        )
    part.chmod(0o755)
    part.replace(target)  # atomic: a half-written engine is never at the real path
    emit(f"installed {target}")


def http_download(url: str, part: Path, size: int, emit: Emit) -> None:
    """Fetch ``url`` into ``part``, resuming what is there (HTTP Range); a stalled or broken
    connection is retried, continuing where it stopped."""
    last_error = ""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            return
        if have > size:
            part.unlink()
            have = 0
        try:
            _fetch_from(url, part, have)
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            last_error = str(exc) or type(exc).__name__
            emit(f"download interrupted ({last_error}); resuming ({attempt}/{DOWNLOAD_ATTEMPTS})")
            continue
    if not part.exists() or part.stat().st_size != size:
        raise SetupError(f"download of {url} did not finish: {last_error or 'short file'}")


def _fetch_from(url: str, part: Path, offset: int) -> None:
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - pinned https URL
    with urllib.request.urlopen(request, timeout=READ_TIMEOUT_S) as response:  # noqa: S310
        resumed = response.status == 206
        mode = "ab" if resumed else "wb"
        with part.open(mode) as handle:
            for block in iter(lambda: response.read(_CHUNK), b""):
                handle.write(block)
            handle.flush()
            with contextlib.suppress(OSError):
                os.fsync(handle.fileno())
