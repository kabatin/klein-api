"""``--service``: run ``klein-api serve`` at login under launchd, then wait for it to answer.

The LaunchAgent runs the installed CLI's real path with ``serve``, restarts it only after a
crash, and logs to ``~/Library/Logs``. The first time it reads Draw Things' Models folder
(inside another app's container), macOS asks on screen whether it may access other apps'
data; until someone approves, jobs hang. Setup cannot click that: it reports it.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from klein_api.setup.errors import Emit, SetupError

LABEL = "com.github.kabatin.klein-api"
REPO = "klein-api"
HEALTH_WAIT_S = 60.0
HEALTH_POLL_S = 1.0
PATH_ENV = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
DIALOG_HINT = (
    "approve the dialog on the Mac's screen: macOS asks whether klein-api may access data "
    "from other apps (Draw Things' Models folder); jobs hang until it is approved"
)

Run = Callable[[Sequence[str]], int]
# url -> HTTP status, or None when nothing answers
HttpGet = Callable[[str], int | None]


@dataclass(frozen=True)
class ServicePaths:
    label: str
    plist: Path
    log: Path


def service_paths(home: Path, suffix: str | None = None) -> ServicePaths:
    label = f"{LABEL}.{suffix}" if suffix else LABEL
    name = f"{REPO}.{suffix}" if suffix else REPO
    return ServicePaths(
        label=label,
        plist=home / "Library" / "LaunchAgents" / f"{label}.plist",
        log=home / "Library" / "Logs" / f"{name}.log",
    )


def plist_bytes(program: str, paths: ServicePaths, env: Mapping[str, str]) -> bytes:
    return plistlib.dumps(
        {
            "Label": paths.label,
            "ProgramArguments": [program, "serve"],
            "EnvironmentVariables": {"PATH": PATH_ENV, **env},
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "StandardOutPath": str(paths.log),
            "StandardErrorPath": str(paths.log),
        }
    )


def run_quiet(cmd: Sequence[str]) -> int:
    return subprocess.run(list(cmd), capture_output=True, check=False).returncode


def http_status(url: str) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - loopback http
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return exc.code
    except (OSError, urllib.error.URLError):
        return None


def install_service(
    paths: ServicePaths,
    *,
    port: int,
    config_env: Mapping[str, str],
    which: Callable[[str], str | None] = shutil.which,
    run: Run = run_quiet,
    http_get: HttpGet = http_status,
    uid: int | None = None,
    emit: Emit = lambda _: None,
    wait_s: float = HEALTH_WAIT_S,
    poll_s: float = HEALTH_POLL_S,
) -> bool:
    """Write the LaunchAgent, (re)load it and wait for ``/v1/health``. Returns healthy."""
    found = which(REPO)
    if found is None:
        raise SetupError(
            "klein-api is not on PATH: install it first "
            "(uv tool install git+https://github.com/kabatin/klein-api)"
        )
    program = str(Path(found).resolve())
    paths.plist.parent.mkdir(parents=True, exist_ok=True)
    paths.log.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.plist.with_name(paths.plist.name + ".tmp")
    tmp.write_bytes(plist_bytes(program, paths, config_env))
    tmp.chmod(0o644)
    tmp.replace(paths.plist)
    emit(f"wrote {paths.plist}")
    domain = f"gui/{os.getuid() if uid is None else uid}"
    run(["launchctl", "bootout", f"{domain}/{paths.label}"])  # not loaded yet is fine
    if run(["launchctl", "bootstrap", domain, str(paths.plist)]) != 0:
        raise SetupError(f"launchctl bootstrap {domain} {paths.plist} failed")
    emit(f"loaded {paths.label}; waiting for http://127.0.0.1:{port}/v1/health")
    return wait_healthy(port, http_get=http_get, wait_s=wait_s, poll_s=poll_s)


def wait_healthy(
    port: int, *, http_get: HttpGet = http_status, wait_s: float, poll_s: float
) -> bool:
    url = f"http://127.0.0.1:{port}/v1/health"
    deadline = time.monotonic() + wait_s
    while True:
        if http_get(url) == 200:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)
