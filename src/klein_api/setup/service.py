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

from klein_api.jobs.coordination import no_proxy_opener
from klein_api.setup.errors import Emit, SetupError

LABEL = "com.github.kabatin.klein-api"
REPO = "klein-api"
HEALTH_WAIT_S = 60.0
HEALTH_POLL_S = 1.0
# after bootout, launchd removes the job asynchronously; bootstrap before that fails
UNLOAD_WAIT_S = 10.0
LAUNCHD_POLL_S = 0.25
BOOTSTRAP_ATTEMPTS = 5
BOOTSTRAP_RETRY_CODES = (5, 37)  # EIO / EALREADY: the old job is still being torn down
PATH_ENV = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
DIALOG_HINT = (
    "approve the dialog on the Mac's screen: macOS asks whether klein-api may access data "
    "from other apps (Draw Things' Models folder); jobs hang until it is approved"
)

Run = Callable[[Sequence[str]], int]
# (url, bearer token or None) -> HTTP status, or None when nothing answers
HttpGet = Callable[[str, str | None], int | None]


@dataclass(frozen=True)
class ServiceResult:
    healthy: bool
    # the plist was new or different: a reload with new settings (the macOS dialog may show)
    changed: bool


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


def http_status(url: str, token: str | None = None) -> int | None:
    # a server with a token answers every route, /v1/health too, with 401 without it
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - loopback http
    try:
        # never through a proxy (the token must only reach this server), no redirects
        with no_proxy_opener().open(request, timeout=2) as response:
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
    token: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    run: Run = run_quiet,
    http_get: HttpGet = http_status,
    uid: int | None = None,
    emit: Emit = lambda _: None,
    wait_s: float = HEALTH_WAIT_S,
    poll_s: float = HEALTH_POLL_S,
    launchd_poll_s: float = LAUNCHD_POLL_S,
) -> ServiceResult:
    """Write the LaunchAgent (tmp + rename), (re)load it and wait for ``/v1/health``."""
    found = which(REPO)
    if found is None:
        raise SetupError(
            "klein-api is not on PATH: install it first "
            "(uv tool install git+https://github.com/kabatin/klein-api)"
        )
    program = str(Path(found).resolve())
    paths.plist.parent.mkdir(parents=True, exist_ok=True)
    paths.log.parent.mkdir(parents=True, exist_ok=True)
    content = plist_bytes(program, paths, config_env)
    changed = not paths.plist.is_file() or paths.plist.read_bytes() != content
    if changed:
        tmp = paths.plist.with_name(paths.plist.name + ".tmp")
        tmp.write_bytes(content)
        tmp.chmod(0o644)
        tmp.replace(paths.plist)
        emit(f"wrote {paths.plist}")
    else:
        emit(f"{paths.plist} unchanged; reloading it")
    domain = f"gui/{os.getuid() if uid is None else uid}"
    _reload(run, domain, paths, launchd_poll_s)
    emit(f"loaded {paths.label}; waiting for http://127.0.0.1:{port}/v1/health")
    healthy = wait_healthy(port, token, http_get=http_get, wait_s=wait_s, poll_s=poll_s)
    return ServiceResult(healthy=healthy, changed=changed)


def _reload(run: Run, domain: str, paths: ServicePaths, poll_s: float) -> None:
    target = f"{domain}/{paths.label}"
    run(["launchctl", "bootout", target])  # not loaded yet is fine
    deadline = time.monotonic() + UNLOAD_WAIT_S
    while run(["launchctl", "print", target]) == 0 and time.monotonic() < deadline:
        time.sleep(poll_s)
    for attempt in range(1, BOOTSTRAP_ATTEMPTS + 1):
        code = run(["launchctl", "bootstrap", domain, str(paths.plist)])
        if code == 0:
            return
        if code not in BOOTSTRAP_RETRY_CODES or attempt == BOOTSTRAP_ATTEMPTS:
            raise SetupError(f"launchctl bootstrap {domain} {paths.plist} failed ({code})")
        time.sleep(poll_s)


def wait_healthy(
    port: int,
    token: str | None = None,
    *,
    http_get: HttpGet = http_status,
    wait_s: float,
    poll_s: float,
) -> bool:
    url = f"http://127.0.0.1:{port}/v1/health"
    deadline = time.monotonic() + wait_s
    while True:
        if http_get(url, token) == 200:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)
