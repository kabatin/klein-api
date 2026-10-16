"""``klein-api doctor``: check that this machine can actually serve the registered workflows
(draw-things-cli and the model files, disk, power, memory, the GPU servers to wait for)."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from klein_api.backends.base import KleinBackend
from klein_api.checks import Check, Level
from klein_api.jobs.coordination import PeerUnreachableError, probe_health
from klein_api.jobs.store import JobStore, StoreLockedError
from klein_api.settings import PeerServer, Settings
from klein_api.workflows.base import Workflow, WorkflowFailedError, WorkflowRegistry

MIN_FREE_GB = 10
MIN_MEMORY_FREE_PERCENT = 20
MIN_MACOS = (14, 0)
SMOKE_TIMEOUT_S = 600


def _run(cmd: list[str], timeout: float = 20) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def check_platform() -> Check:
    system, machine = platform.system(), platform.machine()
    if system != "Darwin" or machine != "arm64":
        return Check("platform", Level.FAIL, f"{system}/{machine}: an Apple Silicon Mac is needed")
    return Check("platform", Level.OK, f"Apple Silicon ({_chip() or machine})")


def _chip() -> str | None:
    result = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return result.stdout.strip() if result is not None and result.returncode == 0 else None


def check_macos(version: str | None = None) -> Check:
    raw = version if version is not None else platform.mac_ver()[0]
    parts = tuple(int(p) for p in re.findall(r"\d+", raw)[:2])
    if not parts:
        return Check("macOS", Level.WARN, "version unknown")
    if parts < MIN_MACOS:
        wanted = ".".join(map(str, MIN_MACOS))
        return Check("macOS", Level.FAIL, f"{raw}: macOS {wanted}+ is needed")
    return Check("macOS", Level.OK, raw)


def check_python() -> Check:
    version = ".".join(map(str, sys.version_info[:3]))
    level = Level.OK if sys.version_info >= (3, 12) else Level.FAIL
    return Check("python", level, version)


def check_peer(peer: PeerServer) -> Check:
    """A GPU server to wait for: reachable, and its current load (never fails the doctor:
    an unreachable server just counts as idle)."""
    name = f"coordinate with {peer.name}"
    try:
        health = probe_health(peer)
    except PeerUnreachableError as exc:
        return Check(name, Level.WARN, f"unreachable ({exc}): it will count as idle")
    state = "idle" if health.idle else "busy"
    return Check(name, Level.OK, f"{state} (running {health.running}, waiting {health.waiting})")


def check_config_file(path: Path | None) -> Check:
    if path is None or not path.is_file():
        return Check("config file", Level.OK, "none (defaults and environment)")
    loose = path.stat().st_mode & 0o077
    if loose and "token" in path.read_text(encoding="utf-8", errors="replace"):
        return Check(
            "config file",
            Level.WARN,
            f"{path} holds the token but others can read it — chmod 600 {path}",
        )
    return Check("config file", Level.OK, str(path))


def check_disk(path: Path) -> Check:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    free_gb = shutil.disk_usage(target).free / 1e9
    level = Level.OK if free_gb >= MIN_FREE_GB else Level.WARN
    return Check("disk", level, f"{free_gb:.0f} GB free at {target}")


def check_power() -> Check:
    result = _run(["pmset", "-g", "batt"])
    if result is None:
        return Check("power", Level.WARN, "pmset unavailable")
    if "AC Power" in result.stdout:
        return Check("power", Level.OK, "on AC power")
    return Check("power", Level.WARN, "on battery: long generations drain it and run slower")


def check_memory() -> Check:
    result = _run(["memory_pressure", "-Q"])
    match = re.search(r"free percentage: (\d+)%", result.stdout if result else "")
    if match is None:
        return Check("memory", Level.WARN, "memory_pressure unavailable")
    free = int(match.group(1))
    level = Level.OK if free >= MIN_MEMORY_FREE_PERCENT else Level.WARN
    return Check("memory", level, f"{free}% free (a standard image needs ~10 GB)")


def run_smoke(settings: Settings, backend: KleinBackend, workflow: Workflow) -> Check:
    """Run the workflow's cheapest job end-to-end, bypassing the HTTP server and the queue.

    Refuses while a server is serving the same data_dir: two generations at once would
    fight over the GPU and memory (a 32 GB Mac cannot hold both).
    """
    name = f"smoke {workflow.id}"
    store = JobStore(settings.data_dir)
    try:
        store.acquire_instance_lock()
    except StoreLockedError:
        return Check(
            name, Level.FAIL, "a klein-api server is running on this data_dir: stop it first"
        )
    try:
        return _smoke(settings, backend, workflow)
    finally:
        store.release_instance_lock()


def _smoke(settings: Settings, backend: KleinBackend, workflow: Workflow) -> Check:
    params = workflow.params_model.model_validate(workflow.smoke_params())
    smoke_root = settings.data_dir / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix=f"{workflow.id}-", dir=smoke_root))
    name = f"smoke {workflow.id}"
    job_id = "job_" + "0" * 26
    stored = workflow.store_inputs(params, job_dir)
    request = workflow.build_request(job_id, stored, job_dir)
    started = time.monotonic()
    raw = backend.generate(request, job_dir, timeout_s=SMOKE_TIMEOUT_S, cancel=threading.Event())
    if not raw.ok:
        where = f" — log: {raw.log_path}" if raw.log_path else ""
        return Check(name, Level.FAIL, f"{raw.describe_failure()}{where}")
    try:
        output = workflow.finalize(job_id, stored, job_dir, raw)
    except WorkflowFailedError as exc:
        return Check(name, Level.FAIL, f"{exc} — files kept in {job_dir}")
    elapsed = time.monotonic() - started
    first = output.result.get("outputs", [{}])[0]
    shutil.rmtree(job_dir, ignore_errors=True)
    return Check(
        name,
        Level.OK,
        f"{first.get('width')}x{first.get('height')} PNG "
        f"(generated {raw.plan.width}x{raw.plan.height}) in {elapsed:.0f}s",
    )


def run_doctor(
    settings: Settings,
    backend: KleinBackend,
    registry: WorkflowRegistry,
    *,
    smoke: bool = False,
    emit: Callable[[Check], None] | None = None,
    config_path: Path | None = None,
) -> list[Check]:
    checks: list[Check] = [
        check_platform(),
        check_macos(),
        check_python(),
        check_config_file(config_path),
        *backend.health(),
        check_disk(settings.data_dir),
        check_power(),
        check_memory(),
        *(check_peer(peer) for peer in settings.coordinate_with),
    ]
    for check in checks:
        if emit is not None:
            emit(check)
    if smoke and not any(check.failed for check in checks):
        for workflow in registry:
            check = run_smoke(settings, backend, workflow)
            checks.append(check)
            if emit is not None:
                emit(check)
    return checks
