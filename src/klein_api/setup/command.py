"""``klein-api setup``: report (``--check``) or install components, write the config, and
optionally run the server under launchd. Every side effect is injected through :class:`Deps`
so tests never touch the network, the real Models folder or launchctl."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Container, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from klein_api import __version__
from klein_api.backends.base import ModelId
from klein_api.backends.drawthings import DEFAULT_MODELS_DIR
from klein_api.settings import DEFAULT_PORT, ENV_PREFIX, is_loopback
from klein_api.setup import service as svc
from klein_api.setup.config_write import ConfigRequest, read_config, write_config
from klein_api.setup.engine import Download, EngineState, http_download, install_engine
from klein_api.setup.engine import inspect_engine as _inspect_engine
from klein_api.setup.errors import Emit, SetupError
from klein_api.setup.machine import Machine, machine_facts
from klein_api.setup.models import EnsureRun, ModelEnsurer, missing_files, run_ensure
from klein_api.setup.pins import COMPONENTS, ENGINE_LICENCE, FILE_BYTES

SERVER = "klein-api"
# room for the CLI's own work (a text encoder is split into index + -tensordata after download)
DISK_HEADROOM_BYTES = 2_000_000_000
EXIT_OK, EXIT_FAILED, EXIT_HUMAN = 0, 1, 3


def _stderr(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


@dataclass(frozen=True)
class SetupArgs:
    models: tuple[ModelId, ...]
    config_path: Path
    config_is_default: bool = True
    check: bool = False
    yes: bool = False
    lan: bool = False
    service: bool = False
    port: int | None = None
    prefix: Path = Path("~/.local/opt/klein-api")
    label_suffix: str | None = None
    coordinate_with: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Deps:
    download: Download = http_download
    run_ensure: EnsureRun = run_ensure
    launchctl: svc.Run = svc.run_quiet
    which: Callable[[str], str | None] = shutil.which
    http_get: svc.HttpGet = svc.http_status
    machine: Callable[[Path], Machine] = machine_facts
    is_tty: Callable[[], bool] = lambda: sys.stdin.isatty()
    ask: Callable[[str], str] = input
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    home: Path = field(default_factory=Path.home)
    uid: int | None = None
    emit: Emit = _stderr
    health_wait_s: float = svc.HEALTH_WAIT_S


@dataclass(frozen=True)
class Row:
    id: ModelId
    status: str
    message: str
    download_bytes: int = 0
    fix: dict[str, Any] | None = None
    missing: tuple[str, ...] = ()  # model files to fetch

    def as_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "downloadBytes": self.download_bytes,
        }
        if self.fix is not None:
            row["fix"] = self.fix
        return row


def _fix(hint: str, command: str | None = None, needs_human: bool = True) -> dict[str, Any]:
    return {"hint": hint, "command": command, "needsHuman": needs_human}


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def plan_rows(
    args: SetupArgs, machine: Machine, engine: EngineState, models_dir: Path
) -> list[Row]:
    """Per-component state. Shared downloads (the engine, a VAE) count on the first
    component that needs them, so the rows add up to the real total."""
    rows: list[Row] = []
    counted: set[str] = set()
    engine_counted = engine.ready
    for model in args.models:
        component = COMPONENTS[model]
        unsupported = machine.unsupported()
        if unsupported is None and component.min_ram_gb and machine.ram_gb < component.min_ram_gb:
            unsupported = f"needs {component.min_ram_gb} GB of memory ({component.measured})"
        if unsupported is not None:
            rows.append(Row(model, "unsupported", unsupported))
            continue
        if engine.blocked is not None:
            rows.append(Row(model, "needs_human", engine.blocked, fix=_fix(engine.blocked)))
            continue
        missing = missing_files(component, models_dir)
        if not missing and engine.ready:
            rows.append(Row(model, "already", f"{component.title}: ready in {models_dir}"))
            continue
        size = sum(FILE_BYTES[name] for name in missing if name not in counted)
        counted.update(missing)
        parts = [f"model files: {', '.join(missing)}"] if missing else []
        if not engine_counted:
            size += engine.download_bytes
            engine_counted = True
            parts.insert(0, f"draw-things-cli into {engine.cli.parent}")
        message = f"{component.title}: needs {'; '.join(parts)}"
        rows.append(Row(model, "missing", message, size, missing=tuple(missing)))
    return rows


@dataclass(frozen=True)
class Outcome:
    rows: list[Row]
    data: dict[str, Any]
    service: dict[str, Any]
    exit_code: int
    error: str | None = None


def run_setup(args: SetupArgs, deps: Deps | None = None) -> tuple[int, dict[str, Any]]:
    d = deps or Deps()
    try:
        current = read_config(args.config_path)
    except SetupError as exc:
        return EXIT_FAILED, _report(args, d, Outcome([], {}, {}, EXIT_FAILED, str(exc)), None)
    drawthings = current.get("backends", {}).get("drawthings", {})
    models_raw = str(drawthings.get("models_dir") or DEFAULT_MODELS_DIR)
    models_dir = Path(models_raw).expanduser()
    engine = _inspect_engine(drawthings.get("cli"), args.prefix.expanduser())
    machine = d.machine(models_dir)
    rows = plan_rows(args, machine, engine, models_dir)
    if args.check:
        outcome = Outcome(rows, current, _service_state(args, d, current), EXIT_OK)
        return EXIT_OK, _report(args, d, outcome, machine, action="check")
    outcome = _install(args, d, rows, engine, machine, models_dir, models_raw, current)
    if outcome.exit_code == EXIT_OK and any(r.status == "needs_human" for r in outcome.rows):
        outcome = replace(outcome, exit_code=EXIT_HUMAN)
    return outcome.exit_code, _report(args, d, outcome, machine)


def _with(rows: list[Row], ids: Container[str], **changes: Any) -> list[Row]:
    return [replace(row, **changes) if row.id in ids else row for row in rows]


def _confirmation_fix(args: SetupArgs, d: Deps, todo: list[Row], total: int) -> dict | None:
    """None when the download may start; else the fix to report."""
    if total == 0 or args.yes:
        return None
    rerun = f"klein-api setup --models {','.join(r.id for r in todo)} --yes"
    if not d.is_tty():
        return _fix("rerun with --yes to download", rerun, needs_human=False)
    d.emit(f"klein-api setup will download {_gb(total)}:")
    for row in todo:
        d.emit(f"  {row.id}: {_gb(row.download_bytes)} — {COMPONENTS[row.id].licence}")
    d.emit(f"  ({ENGINE_LICENCE})")
    if d.ask("Continue? [y/N] ").strip().lower() in ("y", "yes"):
        return None
    return _fix("declined; rerun to install", rerun)


def disk_needed(todo: list[Row]) -> int:
    """The download, plus the largest file once more (the CLI rewrites a text encoder into
    index + -tensordata), plus headroom."""
    total = sum(row.download_bytes for row in todo)
    if total == 0:
        return 0
    largest = max((FILE_BYTES[name] for row in todo for name in row.missing), default=0)
    return total + largest + DISK_HEADROOM_BYTES


def _install(
    args: SetupArgs,
    d: Deps,
    rows: list[Row],
    engine: EngineState,
    machine: Machine,
    models_dir: Path,
    models_raw: str,
    current: dict[str, Any],
) -> Outcome:
    todo = [row for row in rows if row.status == "missing"]
    ids = {row.id for row in todo}
    total = sum(row.download_bytes for row in todo)
    fix = _confirmation_fix(args, d, todo, total)
    if fix is not None:
        return Outcome(_with(rows, ids, fix=fix), current, {}, EXIT_HUMAN)
    need = disk_needed(todo)
    if machine.free_bytes < need:
        hint = f"free disk space: needs {_gb(need)}, {_gb(machine.free_bytes)} free"
        failed = _with(rows, ids, status="failed", fix=_fix(hint))
        return Outcome(failed, current, {}, EXIT_FAILED)
    if todo and not engine.ready:
        try:
            engine = install_engine(engine, download=d.download, emit=d.emit)
        except SetupError as exc:
            failed = _with(rows, ids, status="failed", message=str(exc))
            return Outcome(failed, current, {}, EXIT_FAILED)
    ensurer = ModelEnsurer(
        engine.cli, models_dir, _work_dir(current), run=d.run_ensure, emit=d.emit
    )
    done = [_ensure(ensurer, row) if row.id in ids else row for row in rows]
    return _configure(args, d, done, engine, models_raw, current)


def _ensure(ensurer: ModelEnsurer, row: Row) -> Row:
    component = COMPONENTS[row.id]
    try:
        ensurer.ensure(component)
    except SetupError as exc:
        return replace(row, status="failed", message=str(exc))
    return replace(row, status="installed", message=f"{component.title}: installed")


def _work_dir(current: Mapping[str, Any]) -> Path:
    data_dir = current.get("data_dir") or "~/.local/share/klein-api"
    return Path(str(data_dir)).expanduser() / "setup"


def _configure(
    args: SetupArgs,
    d: Deps,
    rows: list[Row],
    engine: EngineState,
    models_raw: str,
    current: dict[str, Any],
) -> Outcome:
    failed = any(row.status == "failed" for row in rows)
    request = ConfigRequest(
        cli=engine.cli if engine.ready and not engine.configured else None,
        models_dir=models_raw,
        lan=args.lan,
        port=args.port,
        coordinate_with=args.coordinate_with,
    )
    try:
        data, changed = write_config(args.config_path, request, d.env)
    except SetupError as exc:
        return Outcome(rows, current, {}, EXIT_FAILED, str(exc))
    if changed:
        d.emit(f"updated {args.config_path}")
    if not args.service or failed:
        return Outcome(rows, data, _service_state(args, d, data), EXIT_FAILED if failed else 0)
    paths = svc.service_paths(d.home, args.label_suffix)
    env = {} if args.config_is_default else {ENV_PREFIX + "CONFIG": str(args.config_path)}
    try:
        healthy = svc.install_service(
            paths,
            port=_port(data),
            config_env=env,
            token=_token(data, d.env),
            which=d.which,
            run=d.launchctl,
            http_get=d.http_get,
            uid=d.uid,
            emit=d.emit,
            wait_s=d.health_wait_s,
        )
    except SetupError as exc:
        service = {"installed": False, "label": paths.label, "healthy": False}
        return Outcome(rows, data, service, EXIT_FAILED, str(exc))
    service = {"installed": True, "label": paths.label, "healthy": healthy}
    if not healthy:
        return Outcome(rows, data, service, EXIT_FAILED, f"no answer from the server: {paths.log}")
    ready = {row.id for row in rows if row.status in ("installed", "already")}
    waiting = _with(rows, ready, status="needs_human", fix=_fix(svc.DIALOG_HINT))
    return Outcome(waiting, data, service, EXIT_HUMAN)


def _port(data: Mapping[str, Any]) -> int:
    return int(data.get("port") or DEFAULT_PORT)


def _token(data: Mapping[str, Any], env: Mapping[str, str]) -> str | None:
    token = env.get(ENV_PREFIX + "TOKEN") or data.get("token")
    return str(token) if token else None


def _service_state(args: SetupArgs, d: Deps, data: Mapping[str, Any]) -> dict[str, Any]:
    paths = svc.service_paths(d.home, args.label_suffix)
    installed = paths.plist.is_file()
    url = f"http://127.0.0.1:{_port(data)}/v1/health"
    healthy = installed and d.http_get(url, _token(data, d.env)) == 200
    return {"installed": installed, "label": paths.label, "healthy": healthy}


def _report(
    args: SetupArgs,
    d: Deps,
    outcome: Outcome,
    machine: Machine | None,
    action: str = "install",
) -> dict[str, Any]:
    data = outcome.data
    host = str(data.get("host") or "127.0.0.1")
    report: dict[str, Any] = {
        "server": SERVER,
        "version": __version__,
        "ok": outcome.error is None and all(r.status != "failed" for r in outcome.rows),
        "action": action,
        "url": f"http://127.0.0.1:{_port(data)}",
        "lan": not is_loopback(host),
        "tokenSet": _token(data, d.env) is not None,
        "configPath": str(args.config_path),
        "service": outcome.service or None,
        "machine": machine.as_json() if machine is not None else None,
        "components": [row.as_json() for row in outcome.rows],
    }
    if outcome.error is not None:
        report["error"] = outcome.error
    return report
