"""``klein-api`` command line: serve, doctor, workflows, config, setup."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import sys
from collections.abc import Sequence
from pathlib import Path

from klein_api import __version__
from klein_api.backends import build_backend
from klein_api.checks import Check, Level
from klein_api.doctor import run_doctor
from klein_api.jobs.coordination import Coordinator
from klein_api.logs import configure_logging
from klein_api.settings import (
    DEFAULT_CONFIG_PATH,
    Settings,
    SettingsError,
    load_settings,
    resolve_config_path,
)
from klein_api.setup.pins import COMPONENT_IDS
from klein_api.workflows import build_registry

_MARKS = {Level.OK: "✓", Level.WARN: "!", Level.FAIL: "✗"}
MAX_CONNECTIONS = 64
OPEN_FILES_WANTED = 4096


def _out(message: str = "") -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _err(message: str) -> None:
    sys.stderr.write(message + "\n")


def _print_check(check: Check) -> None:
    _out(f"  {_MARKS[check.level]} {check.name:<30} {check.detail}")


def _raise_open_file_limit() -> None:
    """launchd starts agents with a soft limit of 256 open files; ask for more."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = OPEN_FILES_WANTED if hard == resource.RLIM_INFINITY else min(OPEN_FILES_WANTED, hard)
    if soft < wanted:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))


def cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    from klein_api.api.app import create_app
    from klein_api.jobs.queue import JobQueue
    from klein_api.jobs.store import JobStore, StoreLockedError

    settings = settings.with_overrides(host=args.host, port=args.port)
    # job records hold prompts and images: keep them (and outputs) private to this user
    os.umask(0o077)
    _raise_open_file_limit()
    backend = build_backend(settings)
    problems = [check for check in backend.health() if check.failed]
    if problems:
        _err(f"error: the {backend.name} backend cannot generate:")
        for check in problems:
            _err(f"  ✗ {check.name}: {check.detail}")
        _err("run `klein-api doctor` for details")
        return 1
    store = JobStore(settings.data_dir)
    try:
        store.acquire_instance_lock()
    except StoreLockedError as exc:
        _err(f"error: {exc}")
        return 1
    registry = build_registry(backend)
    coordinator = Coordinator(
        settings.coordinate_with,
        settle_seconds=settings.settle_seconds,
        poll_seconds=settings.coordination_poll_seconds,
    )
    if coordinator.peers:
        # an engine keeping a model resident gives the memory back when a peer gets busy
        backend.set_peer_check(lambda: coordinator.check() is not None)
    queue = JobQueue(
        store,
        registry,
        backend,
        max_waiting=settings.max_waiting,
        timeout_factor=settings.job_timeout_factor,
        retention_days=settings.retention_days,
        coordinator=coordinator,
    )
    token = settings.token.get_secret_value() if settings.token else None
    app = create_app(
        queue,
        registry,
        token=token,
        max_body_bytes=settings.max_body_mb << 20,
        waits_for=coordinator.names,
    )
    waits = ", ".join(coordinator.names) or "nothing"
    _out(
        f"klein-api {__version__} on http://{settings.host}:{settings.port} "
        f"(auth: {'bearer' if token else 'none, loopback only'}, backend: {backend.name}, "
        f"reference images: {backend.max_reference_images}, "
        f"workflows: {', '.join(w.id for w in registry)}, "
        f"queue: 1 running + {settings.max_waiting} waiting, waits for: {waits})"
    )
    try:
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            workers=1,
            log_config=None,
            server_header=False,
            # bounded connections: idle or slow clients cannot exhaust file descriptors
            limit_concurrency=MAX_CONNECTIONS,
            timeout_keep_alive=5,
        )
    finally:
        store.release_instance_lock()
    return 0


def cmd_doctor(settings: Settings, args: argparse.Namespace) -> int:
    _out(f"klein-api {__version__} doctor")
    backend = build_backend(settings)
    checks = run_doctor(
        settings,
        backend,
        build_registry(backend),
        smoke=args.smoke,
        emit=_print_check,
        config_path=resolve_config_path(config_path=Path(args.config) if args.config else None),
    )
    failed = [c for c in checks if c.failed]
    _out(f"\n{len(failed)} problem(s)" if failed else "\nall good")
    return 1 if failed else 0


def cmd_workflows(settings: Settings, _: argparse.Namespace) -> int:
    backend = build_backend(settings)
    for workflow in build_registry(backend):
        _out(f"{workflow.id}\n  {workflow.description}")
        _out(f"  engine: {backend.name}, reference images: up to {backend.max_reference_images}")
    return 0


def cmd_config(settings: Settings, _: argparse.Namespace) -> int:
    for key, value in settings.model_dump(exclude={"backends", "coordinate_with"}).items():
        shown = "********" if key == "token" and value is not None else value
        _out(f"{key} = {shown}")
    for peer in settings.coordinate_with:
        token = "********" if peer.token is not None else "none"
        _out(
            f"[[coordinate_with]] name = {peer.name}, health_url = {peer.health_url}, "
            f"token = {token}"
        )
    for name, options in settings.backends.items():
        _out(f"[backends.{name}] {options}")
    return 0


def _component_ids(raw: str) -> tuple[str, ...]:
    ids = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    unknown = [i for i in ids if i not in COMPONENT_IDS]
    if unknown or not ids:
        raise argparse.ArgumentTypeError(
            f"unknown component id(s) {', '.join(unknown) or '(none)'}; "
            f"valid: {', '.join(COMPONENT_IDS)}"
        )
    return ids


def _port(raw: str) -> int:
    if not raw.isdigit() or not 1 <= int(raw) <= 65535:
        raise argparse.ArgumentTypeError("expected a port number 1-65535")
    return int(raw)


def _peer(raw: str) -> tuple[str, str]:
    name, sep, url = raw.partition("=")
    if not sep or not name.strip() or not url.strip():
        raise argparse.ArgumentTypeError("expected NAME=HEALTH_URL")
    return name.strip(), url.strip()


def _setup_lines(report: dict) -> list[str]:
    lines = [f"klein-api {report['version']} setup ({report['action']})"]
    for row in report["components"]:
        size = f"{row['downloadBytes'] / 1e9:.1f} GB" if row["downloadBytes"] else ""
        lines.append(f"  {row['id']:<14} {row['status']:<12} {size:>8}  {row['message']}")
        fix = row.get("fix")
        if fix:
            command = f" — {fix['command']}" if fix.get("command") else ""
            lines.append(f"  {'':<14} fix: {fix['hint']}{command}")
    lines.append(f"  config: {report['configPath']}  url: {report['url']}")
    service = report.get("service")
    if service:
        lines.append(
            f"  service: {service['label']} installed={service['installed']} "
            f"healthy={service.get('healthy')}"
        )
    if "error" in report:
        lines.append(f"  error: {report['error']}")
    return lines


def cmd_setup(args: argparse.Namespace) -> int:
    from klein_api.setup.command import SetupArgs, run_setup

    config_path = resolve_config_path(config_path=Path(args.config) if args.config else None)
    setup_args = SetupArgs(
        models=tuple(args.models or COMPONENT_IDS),  # type: ignore[arg-type]
        config_path=config_path,
        config_is_default=config_path == DEFAULT_CONFIG_PATH.expanduser(),
        check=args.check,
        yes=args.yes,
        lan=args.lan,
        service=args.service,
        port=args.port,
        prefix=Path(args.prefix).expanduser(),
        label_suffix=args.label_suffix,
        coordinate_with=tuple(args.coordinate_with or ()),
    )
    code, report = run_setup(setup_args)
    if args.json:
        _out(json.dumps(report, ensure_ascii=False))
    else:
        for line in _setup_lines(report):
            _out(line)
    return code


def _add_setup(sub: argparse._SubParsersAction) -> None:
    setup = sub.add_parser("setup", help="install the engine and models, write the config")
    setup.add_argument(
        "--models",
        type=_component_ids,
        help=f"comma list of components (default: all of {', '.join(COMPONENT_IDS)})",
    )
    setup.add_argument("--check", action="store_true", help="report only, change nothing")
    setup.add_argument("--yes", action="store_true", help="download without asking")
    setup.add_argument("--json", action="store_true", help="one JSON object on stdout")
    setup.add_argument("--lan", action="store_true", help="listen on 0.0.0.0 with a token")
    setup.add_argument("--service", action="store_true", help="run `serve` at login (launchd)")
    setup.add_argument("--port", type=_port, metavar="N")
    setup.add_argument("--prefix", default="~/.local/opt/klein-api", help="where the CLI goes")
    setup.add_argument("--label-suffix", type=str, help="launchd label suffix (tests)")
    setup.add_argument(
        "--coordinate-with",
        action="append",
        type=_peer,
        metavar="NAME=HEALTH_URL",
        help="another GPU server to wait for (repeatable)",
    )
    setup.set_defaults(func=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="klein-api", description=__doc__)
    parser.add_argument("--version", action="version", version=f"klein-api {__version__}")
    parser.add_argument("--config", help="config file (default ~/.config/klein-api/config.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=cmd_serve)

    doctor = sub.add_parser("doctor", help="check this machine")
    doctor.add_argument("--smoke", action="store_true", help="also run one short generation")
    doctor.set_defaults(func=cmd_doctor)

    sub.add_parser("workflows", help="list workflows").set_defaults(func=cmd_workflows)
    sub.add_parser("config", help="show the effective configuration").set_defaults(func=cmd_config)
    _add_setup(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    if args.command == "setup":
        # setup must work on a machine whose config is missing or not yet complete
        try:
            return cmd_setup(args)
        except KeyboardInterrupt:
            return 130
    try:
        settings = load_settings(config_path=Path(args.config) if args.config else None)
        return int(args.func(settings, args))
    except SettingsError as exc:
        _err(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130
