"""``klein-api`` command line: serve, doctor, workflows, config."""

from __future__ import annotations

import argparse
import contextlib
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
from klein_api.settings import Settings, SettingsError, load_settings, resolve_config_path
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        settings = load_settings(config_path=Path(args.config) if args.config else None)
        return int(args.func(settings, args))
    except SettingsError as exc:
        _err(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130
