"""``klein-api setup`` on the command line: arguments, their validation (before anything
runs), and the output. A usage error with ``--json`` is still one JSON report (exit 2)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from klein_api import __version__
from klein_api.settings import DEFAULT_CONFIG_PATH, DEFAULT_PORT, resolve_config_path
from klein_api.setup.config_write import check_lan_token, peer_entry
from klein_api.setup.errors import SetupError
from klein_api.setup.pins import COMPONENT_IDS

EXIT_USAGE = 2
_LABEL_SUFFIX = re.compile(r"[A-Za-z0-9-]{1,32}")


class UsageError(Exception):
    """A bad command line (exit 2); ``parser`` prints the matching usage line."""

    def __init__(self, message: str, parser: argparse.ArgumentParser | None = None) -> None:
        super().__init__(message)
        self.parser = parser


class RaisingParser(argparse.ArgumentParser):
    """Raises instead of printing and exiting, so ``setup --json`` can still answer in JSON."""

    def error(self, message: str) -> Any:  # type: ignore[override]
        raise UsageError(message, self)


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


def _label_suffix(raw: str) -> str:
    if not _LABEL_SUFFIX.fullmatch(raw):
        raise argparse.ArgumentTypeError("expected 1-32 of [A-Za-z0-9-]")
    return raw


def add_setup(sub: argparse._SubParsersAction) -> None:
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
    setup.add_argument(
        "--label-suffix", type=_label_suffix, help="launchd label suffix (needs --port)"
    )
    setup.add_argument(
        "--coordinate-with",
        action="append",
        type=_peer,
        metavar="NAME=HEALTH_URL",
        help="another GPU server to wait for (repeatable)",
    )
    setup.set_defaults(func=None)


def validate(args: argparse.Namespace, env: Mapping[str, str]) -> None:
    """Everything that can be refused before a download starts (exit 2, secrets never
    echoed)."""
    if args.label_suffix and args.port is None:
        raise UsageError("--label-suffix needs --port: a second service must not take the port")
    try:
        if args.lan:
            check_lan_token(env)
        for name, url in args.coordinate_with or ():
            peer_entry(name, url, env)
    except SetupError as exc:
        raise UsageError(str(exc)) from exc


def _config_from_argv(argv: Sequence[str]) -> Path:
    for index, arg in enumerate(argv):
        if arg == "--config" and index + 1 < len(argv):
            return resolve_config_path(config_path=Path(argv[index + 1]))
        if arg.startswith("--config="):
            return resolve_config_path(config_path=Path(arg.split("=", 1)[1]))
    return resolve_config_path()


def usage_report(argv: Sequence[str], message: str) -> dict[str, Any]:
    """The report of a run that never started: every key, ``ok`` false, the error."""
    return {
        "server": "klein-api",
        "version": __version__,
        "ok": False,
        "action": "check" if "--check" in argv else "install",
        "url": f"http://127.0.0.1:{DEFAULT_PORT}",
        "lan": False,
        "tokenSet": False,
        "configPath": str(_config_from_argv(argv)),
        "service": None,
        "machine": None,
        "components": [],
        "error": message,
    }


def wants_json(argv: Sequence[str]) -> bool:
    return "setup" in argv and "--json" in argv


def report_usage_error(argv: Sequence[str], exc: UsageError) -> int:
    parser = exc.parser
    if parser is not None:
        parser.print_usage(sys.stderr)
    prog = parser.prog if parser is not None else "klein-api setup"
    sys.stderr.write(f"{prog}: error: {exc}\n")
    if wants_json(argv):
        sys.stdout.write(json.dumps(usage_report(argv, str(exc)), ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return EXIT_USAGE


def setup_lines(report: dict[str, Any]) -> list[str]:
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

    validate(args, os.environ)
    config_path = resolve_config_path(config_path=Path(args.config) if args.config else None)
    setup_args = SetupArgs(
        models=tuple(args.models or COMPONENT_IDS),  # type: ignore[arg-type]
        explicit=args.models is not None,
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
        sys.stdout.write(json.dumps(report, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write("\n".join(setup_lines(report)) + "\n")
    sys.stdout.flush()
    return code
