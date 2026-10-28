"""Add what setup needs to the config file ``settings.py`` reads, never removing anything.

Existing keys keep their values (only ``--lan`` / ``--port`` set ``host`` / ``port``, because
the user asked for exactly that). The result is validated as :class:`Settings` before it is
written — atomically (temp file + rename), mode 0600. The token is never printed.

Python has no TOML writer: :func:`dumps` emits the subset used here (tables, arrays of
tables, str / int / float / bool / list of str).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import secrets
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from klein_api.settings import (
    _TOKEN,
    DEFAULT_CONFIG_PATH,
    ENV_PREFIX,
    PeerServer,
    Settings,
)
from klein_api.setup.errors import SetupError

LAN_HOST = "0.0.0.0"  # noqa: S104 - only with --lan, and never without a token
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


# -- TOML ----------------------------------------------------------------------------------
def _key(name: str) -> str:
    return name if _BARE_KEY.fullmatch(name) else json.dumps(name)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("only finite floats are supported")
        return repr(value)
    if isinstance(value, str):
        # JSON's escapes (\" \\ \n \t \uXXXX) are valid in TOML basic strings
        return json.dumps(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "[" + ", ".join(json.dumps(item) for item in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def _is_table_array(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)


def _emit_table(lines: list[str], path: Sequence[str], table: Mapping[str, Any]) -> None:
    scalars = {k: v for k, v in table.items() if not isinstance(v, dict) and not _is_table_array(v)}
    if path and (scalars or not table):
        lines += ["", f"[{'.'.join(_key(p) for p in path)}]"]
    lines += [f"{_key(k)} = {_value(v)}" for k, v in scalars.items()]
    for name, value in table.items():
        if isinstance(value, dict):
            _emit_table(lines, [*path, name], value)
        elif _is_table_array(value):
            for item in value:
                lines += ["", f"[[{'.'.join(_key(p) for p in [*path, name])}]]"]
                lines += [f"{_key(k)} = {_value(v)}" for k, v in item.items()]


def dumps(data: Mapping[str, Any]) -> str:
    lines: list[str] = []
    _emit_table(lines, [], data)
    return "\n".join(lines).lstrip("\n") + "\n"


# -- merge ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfigRequest:
    cli: Path | None = None
    models_dir: str | None = None
    lan: bool = False
    port: int | None = None
    coordinate_with: tuple[tuple[str, str], ...] = ()


def read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SetupError(f"cannot read config file {path}: {exc}") from exc


def coord_token_env(name: str) -> str:
    return f"{ENV_PREFIX}COORD_TOKEN_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def _token(existing: Any, env: Mapping[str, str]) -> str:
    from_env = env.get(ENV_PREFIX + "TOKEN", "").strip()
    if from_env:
        if not _TOKEN.fullmatch(from_env):
            raise SetupError(
                f"{ENV_PREFIX}TOKEN must be at least 32 printable ASCII characters without spaces"
            )
        return from_env
    if isinstance(existing, str) and existing:
        return existing
    return secrets.token_urlsafe(32)


def merged(current: Mapping[str, Any], request: ConfigRequest, env: Mapping[str, str]) -> dict:
    """``current`` plus what ``request`` needs; existing values win (except host/port)."""
    data: dict[str, Any] = {key: value for key, value in current.items()}
    backends = dict(data.get("backends", {}))
    drawthings = dict(backends.get("drawthings", {}))
    if request.cli is not None:
        drawthings.setdefault("cli", str(request.cli))
    if request.models_dir is not None:
        drawthings.setdefault("models_dir", request.models_dir)
    if drawthings:
        data["backends"] = {**backends, "drawthings": drawthings}
    if request.port is not None:
        data["port"] = request.port
    if request.lan:
        data["host"] = LAN_HOST
        data["token"] = _token(data.get("token"), env)
    peers = [dict(peer) for peer in data.get("coordinate_with", [])]
    known = {peer.get("name") for peer in peers}
    for name, url in request.coordinate_with:
        if name in known:
            continue
        peer: dict[str, Any] = {"name": name, "health_url": url}
        token = env.get(coord_token_env(name), "").strip()
        if token:
            peer["token"] = token
        try:
            PeerServer.model_validate(peer)
        except ValidationError as exc:
            problems = "; ".join(str(err["msg"]) for err in exc.errors())
            raise SetupError(f"--coordinate-with {name}: {problems}") from exc
        peers.append(peer)
        known.add(name)
    if peers:
        data["coordinate_with"] = peers
    return data


def _validate(data: Mapping[str, Any]) -> None:
    expanded = {
        k: (str(Path(v).expanduser()) if isinstance(v, str) and v.startswith("~") else v)
        for k, v in data.items()
    }
    try:
        Settings.model_validate(expanded)
    except ValidationError as exc:
        # never echo the input: it may hold a token
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'settings'}: {err['msg']}"
            for err in exc.errors()
        )
        raise SetupError(f"the config would be invalid: {problems}") from exc


def write_config(
    path: Path, request: ConfigRequest, env: Mapping[str, str] | None = None
) -> tuple[dict[str, Any], bool]:
    """Merge and write; returns the resulting data and whether the file changed."""
    environ = os.environ if env is None else env
    current = read_config(path)
    data = merged(current, request, environ)
    if data == current:
        return data, False
    _validate(data)
    text = dumps(data)
    if tomllib.loads(text) != data:  # pragma: no cover - the emitter's own guarantee
        raise SetupError("internal error: the TOML writer does not round-trip")
    _atomic_write(path, text)
    return data, True


def _atomic_write(path: Path, text: str) -> None:
    parent = path.parent
    created = not parent.exists()
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if created or parent == DEFAULT_CONFIG_PATH.expanduser().parent:
        parent.chmod(0o700)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(tmp).chmod(0o600)
        Path(tmp).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp).unlink()
        raise
