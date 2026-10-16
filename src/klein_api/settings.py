"""Configuration: defaults < TOML file < ``KLEIN_API_*`` environment variables.

The TOML file is ``$KLEIN_API_CONFIG`` or ``~/.config/klein-api/config.toml``. Backend options
live in ``[backends.<name>]`` tables and are validated by the backend itself. Other GPU servers
to wait for are ``[[coordinate_with]]`` tables (file only: they hold tokens).
"""

from __future__ import annotations

import ipaddress
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

ENV_PREFIX = "KLEIN_API_"
DEFAULT_CONFIG_PATH = Path("~/.config/klein-api/config.toml")
DEFAULT_DATA_DIR = Path("~/.local/share/klein-api")
DEFAULT_PORT = 8767
MIN_TOKEN_LENGTH = 32
_TOKEN = re.compile(rf"[\x21-\x7e]{{{MIN_TOKEN_LENGTH},}}")
# A peer's own token only has to be something an HTTP header can carry.
_PEER_TOKEN = re.compile(r"[\x21-\x7e]{1,512}")

# Scalar settings that may come from the environment (field name -> env suffix).
_ENV_FIELDS = (
    "data_dir",
    "host",
    "port",
    "token",
    "max_waiting",
    "retention_days",
    "job_timeout_factor",
    "max_body_mb",
    "settle_seconds",
    "coordination_poll_seconds",
)


class SettingsError(ValueError):
    """Raised when configuration is missing or invalid (message is user-facing)."""


class PeerServer(BaseModel):
    """Another GPU server on this Mac whose ``/v1/health`` must be idle before a job starts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    health_url: str = Field(max_length=2048)
    token: SecretStr | None = None

    @field_validator("health_url")
    @classmethod
    def _url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("health_url must be an http(s) URL with a host")
        if parts.username or parts.password:
            raise ValueError("health_url must not contain credentials (use token)")
        return value

    @field_validator("token")
    @classmethod
    def _token(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not _PEER_TOKEN.fullmatch(value.get_secret_value()):
            raise ValueError("token must be printable ASCII without spaces")
        return value


class Settings(BaseModel):
    """Validated server configuration. Immutable once loaded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_dir: Path = DEFAULT_DATA_DIR
    host: str = "127.0.0.1"
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    token: SecretStr | None = None
    max_waiting: int = Field(default=2, ge=0, le=20)
    retention_days: int = Field(default=7, ge=1, le=365)
    job_timeout_factor: float = Field(default=3.0, ge=1.0, le=20.0)
    max_body_mb: int = Field(default=64, ge=1, le=256)
    coordinate_with: tuple[PeerServer, ...] = ()
    settle_seconds: float = Field(default=30.0, ge=0.0, le=600.0)
    coordination_poll_seconds: float = Field(default=5.0, ge=0.05, le=300.0)
    backends: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.token is not None and not _TOKEN.fullmatch(self.token.get_secret_value()):
            raise ValueError(
                f"token must be at least {MIN_TOKEN_LENGTH} printable ASCII characters without "
                "spaces (e.g. `openssl rand -hex 32`)"
            )
        if not is_loopback(self.host) and self.token is None:
            raise ValueError(
                f"host {self.host!r} is not a loopback address: set KLEIN_API_TOKEN "
                "so the API is not exposed without authentication"
            )
        names = [peer.name for peer in self.coordinate_with]
        if len(set(names)) != len(names):
            raise ValueError("coordinate_with names must be unique")
        return self

    def backend_options(self, name: str = "drawthings") -> Mapping[str, Any]:
        return self.backends.get(name, {})

    def with_overrides(self, **values: Any) -> Settings:
        """A re-validated copy (e.g. ``serve --host``): the token rule applies again."""
        data = self.model_dump() | {k: v for k, v in values.items() if v is not None}
        try:
            return Settings.model_validate(data)
        except ValidationError as exc:
            raise SettingsError(_describe(exc)) from exc


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _expand(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("~"):
        return str(Path(value).expanduser())
    return value


def _describe(exc: ValidationError) -> str:
    # never echo the offending input: it may be a token
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'settings'}: {err['msg']}"
        for err in exc.errors()
    )
    return f"invalid configuration: {problems}"


def resolve_config_path(
    env: Mapping[str, str] | None = None, config_path: Path | None = None
) -> Path:
    environ = os.environ if env is None else env
    path = config_path or Path(environ.get(ENV_PREFIX + "CONFIG", str(DEFAULT_CONFIG_PATH)))
    return path.expanduser()


def load_settings(
    env: Mapping[str, str] | None = None, config_path: Path | None = None
) -> Settings:
    """Merge defaults, the TOML file and the environment, then validate."""
    environ = os.environ if env is None else env
    path = resolve_config_path(environ, config_path)

    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise SettingsError(f"cannot read config file {path}: {exc}") from exc

    for field in _ENV_FIELDS:
        raw = environ.get(ENV_PREFIX + field.upper())
        if raw is not None and raw.strip() != "":
            data[field] = raw

    data = {key: _expand(value) for key, value in data.items()}
    if "data_dir" not in data:
        data["data_dir"] = str(DEFAULT_DATA_DIR.expanduser())
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise SettingsError(_describe(exc)) from exc
