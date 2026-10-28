"""Facts about this Mac that decide what setup can install."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIN_MACOS = (14, 0)


@dataclass(frozen=True)
class Machine:
    arm64: bool
    macos: str  # "" when not macOS
    ram_gb: int
    free_bytes: int
    tools: dict[str, bool]

    def unsupported(self) -> str | None:
        """Why nothing can be installed here, else None."""
        if not self.macos or not self.arm64:
            return "needs an Apple Silicon Mac"
        parts = tuple(int(p) for p in re.findall(r"\d+", self.macos)[:2])
        if parts and parts < MIN_MACOS:
            return f"needs macOS {'.'.join(map(str, MIN_MACOS))}+ (this Mac: {self.macos})"
        return None

    def as_json(self) -> dict[str, Any]:
        return {
            "arm64": self.arm64,
            "macos": self.macos,
            "ramGb": self.ram_gb,
            "freeBytes": self.free_bytes,
            "tools": dict(self.tools),
        }


def existing_parent(path: Path) -> Path:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return target


def _sysctl(name: str) -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _ram_bytes() -> int:
    raw = _sysctl("hw.memsize")
    return int(raw) if raw.isdigit() else 0


def machine_facts(disk_path: Path) -> Machine:
    is_mac = platform.system() == "Darwin"
    return Machine(
        # hw.optional.arm64: Apple Silicon even when this Python runs under Rosetta
        arm64=platform.machine() == "arm64" or (is_mac and _sysctl("hw.optional.arm64") == "1"),
        macos=platform.mac_ver()[0] if is_mac else "",
        ram_gb=round(_ram_bytes() / 2**30),
        free_bytes=shutil.disk_usage(existing_parent(disk_path)).free,
        tools={"launchctl": shutil.which("launchctl") is not None},
    )
