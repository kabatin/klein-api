"""Reading draw-things-cli's output. Isolated here so the formats can be corrected after
checks against the real CLI without touching process handling.

Measured with draw-things-cli v26.1011.0 running FLUX.2 [klein] under a pty (2026-10-16):

- progress: ``ESC[1A ESC[K Sampling... i / n [bar] pct %`` + ``\\r\\n``, then
  ``Finishing... [█] 82 %`` and ``Generated [█] 100%``;
- ``Wrote: <path>``, ``Generation timing:``, ``  Sampling step time (n step(s)): avg x s, ...``;
- failures: ``Error: ...`` with exit code 1 (bad size, missing input image) or 64 (unknown
  model: a model table first, then the error, then ``Usage: ...``).

Parsing is tolerant: unknown lines are ignored, and success does not depend on the ``Wrote:``
line (exit 0, no ``Error:`` and a fresh, decodable PNG decide; the workflow checks the PNG).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
SAMPLING = re.compile(r"Sampling\.\.\.\s+(\d+)\s*/\s*(\d+)")
STEP_TIME = re.compile(r"Sampling step time \((\d+) step\(s\)\):\s*avg\s+([0-9.]+)\s*s")
ERROR = re.compile(r"^(?:Error|error|ERROR):\s*(.+)$")
WROTE = re.compile(r"^Wrote:\s*(.+?)\s*$")
MAX_ERROR_CHARS = 500
# Absolute or home-relative paths inside error text (the CLI names input files): never shown
# to clients, which only need to know what went wrong.
PATH = re.compile(r"(?<![\w.])(?:~|/)[^\s'\"]*")


def scrub_paths(text: str) -> str:
    return PATH.sub("<path>", text)


class OutputWatch:
    """Fed one output line at a time; ``on_step`` gets the sampling fraction (0..1)."""

    def __init__(self, on_step: Callable[[float], None] | None = None) -> None:
        self._on_step = on_step
        self.sampling_seconds: float | None = None
        self.error: str | None = None
        self.wrote: Path | None = None

    def line(self, raw: str) -> None:
        # a pty may deliver several carriage-return redraws in one line: the last one counts
        segments = [ANSI.sub("", part).strip() for part in raw.split("\r")]
        for text in reversed([s for s in segments if s]):
            if self._parse(text):
                return

    def _parse(self, text: str) -> bool:
        steps = SAMPLING.findall(text)
        if steps:
            done, total = (int(n) for n in steps[-1])
            if total > 0 and self._on_step is not None:
                self._on_step(min(done, total) / total)
            return True
        timing = STEP_TIME.search(text)
        if timing is not None:
            self.sampling_seconds = int(timing.group(1)) * float(timing.group(2))
            return True
        error = ERROR.match(text)
        if error is not None:
            if self.error is None:  # the first error is the cause; later ones are follow-ups
                self.error = scrub_paths(error.group(1))[:MAX_ERROR_CHARS]
            return True
        wrote = WROTE.match(text)
        if wrote is not None:
            self.wrote = Path(wrote.group(1))
            return True
        return False
