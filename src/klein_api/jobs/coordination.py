"""Waiting for the other GPU servers on this Mac (vpipe-api, wan-api) before a job starts.

They do not know about klein-api, so the job at the head of the queue starts only when every
configured server reports ``running == 0 && waiting == 0`` (and ``outside_runs == 0`` when
present). The coordinator remembers when it last saw any server busy: if that was less than
``settle_seconds`` ago, the job waits until ``settle_seconds`` after that observation and
checks again (clients often submit a follow-up job right after one finishes); otherwise one
all-idle check is enough. A server that cannot be asked counts as idle (and is logged as a
warning). Tokens are sent only as ``Authorization`` headers and never
logged; redirects are not followed, so a token never reaches another host.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from klein_api.logs import event
from klein_api.settings import PeerServer

log = logging.getLogger("klein_api.coordination")

PROBE_TIMEOUT_S = 5.0
MAX_HEALTH_BYTES = 64 * 1024


class PeerHealth(BaseModel):
    """The part of a peer's ``/v1/health`` body that matters here (vpipe-api / wan-api)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    running: int = Field(ge=0)
    waiting: int = Field(ge=0)
    outside_runs: int | None = Field(default=None, ge=0)

    @property
    def idle(self) -> bool:
        return self.running == 0 and self.waiting == 0 and not self.outside_runs


class PeerUnreachableError(Exception):
    """The peer could not be asked; the message is safe to log (no token, no body)."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None  # a 3xx is then raised as HTTPError instead of being followed


_OPENER = urllib.request.build_opener(_NoRedirect)


def probe_health(peer: PeerServer, timeout_s: float = PROBE_TIMEOUT_S) -> PeerHealth:
    """Ask one peer. Raises :class:`PeerUnreachableError` for anything but a usable answer."""
    request = urllib.request.Request(peer.health_url, method="GET")  # noqa: S310 - http(s) only
    request.add_header("Accept", "application/json")
    if peer.token is not None:
        request.add_header("Authorization", f"Bearer {peer.token.get_secret_value()}")
    try:
        with _OPENER.open(request, timeout=timeout_s) as response:
            body = response.read(MAX_HEALTH_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise PeerUnreachableError(f"HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise PeerUnreachableError(type(exc).__name__) from None
    if len(body) > MAX_HEALTH_BYTES:
        raise PeerUnreachableError("health response too large")
    try:
        return PeerHealth.model_validate(json.loads(body))
    except (ValueError, ValidationError) as exc:
        raise PeerUnreachableError(f"unexpected health response ({type(exc).__name__})") from None


Probe = Callable[[PeerServer], PeerHealth]
# ``sleep(seconds)`` returns False when the wait must be abandoned (cancel, shutdown, ...)
Sleep = Callable[[float], bool]
Clock = Callable[[], float]


class Coordinator:
    """Decides when the head job may start. Its only state is the monotonic time of the last
    busy observation, which outlives a single job on purpose."""

    def __init__(
        self,
        peers: Sequence[PeerServer],
        *,
        settle_seconds: float = 30.0,
        poll_seconds: float = 5.0,
        probe: Probe = probe_health,
        clock: Clock = time.monotonic,
    ) -> None:
        self.peers = tuple(peers)
        self.settle_seconds = settle_seconds
        self.poll_seconds = poll_seconds
        self._probe = probe
        self._clock = clock
        self._lock = threading.Lock()
        self._last_busy: float | None = None

    @property
    def last_busy(self) -> float | None:
        with self._lock:
            return self._last_busy

    @property
    def names(self) -> list[str]:
        return [peer.name for peer in self.peers]

    def busy_peer(self, job_id: str | None = None) -> str | None:
        """The first peer that is busy right now, or None when all count as idle."""
        for peer in self.peers:
            try:
                health = self._probe(peer)
            except PeerUnreachableError as exc:
                event(
                    log,
                    "coordination.unreachable",
                    level=logging.WARNING,
                    job_id=job_id,
                    peer=peer.name,
                    reason=str(exc),
                )
                continue
            if not health.idle:
                return peer.name
        return None

    def _check(self, job_id: str) -> str | None:
        """One check of every peer; a busy answer is remembered with the time it was seen."""
        busy = self.busy_peer(job_id)
        if busy is not None:
            with self._lock:
                self._last_busy = self._clock()
        return busy

    def settle_left(self) -> float:
        """Seconds until ``settle_seconds`` have passed since the last busy observation."""
        last = self.last_busy
        if last is None:
            return 0.0
        return max(self.settle_seconds - (self._clock() - last), 0.0)

    def wait_until_idle(
        self, job_id: str, *, sleep: Sleep, on_waiting: Callable[[str], None]
    ) -> bool:
        """Block until a check finds every peer idle and no peer was seen busy within the
        last ``settle_seconds``. Returns False if ``sleep`` says to give up. ``on_waiting``
        gets the peer being waited for."""
        if not self.peers:
            return True
        while True:
            busy = self._check(job_id)
            if busy is not None:
                on_waiting(busy)
                if not sleep(self.poll_seconds):
                    return False
                continue
            left = self.settle_left()
            if left <= 0:
                return True
            # idle now, but busy a moment ago: check again settle_seconds after that
            on_waiting(self.peers[0].name)
            if not sleep(left):
                return False
