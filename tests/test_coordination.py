"""Coordination with other GPU servers: the health probe (against a real local HTTP server),
the double check ``settle_seconds`` apart, and the queue's ``waiting_for``."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from klein_api.api.app import create_app
from klein_api.jobs.coordination import (
    Coordinator,
    PeerHealth,
    PeerUnreachableError,
    probe_health,
)
from klein_api.jobs.models import JobStatus
from klein_api.jobs.queue import JobQueue
from klein_api.jobs.store import JobStore
from klein_api.settings import PeerServer, SettingsError, load_settings
from klein_api.workflows.base import WorkflowRegistry
from tests.fakes import EchoParams, EchoWorkflow, FakeKleinBackend

PEER_TOKEN = "peer-token-0123456789"


class FakePeer:
    """A vpipe-api-like ``/v1/health`` on 127.0.0.1; ``state`` is served as JSON."""

    def __init__(self) -> None:
        self.state: dict = {
            "status": "ok",
            "version": "0.3.0",
            "running": 0,
            "waiting": 0,
            "max_waiting": 2,
        }
        self.status = 200
        self.headers_seen: list[dict[str, str]] = []
        peer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                peer.headers_seen.append(dict(self.headers))
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "http://127.0.0.1:1/steal")
                    self.end_headers()
                    return
                if self.headers.get("Authorization") != f"Bearer {PEER_TOKEN}":
                    self.send_response(401)
                    self.end_headers()
                    return
                body = json.dumps(peer.state).encode()
                self.send_response(peer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def peer(self, path: str = "/v1/health", token: str | None = PEER_TOKEN) -> PeerServer:
        return PeerServer.model_validate(
            {"name": "vpipe-api", "health_url": self.url + path, "token": token}
        )


@pytest.fixture
def fake_peer() -> Iterator[FakePeer]:
    peer = FakePeer()
    yield peer
    peer.server.shutdown()


# -- the probe -------------------------------------------------------------------------


def test_probe_reads_vpipe_health(fake_peer: FakePeer) -> None:
    health = probe_health(fake_peer.peer())
    assert health.idle and (health.running, health.waiting, health.outside_runs) == (0, 0, None)
    assert fake_peer.headers_seen[0]["Authorization"] == f"Bearer {PEER_TOKEN}"
    fake_peer.state |= {"running": 1}
    assert not probe_health(fake_peer.peer()).idle
    fake_peer.state |= {"running": 0, "waiting": 0, "outside_runs": 2}
    assert not probe_health(fake_peer.peer()).idle
    fake_peer.state |= {"outside_runs": 0}
    assert probe_health(fake_peer.peer()).idle


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        ({"token": None}, "HTTP 401"),
        ({"path": "/redirect"}, "HTTP 302"),
    ],
)
def test_probe_failures_are_unreachable(fake_peer: FakePeer, change: dict, needle: str) -> None:
    peer = fake_peer.peer(**change)
    with pytest.raises(PeerUnreachableError, match=needle) as info:
        probe_health(peer)
    assert PEER_TOKEN not in str(info.value)
    # a redirect is never followed, so the token never travels to another host
    assert len(fake_peer.headers_seen) == 1


def test_probe_of_a_bad_body_or_dead_server(fake_peer: FakePeer) -> None:
    fake_peer.state = {"status": "ok"}
    with pytest.raises(PeerUnreachableError, match="unexpected health response"):
        probe_health(fake_peer.peer())
    fake_peer.status = 503
    with pytest.raises(PeerUnreachableError, match="HTTP 503"):
        probe_health(fake_peer.peer())
    dead = PeerServer(name="gone", health_url="http://127.0.0.1:1/v1/health")
    with pytest.raises(PeerUnreachableError):
        probe_health(dead, timeout_s=1)


# -- settings ----------------------------------------------------------------------------


def test_coordinate_with_from_the_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        "settle_seconds = 10\n"
        "[[coordinate_with]]\n"
        'name = "vpipe-api"\nhealth_url = "http://127.0.0.1:8765/v1/health"\n'
        f'token = "{PEER_TOKEN}"\n'
        "[[coordinate_with]]\n"
        'name = "wan-api"\nhealth_url = "http://127.0.0.1:8766/v1/health"\n'
    )
    s = load_settings(env={}, config_path=cfg)
    assert [p.name for p in s.coordinate_with] == ["vpipe-api", "wan-api"]
    assert s.settle_seconds == 10.0 and s.coordinate_with[1].token is None
    assert PEER_TOKEN not in repr(s)
    assert s.with_overrides(port=9999).coordinate_with == s.coordinate_with


@pytest.mark.parametrize(
    ("entry", "needle"),
    [
        ('name = "x"\nhealth_url = "file:///etc/passwd"', "http"),
        ('name = "x"\nhealth_url = "http://user:pw@host/v1/health"', "credentials"),
        ('name = "bad name"\nhealth_url = "http://h/v1/health"', "name"),
        ('name = "x"\nhealth_url = "http://h/v1/health"\ntoken = "has space"', "token"),
        ('name = "x"\nhealth_url = "http://h/"\nextra = 1', "extra"),
    ],
)
def test_bad_peers_are_refused(tmp_path: Path, entry: str, needle: str) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text(f"[[coordinate_with]]\n{entry}\n")
    with pytest.raises(SettingsError, match=needle):
        load_settings(env={}, config_path=cfg)


def test_peer_names_must_be_unique(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    entry = '[[coordinate_with]]\nname = "x"\nhealth_url = "http://h/v1/health"\n'
    cfg.write_text(entry * 2)
    with pytest.raises(SettingsError, match="unique"):
        load_settings(env={}, config_path=cfg)


# -- the settle rule ---------------------------------------------------------------------


class ScriptedProbe:
    """Answers from a list of states per call (the last one repeats)."""

    def __init__(self, *states: PeerHealth | Exception) -> None:
        self.states = list(states)
        self.calls = 0

    def __call__(self, peer: PeerServer) -> PeerHealth:
        state = self.states[min(self.calls, len(self.states) - 1)]
        self.calls += 1
        if isinstance(state, Exception):
            raise state
        return state


class FakeClock:
    """Monotonic time that only moves when the coordinator sleeps (or the test says so)."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


IDLE = PeerHealth(running=0, waiting=0)
BUSY = PeerHealth(running=1, waiting=0)
PEER = PeerServer(name="vpipe-api", health_url="http://127.0.0.1:8765/v1/health")


def coordinator_with(
    probe, clock: FakeClock, *, peers=(PEER,), settle: float = 30, poll: float = 5
) -> Coordinator:
    return Coordinator(peers, settle_seconds=settle, poll_seconds=poll, probe=probe, clock=clock)


def run_wait(
    coordinator: Coordinator, clock: FakeClock
) -> tuple[bool, list[float], list[str], float]:
    """Returns (started, sleeps, waiting_for values, clock time when it returned)."""
    sleeps: list[float] = []
    waiting: list[str] = []

    def sleep(seconds: float) -> bool:
        sleeps.append(round(seconds, 3))
        clock.now += seconds
        return True

    ok = coordinator.wait_until_idle("job_x", sleep=sleep, on_waiting=waiting.append)
    return ok, sleeps, waiting, clock.now


def test_first_ever_idle_check_starts_immediately() -> None:
    """Guards against the old rule (always a second check settle_seconds later)."""
    clock, probe = FakeClock(), ScriptedProbe(IDLE)
    ok, sleeps, waiting, _ = run_wait(coordinator_with(probe, clock), clock)
    assert ok and probe.calls == 1 and sleeps == [] and waiting == []


def test_idle_long_after_the_last_busy_sighting_starts_immediately() -> None:
    clock, probe = FakeClock(), ScriptedProbe(BUSY, IDLE)
    coordinator = coordinator_with(probe, clock)
    assert run_wait(coordinator, clock)[0]  # first job: busy, then settled
    clock.now += 31  # next job arrives more than settle_seconds after the last busy sighting
    calls = probe.calls
    ok, sleeps, waiting, _ = run_wait(coordinator, clock)
    assert ok and probe.calls == calls + 1 and sleeps == [] and waiting == []


def test_busy_then_idle_waits_settle_seconds_after_the_busy_sighting() -> None:
    """Guards against a rule without settling (start on the first idle check)."""
    clock, probe = FakeClock(), ScriptedProbe(BUSY, IDLE)
    busy_at = clock.now
    ok, sleeps, waiting, ended = run_wait(coordinator_with(probe, clock, poll=5), clock)
    # busy at t, idle at t+5 (too soon), idle again at t+30 -> start
    assert ok and probe.calls == 3 and sleeps == [5.0, 25.0]
    assert ended == busy_at + 30
    assert waiting == ["vpipe-api", "vpipe-api"]


def test_a_recent_busy_sighting_from_an_earlier_wait_still_counts() -> None:
    clock, probe = FakeClock(), ScriptedProbe(BUSY)
    coordinator = coordinator_with(probe, clock, poll=5)
    # job A sees the peer busy at t=1000 and is canceled while waiting
    abandoned = coordinator.wait_until_idle("a", sleep=lambda _s: False, on_waiting=print)
    assert not abandoned and coordinator.last_busy == 1000.0
    # job B checks at t=1010: the peer is idle, but was busy 10 s ago -> wait until t=1030
    clock.now = 1010.0
    probe.states, probe.calls = [IDLE], 0
    ok, sleeps, waiting, ended = run_wait(coordinator, clock)
    assert ok and probe.calls == 2 and sleeps == [20.0] and ended == 1030.0
    assert waiting == ["vpipe-api"]


def test_busy_again_during_the_settle_wait_moves_the_window() -> None:
    clock, probe = FakeClock(), ScriptedProbe(BUSY, IDLE, BUSY, IDLE, IDLE)
    ok, sleeps, _, ended = run_wait(coordinator_with(probe, clock, poll=5), clock)
    # busy t, idle t+5, busy t+30, idle t+35, idle t+60 -> start
    assert ok and probe.calls == 5 and sleeps == [5.0, 25.0, 5.0, 25.0]
    assert ended == 1060.0


def test_zero_settle_starts_on_the_first_idle_check() -> None:
    clock, probe = FakeClock(), ScriptedProbe(BUSY, IDLE)
    ok, sleeps, _, _ = run_wait(coordinator_with(probe, clock, settle=0, poll=5), clock)
    assert ok and sleeps == [5.0]


def test_unreachable_counts_as_idle(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    probe = ScriptedProbe(PeerUnreachableError("ConnectionRefusedError"))
    ok, sleeps, _, _ = run_wait(coordinator_with(probe, clock), clock)
    assert ok and probe.calls == 1 and sleeps == []
    assert "coordination.unreachable" in caplog.text


def test_every_peer_must_be_idle() -> None:
    wan = PeerServer(name="wan-api", health_url="http://127.0.0.1:8766/v1/health")
    states = {"vpipe-api": [IDLE], "wan-api": [BUSY, IDLE]}

    def probe(peer: PeerServer) -> PeerHealth:
        queue = states[peer.name]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    clock = FakeClock()
    ok, sleeps, waiting, _ = run_wait(
        coordinator_with(probe, clock, peers=(PEER, wan), settle=10, poll=2), clock
    )
    assert ok and waiting[0] == "wan-api" and sleeps == [2.0, 8.0]


def test_an_abandoned_wait_returns_false() -> None:
    coordinator = Coordinator([PEER], settle_seconds=30, probe=ScriptedProbe(BUSY))
    assert not coordinator.wait_until_idle("j", sleep=lambda _s: False, on_waiting=lambda _n: None)
    settling = Coordinator([PEER], settle_seconds=30, probe=ScriptedProbe(BUSY, IDLE))
    assert not settling.wait_until_idle(
        "j", sleep=lambda s: s < 10, on_waiting=lambda _n: None
    )  # poll sleep ok, settle sleep abandoned
    assert Coordinator([]).wait_until_idle("j", sleep=lambda _s: False, on_waiting=print)


# -- through the queue -------------------------------------------------------------------


class Switch:
    """A probe the test flips between busy and idle."""

    def __init__(self) -> None:
        self.busy = True
        self.calls = 0

    def __call__(self, peer: PeerServer) -> PeerHealth:
        self.calls += 1
        return BUSY if self.busy else IDLE


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


@pytest.fixture
def switch() -> Switch:
    return Switch()


@pytest.fixture
def backend() -> FakeKleinBackend:
    return FakeKleinBackend()


@pytest.fixture
def client(tmp_path: Path, backend: FakeKleinBackend, switch: Switch) -> Iterator[TestClient]:
    registry = WorkflowRegistry([EchoWorkflow()])
    coordinator = Coordinator([PEER], settle_seconds=0.4, poll_seconds=0.05, probe=switch)
    queue = JobQueue(
        JobStore(tmp_path),
        registry,
        backend,
        max_waiting=1,
        timeout_factor=2,
        retention_days=7,
        coordinator=coordinator,
    )
    app = create_app(
        queue, registry, token=None, max_body_bytes=1 << 20, waits_for=coordinator.names
    )
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c
    backend.release.set()


def test_a_job_waits_queued_while_another_gpu_server_is_busy(
    client: TestClient, backend: FakeKleinBackend, switch: Switch
) -> None:
    job = client.post("/v1/workflows/echo/jobs", json={"text": "a"}).json()
    wait_for(lambda: client.get(f"/v1/jobs/{job['id']}").json()["waiting_for"] == "vpipe-api")
    body = client.get(f"/v1/jobs/{job['id']}").json()
    assert (body["status"], body["queue_position"], body["progress"]) == ("queued", 1, None)
    # waiting for another server is not running: the slot is not claimed
    assert client.get("/v1/health").json() | {} == {
        "status": "ok",
        "version": "0.1.0",
        "running": 0,
        "waiting": 1,
        "max_waiting": 1,
    }
    assert not backend.calls
    time.sleep(0.3)
    assert not backend.calls
    switch.busy = False
    wait_for(lambda: client.get(f"/v1/jobs/{job['id']}").json()["status"] == "succeeded")
    done = client.get(f"/v1/jobs/{job['id']}").json()
    assert done["waiting_for"] is None
    timings = done["timings"]
    # busy for >= 0.3 s, then started no sooner than 0.4 s after the last busy sighting
    assert timings["coordination_seconds"] >= 0.6
    assert timings["queue_seconds"] < timings["coordination_seconds"]
    assert timings["total_seconds"] >= timings["queue_seconds"] + timings["coordination_seconds"]
    assert client.get("/v1/capabilities").json()["coordination"] == {"waits_for": ["vpipe-api"]}


def test_canceling_a_coordinating_job_is_immediate(
    client: TestClient, backend: FakeKleinBackend, switch: Switch
) -> None:
    job = client.post("/v1/workflows/echo/jobs", json={"text": "a"}).json()
    wait_for(lambda: client.get(f"/v1/jobs/{job['id']}").json()["waiting_for"] is not None)
    canceled = client.delete(f"/v1/jobs/{job['id']}")
    assert canceled.status_code == 200 and canceled.json()["status"] == "canceled"
    assert canceled.json()["waiting_for"] is None
    # the next job is the head now and goes through once the peer is idle
    switch.busy = False
    second = client.post("/v1/workflows/echo/jobs", json={"text": "b"}).json()
    wait_for(lambda: client.get(f"/v1/jobs/{second['id']}").json()["status"] == "succeeded")
    assert [c.prompt for c in backend.calls] == ["b"]


def test_stopping_the_server_ends_a_coordination_wait(
    tmp_path: Path, backend: FakeKleinBackend, switch: Switch
) -> None:
    q = JobQueue(
        JobStore(tmp_path),
        WorkflowRegistry([EchoWorkflow()]),
        backend,
        max_waiting=1,
        timeout_factor=2,
        retention_days=7,
        coordinator=Coordinator([PEER], settle_seconds=60, poll_seconds=60, probe=switch),
    )
    q.start()
    record, _ = q.submit(EchoWorkflow(), EchoParams(text="a"))
    wait_for(lambda: q.view(record.id).waiting_for is not None)
    started = time.monotonic()
    q.stop(timeout_s=5)
    assert time.monotonic() - started < 2 and not q.worker_alive
    assert q.view(record.id).record.status is JobStatus.QUEUED  # runs again after a restart
    assert not backend.calls


def test_without_peers_there_is_no_coordination(tmp_path: Path, backend: FakeKleinBackend) -> None:
    probe = ScriptedProbe(BUSY)
    q = JobQueue(
        JobStore(tmp_path),
        WorkflowRegistry([EchoWorkflow()]),
        backend,
        max_waiting=1,
        timeout_factor=2,
        retention_days=7,
        coordinator=Coordinator([], probe=probe),
    )
    q.start()
    try:
        record, _ = q.submit(EchoWorkflow(), EchoParams(text="a"))
        wait_for(lambda: q.view(record.id).record.status is JobStatus.SUCCEEDED)
        assert probe.calls == 0
        assert q.view(record.id).record.timings["coordination_seconds"] == 0.0
    finally:
        q.stop(timeout_s=5)


def test_idle_peers_do_not_delay_a_job(tmp_path: Path, backend: FakeKleinBackend) -> None:
    """With settle_seconds = 60 and peers that were never busy, the job starts at once."""
    probe = ScriptedProbe(IDLE)
    q = JobQueue(
        JobStore(tmp_path),
        WorkflowRegistry([EchoWorkflow()]),
        backend,
        max_waiting=1,
        timeout_factor=2,
        retention_days=7,
        coordinator=Coordinator([PEER], settle_seconds=60, poll_seconds=60, probe=probe),
    )
    q.start()
    try:
        record, _ = q.submit(EchoWorkflow(), EchoParams(text="a"))
        wait_for(lambda: q.view(record.id).record.status is JobStatus.SUCCEEDED)
        assert probe.calls == 1
        assert q.view(record.id).record.timings["coordination_seconds"] < 5
    finally:
        q.stop(timeout_s=5)
