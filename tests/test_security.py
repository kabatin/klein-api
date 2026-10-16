"""Hardening: loopback-only mode, token rules, idempotency, busy gate, ids, cleanup, images."""

import io
import os
import struct
import time
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from klein_api.api.app import create_app
from klein_api.api.security import host_name
from klein_api.checks import Level
from klein_api.doctor import check_config_file
from klein_api.jobs.queue import JobQueue
from klein_api.jobs.store import JobStore
from klein_api.media import MediaError, normalize_image
from klein_api.settings import SettingsError, load_settings
from klein_api.workflows.base import WorkflowRegistry
from tests.fakes import EchoParams, EchoWorkflow, FakeKleinBackend

TOKEN = "t" * 40


@pytest.fixture
def backend() -> FakeKleinBackend:
    return FakeKleinBackend()


@pytest.fixture
def queue(tmp_path: Path, backend: FakeKleinBackend) -> JobQueue:
    return JobQueue(
        JobStore(tmp_path),
        WorkflowRegistry([EchoWorkflow()]),
        backend,
        max_waiting=1,
        timeout_factor=2,
        retention_days=7,
    )


@pytest.fixture
def client(queue: JobQueue, backend: FakeKleinBackend) -> Iterator[TestClient]:
    app = create_app(queue, WorkflowRegistry([EchoWorkflow()]), token=None, max_body_bytes=1 << 20)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c
    backend.release.set()


# -- loopback-only mode (no token) -------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1:8767", "localhost:8767", "[::1]:8767", "LOCALHOST"])
def test_loopback_hosts_are_served(client: TestClient, host: str) -> None:
    assert client.get("/v1/health", headers={"Host": host}).status_code == 200


def test_rebinding_host_is_refused(client: TestClient) -> None:
    denied = client.get("/v1/health", headers={"Host": "attacker.example"})
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "forbidden_host"


@pytest.mark.parametrize("header", ["X-Forwarded-For", "Forwarded", "X-Real-IP"])
def test_proxied_requests_need_a_token(client: TestClient, header: str) -> None:
    denied = client.get("/v1/health", headers={header: "203.0.113.9"})
    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "proxy_requires_token"


def test_host_name_parsing() -> None:
    assert host_name("[::1]:8767") == "::1"
    assert host_name("localhost:1") == "localhost"
    assert host_name("127.0.0.1") == "127.0.0.1"
    assert host_name("[broken") == "[broken"


def test_body_without_content_type_is_not_json(client: TestClient) -> None:
    # a cross-site "simple" POST (no preflight) must not be accepted as a job
    sent = client.post("/v1/workflows/echo/jobs", content=b'{"text":"x"}')
    assert sent.status_code == 422


# -- tokens and binding --------------------------------------------------------------------


@pytest.mark.parametrize("token", ["short", "has space " * 5, "ü" * 40])
def test_weak_tokens_are_rejected(tmp_path: Path, token: str) -> None:
    with pytest.raises(SettingsError, match="printable ASCII"):
        load_settings(env={"KLEIN_API_TOKEN": token}, config_path=tmp_path / "x")


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::", "mac.local"])
def test_remote_bind_without_token_is_refused(tmp_path: Path, host: str) -> None:
    with pytest.raises(SettingsError, match="KLEIN_API_TOKEN"):
        load_settings(env={"KLEIN_API_HOST": host}, config_path=tmp_path / "x")


def test_bearer_scheme_is_case_insensitive(queue: JobQueue) -> None:
    app = create_app(queue, WorkflowRegistry([EchoWorkflow()]), token=TOKEN, max_body_bytes=1024)
    with TestClient(app) as c:
        assert c.get("/v1/health", headers={"Authorization": f"bearer {TOKEN}"}).status_code == 200
        assert c.get("/v1/health", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
        # with a token, any Host is fine (the LAN case)
        ok = c.get("/v1/health", headers={"Authorization": f"Bearer {TOKEN}", "Host": "mac.lan"})
        assert ok.status_code == 200


# -- idempotency and the busy gate -------------------------------------------------------


def test_idempotent_resubmission_over_http(client: TestClient, backend: FakeKleinBackend) -> None:
    backend.block = True
    headers = {"Idempotency-Key": "gen-42"}
    first = client.post("/v1/workflows/echo/jobs", json={"text": "a"}, headers=headers)
    assert first.status_code == 202
    assert backend.started.wait(2)
    again = client.post("/v1/workflows/echo/jobs", json={"text": "a"}, headers=headers)
    assert again.status_code == 200 and again.json()["id"] == first.json()["id"]
    clash = client.post("/v1/workflows/echo/jobs", json={"text": "b"}, headers=headers)
    assert clash.status_code == 409 and clash.json()["error"]["code"] == "idempotency_conflict"
    bad = client.post(
        "/v1/workflows/echo/jobs",
        json={"text": "a"},
        headers={"Idempotency-Key": "no spaces allowed"},
    )
    assert bad.status_code == 422
    assert len(backend.calls) == 1  # one generation, however often the client retried


def test_busy_gate_answers_before_reading_the_body(
    client: TestClient, backend: FakeKleinBackend
) -> None:
    backend.block = True
    client.post("/v1/workflows/echo/jobs", json={"text": "a"}, headers={"Idempotency-Key": "k1"})
    assert backend.started.wait(2)
    client.post("/v1/workflows/echo/jobs", json={"text": "b"})
    # the body is not even JSON: only a gate in front of parsing can answer 429 here
    busy = client.post(
        "/v1/workflows/echo/jobs", content=b"not json", headers={"Content-Type": "application/json"}
    )
    assert busy.status_code == 429 and "retry-after" in busy.headers
    # a key that already has a job still gets that job back
    known = client.post(
        "/v1/workflows/echo/jobs", json={"text": "a"}, headers={"Idempotency-Key": "k1"}
    )
    assert known.status_code == 200


# -- ids, cleanup ------------------------------------------------------------------------


@pytest.mark.parametrize("job_id", ["job_%00", "job_" + "A" * 300, "..", "job_abc"])
def test_malformed_ids_are_404(client: TestClient, job_id: str) -> None:
    assert client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_output_route_never_reads_other_files(client: TestClient) -> None:
    for path in ("/v1/jobs/..%2F..%2Fetc%2Fpasswd/output", "/v1/jobs/job_../output"):
        assert client.get(path).status_code == 404


def _wait(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.02)


def test_inputs_are_dropped_after_cancel_and_finish(
    queue: JobQueue, backend: FakeKleinBackend, tmp_path: Path
) -> None:
    queue.start()
    try:
        backend.block = True
        running, _ = queue.submit(EchoWorkflow(), EchoParams(text="a"))
        assert backend.started.wait(2)
        waiting, _ = queue.submit(EchoWorkflow(), EchoParams(text="b"))
        queue.cancel(waiting.id)
        assert not (tmp_path / "jobs" / waiting.id / "inputs").exists()
        backend.release.set()
        _wait(lambda: queue.view(running.id).record.status.finished)
        _wait(lambda: not (tmp_path / "jobs" / running.id / "inputs").exists())
        assert (tmp_path / "jobs" / running.id / "job.json").is_file()
    finally:
        queue.stop(timeout_s=5)


# -- images --------------------------------------------------------------------------------


def _png_with_huge_ztxt() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="PNG")
    data = buf.getvalue()
    payload = b"k\x00\x00" + zlib.compress(b"x" * (70 * 1024 * 1024))
    chunk = struct.pack(">I", len(payload)) + b"zTXt" + payload
    chunk += struct.pack(">I", zlib.crc32(b"zTXt" + payload) & 0xFFFFFFFF)
    return data[:33] + chunk + data[33:]  # after IHDR


def _png_header_claiming(width: int, height: int) -> bytes:
    """A PNG whose IHDR claims a huge canvas (a decompression bomb's first bytes)."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr
    chunk += struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
    idat = zlib.compress(b"\x00" * 64)
    chunk += struct.pack(">I", len(idat)) + b"IDAT" + idat
    chunk += struct.pack(">I", zlib.crc32(b"IDAT" + idat) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk


def test_hostile_png_chunk_is_a_media_error() -> None:
    with pytest.raises(MediaError, match="cannot decode image"):
        normalize_image(_png_with_huge_ztxt(), "image/png")


def test_decompression_bomb_is_refused_before_decoding() -> None:
    with pytest.raises(MediaError, match=r"too large|cannot decode"):
        normalize_image(_png_header_claiming(20000, 20000), "image/png")


def test_decode_errors_hide_object_reprs() -> None:
    with pytest.raises(MediaError) as info:
        normalize_image(b"garbage", "image/png")
    assert "0x" not in str(info.value)


# -- config file ----------------------------------------------------------------------------


def test_config_file_permissions(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'token = "{TOKEN}"\n')
    os.chmod(cfg, 0o644)
    assert check_config_file(cfg).level is Level.WARN
    os.chmod(cfg, 0o600)
    assert check_config_file(cfg).level is Level.OK
    assert check_config_file(None).level is Level.OK
