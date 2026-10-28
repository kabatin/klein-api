import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from klein_api.api.app import create_app
from klein_api.jobs.queue import JobQueue
from klein_api.jobs.store import JobStore
from klein_api.workflows.base import WorkflowRegistry
from klein_api.workflows.flux2_klein import Flux2KleinWorkflow
from tests.fakes import EchoWorkflow, FakeKleinBackend

TOKEN = "test-token-0123456789-abcdefghijklmnop"
KLEIN = "/v1/workflows/flux2-klein/jobs"
SIZE = {"width": 1920, "height": 1080}


def make_client(
    tmp_path: Path,
    backend: FakeKleinBackend,
    token: str | None = None,
    max_body: int = 1 << 20,
    waits_for: list[str] | None = None,
) -> TestClient:
    registry = WorkflowRegistry([EchoWorkflow(), Flux2KleinWorkflow()])
    queue = JobQueue(
        JobStore(tmp_path), registry, backend, max_waiting=1, timeout_factor=2, retention_days=7
    )
    app = create_app(queue, registry, token=token, max_body_bytes=max_body, waits_for=waits_for)
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def backend() -> FakeKleinBackend:
    return FakeKleinBackend()


@pytest.fixture
def client(tmp_path: Path, backend: FakeKleinBackend) -> Iterator[TestClient]:
    with make_client(tmp_path, backend) as c:
        yield c
    backend.release.set()


def poll(client: TestClient, job_id: str, want: str, timeout: float = 5) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] == want:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job never reached {want}")


def test_health_matches_vpipe_api_shape(client: TestClient) -> None:
    expected = {
        "status": "ok",
        "version": "0.1.0",
        "running": 0,
        "waiting": 0,
        "max_waiting": 1,
        "workflows": {
            "flux2-klein": {"available": True, "reason": None},
            "echo": {"available": True, "reason": None},
        },
    }
    assert client.get("/v1/health").json() == expected
    assert client.get("/healthz").json() == expected


def test_capabilities_match_the_contract(tmp_path: Path, backend: FakeKleinBackend) -> None:
    with make_client(tmp_path, backend, waits_for=["vpipe-api"]) as c:
        body = c.get("/v1/capabilities").json()
    assert body["version"] == "0.1.0"
    assert body["queue"] == {"max_running": 1, "max_waiting": 1}
    assert body["coordination"] == {"waits_for": ["vpipe-api"]}
    klein = next(w for w in body["workflows"] if w["id"] == "flux2-klein")
    assert klein == {
        "id": "flux2-klein",
        "output": {
            "media_type": "image/png",
            "width": {"min": 256, "max": 2048, "multiple_of": 2},
            "height": {"min": 256, "max": 2048, "multiple_of": 2},
            "aspect_ratio": {"min": "9:21", "max": "21:9"},
            "fit": "cover (generated at a supported size, then centre-cropped and resized)",
        },
        "count": {"min": 1, "max": 4},
        "quality": ["draft", "standard"],
        "seed": True,
        "reference_images": {
            "max": 0,
            "media_types": ["image/jpeg", "image/png", "image/webp"],
            "max_bytes": 20971520,
            "max_pixels": 50000000,
        },
        "typical_seconds": {"draft": 15, "standard": 26},
    }
    text = str(body).lower()
    for secret in ("ckpt", "steps", "draw", "i8x", "sampler", "cfg", "9b", "4b"):
        assert secret not in text


def test_no_coordination_is_an_empty_list(client: TestClient) -> None:
    assert client.get("/v1/capabilities").json()["coordination"] == {"waits_for": []}


def test_workflows_and_openapi(client: TestClient) -> None:
    flows = {w["id"]: w for w in client.get("/v1/workflows").json()["workflows"]}
    assert set(flows) == {"echo", "flux2-klein"}
    props = flows["flux2-klein"]["params_schema"]["properties"]
    assert set(props) == {"prompt", "output", "quality", "seed", "count", "reference_images"}
    assert flows["flux2-klein"]["output_media_type"] == "image/png"
    spec = client.get("/openapi.json").json()
    route = spec["paths"][KLEIN]["post"]
    schema_ref = route["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert schema_ref.endswith("Flux2KleinParams")
    assert {"422", "429"} <= set(route["responses"])
    assert client.get("/docs").status_code == 200


def test_full_job_flow(client: TestClient) -> None:
    accepted = client.post("/v1/workflows/echo/jobs", json={"text": "hi"})
    assert accepted.status_code == 202
    job = accepted.json()
    assert set(job) == {"id", "workflow", "status", "created_at"}
    assert (job["workflow"], job["status"]) == ("echo", "queued")
    done = poll(client, job["id"], "succeeded")
    assert done["result"] == {"length": 2} and done["error"] is None
    assert done["progress"] == 1.0 and done["cancel_requested"] is False
    assert done["waiting_for"] is None and done["queue_position"] is None
    assert {
        "queue_seconds",
        "coordination_seconds",
        "backend_seconds",
        "postprocess_seconds",
        "total_seconds",
    } <= set(done["timings"])
    assert done["timings"]["coordination_seconds"] == 0.0
    out = client.get(f"/v1/jobs/{job['id']}/output")
    assert out.status_code == 200 and out.text == "HI"
    assert out.headers["content-type"].startswith("text/plain")
    assert client.get(f"/v1/jobs/{job['id']}/output?index=0").text == "HI"
    missing = client.get(f"/v1/jobs/{job['id']}/output?index=1")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    assert client.get(f"/v1/jobs/{job['id']}/output?index=-1").status_code == 404
    assert client.get(f"/v1/jobs/{job['id']}/output?index=x").status_code == 422
    assert client.delete(f"/v1/jobs/{job['id']}").json()["error"]["code"] == "conflict"


def test_failed_job_over_http(client: TestClient, backend: FakeKleinBackend) -> None:
    backend.mode = "fail"
    job = client.post("/v1/workflows/echo/jobs", json={"text": "x"}).json()
    body = poll(client, job["id"], "failed")
    assert body["error"]["code"] == "generation_failed" and body["error"]["retryable"]
    assert body["result"] is None
    # 409 before the job succeeded, whatever the index
    assert client.get(f"/v1/jobs/{job['id']}/output").status_code == 409
    assert client.get(f"/v1/jobs/{job['id']}/output?index=7").status_code == 409


def test_errors_use_the_envelope(client: TestClient) -> None:
    missing = client.get("/v1/jobs/job_nope")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    assert client.get("/v1/jobs/job_nope/output").status_code == 404
    assert client.delete("/v1/jobs/job_nope").status_code == 404
    assert client.post("/v1/workflows/nope/jobs", json={}).status_code == 404
    rejected = client.post("/v1/workflows/echo/jobs", json={"text": "x", "reject": True})
    assert rejected.status_code == 422 and "rejected" in rejected.json()["error"]["message"]
    assert rejected.json()["error"]["details"] == [{"loc": ["body", "reject"], "msg": "no"}]
    assert client.get("/v1/nothing-here").json()["error"]["code"] == "not_found"


def _image(data: str = "aGVsbG8=", media_type: str = "image/png") -> dict:
    return {"data": data, "media_type": media_type}


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "x", "output": SIZE, "negative_prompt": "blurry"},
        {"prompt": "x", "output": SIZE, "steps": 8},
        {"prompt": "x", "output": SIZE, "cfg": 3.5},
        {"prompt": "x", "output": SIZE, "model": "a.ckpt"},
        {"prompt": "x", "output": SIZE, "sampler": "euler"},
        {"prompt": "x", "output": SIZE, "loras": ["x"]},
        {"prompt": "x", "output": SIZE, "strength": 0.5},
        {"prompt": "x", "output": SIZE, "output_path": "/tmp/x"},
        {"prompt": "x", "output": {**SIZE, "path": "/tmp"}},
        {"prompt": " ", "output": SIZE},
        {"prompt": "", "output": SIZE},
        {"prompt": "x" * 4001, "output": SIZE},
        {"prompt": "x"},
        {"prompt": "x", "output": {"width": 1921, "height": 1080}},
        {"prompt": "x", "output": {"width": 254, "height": 256}},
        {"prompt": "x", "output": {"width": 2050, "height": 2048}},
        {"prompt": "x", "output": {"width": 2048, "height": 876}},  # wider than 21:9
        {"prompt": "x", "output": {"width": 876, "height": 2048}},  # taller than 9:21
        {"prompt": "x", "output": SIZE, "quality": "ultra"},
        {"prompt": "x", "output": SIZE, "seed": -1},
        {"prompt": "x", "output": SIZE, "seed": 2**31},
        {"prompt": "x", "output": SIZE, "count": 0},
        {"prompt": "x", "output": SIZE, "count": 5},
        {"prompt": "x", "output": SIZE, "reference_images": [_image("!!notbase64!!")]},
        {"prompt": "x", "output": SIZE, "reference_images": [_image(media_type="image/gif")]},
        {"prompt": "x", "output": SIZE, "reference_images": [_image()] * 5},
    ],
)
def test_request_validation(client: TestClient, body: dict) -> None:
    response = client.post(KLEIN, json=body)
    error = response.json()["error"]
    assert response.status_code == 422 and error["code"] == "invalid_params"
    assert all(set(d) == {"loc", "msg"} for d in error["details"] or [])


@pytest.mark.parametrize(
    "size",
    [
        {"width": 2048, "height": 878},  # just inside 21:9
        {"width": 878, "height": 2048},
        {"width": 256, "height": 256},
        {"width": 2048, "height": 2048},
    ],
)
def test_size_bounds_are_accepted(client: TestClient, size: dict) -> None:
    assert client.post(KLEIN, json={"prompt": "x", "output": size}).status_code == 202


def test_references_are_refused_when_the_engine_takes_none(
    client: TestClient, backend: FakeKleinBackend, tmp_path: Path
) -> None:
    import base64

    from tests.conftest import make_png

    data = base64.b64encode(make_png(tmp_path / "r.png", (256, 256))).decode()
    refused = client.post(
        KLEIN, json={"prompt": "x", "output": SIZE, "reference_images": [_image(data)]}
    )
    error = refused.json()["error"]
    assert refused.status_code == 422 and error["code"] == "invalid_params"
    assert error["details"] == [
        {
            "loc": ["body", "reference_images"],
            "msg": "at most 0 reference image(s) (see /v1/capabilities)",
        }
    ]
    assert data[:100] not in refused.text
    assert not backend.calls
    assert not any((tmp_path / "jobs").iterdir())


def test_busy_returns_429_with_retry_after(client: TestClient, backend: FakeKleinBackend) -> None:
    backend.block = True
    first = client.post("/v1/workflows/echo/jobs", json={"text": "a"}).json()
    assert backend.started.wait(2)
    second = client.post("/v1/workflows/echo/jobs", json={"text": "b"}).json()
    running = client.get(f"/v1/jobs/{first['id']}").json()
    queued = client.get(f"/v1/jobs/{second['id']}").json()
    assert (running["status"], running["queue_position"]) == ("running", None)
    assert (queued["status"], queued["queue_position"]) == ("queued", 1)
    assert client.get("/v1/health").json()["running"] == 1
    busy = client.post("/v1/workflows/echo/jobs", json={"text": "c"})
    assert busy.status_code == 429
    assert busy.json()["error"] == {
        "code": "busy",
        "message": "the GPU slot and the waiting queue are full",
        "retryable": True,
        "details": None,
    }
    assert int(busy.headers["retry-after"]) >= 30
    assert client.get(f"/v1/jobs/{first['id']}/output").status_code == 409
    assert client.delete(f"/v1/jobs/{second['id']}").json()["status"] == "canceled"
    stopped = client.delete(f"/v1/jobs/{first['id']}")
    assert stopped.status_code == 200 and stopped.json()["status"] == "canceled"


def test_cancel_of_a_slow_stop_is_202(
    client: TestClient, backend: FakeKleinBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("klein_api.jobs.queue.CANCEL_WAIT_S", 0.2)
    backend.mode = "ignore_cancel"
    backend.block = True
    job = client.post("/v1/workflows/echo/jobs", json={"text": "a"}).json()
    assert backend.started.wait(2)
    pending = client.delete(f"/v1/jobs/{job['id']}")
    assert pending.status_code == 202
    assert pending.json()["status"] == "running" and pending.json()["cancel_requested"]
    backend.release.set()
    final = poll(client, job["id"], "canceled")
    assert final["result"] is None
    assert client.get(f"/v1/jobs/{job['id']}/output").status_code == 409


def test_bearer_auth_everywhere(tmp_path: Path, backend: FakeKleinBackend) -> None:
    with make_client(tmp_path, backend, token=TOKEN) as c:
        for path in ("/healthz", "/v1/health", "/v1/capabilities", "/openapi.json", "/docs"):
            denied = c.get(path)
            assert denied.status_code == 401 and denied.headers["www-authenticate"] == "Bearer"
            assert denied.json()["error"]["code"] == "unauthorized"
        assert c.post(KLEIN, json={}).status_code == 401
        assert c.get("/v1/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = c.get("/v1/health", headers={"Authorization": f"Bearer {TOKEN}"})
        assert ok.status_code == 200


def test_body_limits(tmp_path: Path, backend: FakeKleinBackend) -> None:
    with make_client(tmp_path, backend, max_body=100) as c:
        big = c.post("/v1/workflows/echo/jobs", json={"text": "x" * 500})
        assert big.status_code == 413 and big.json()["error"]["code"] == "payload_too_large"
        chunked = c.post(
            "/v1/workflows/echo/jobs",
            content=iter([b'{"text":"x"}']),
            headers={"Content-Type": "application/json"},
        )
        assert chunked.status_code == 411
        assert chunked.json()["error"]["code"] == "length_required"
