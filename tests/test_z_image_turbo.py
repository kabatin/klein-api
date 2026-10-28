"""The ``z-image-turbo`` workflow and per-workflow availability, against the fake backend,
over HTTP (with the real registry)."""

import io
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from klein_api.api.app import create_app
from klein_api.jobs.queue import JobQueue
from klein_api.jobs.store import JobStore
from klein_api.workflows import build_registry
from klein_api.workflows.z_image_turbo import ZImageTurboParams, ZImageTurboWorkflow
from tests.fakes import FakeKleinBackend

Z_IMAGE = "/v1/workflows/z-image-turbo/jobs"
KLEIN = "/v1/workflows/flux2-klein/jobs"
BODY = {"prompt": "a lighthouse at dusk", "output": {"width": 1000, "height": 562}, "seed": 9}


@pytest.fixture
def backend() -> FakeKleinBackend:
    return FakeKleinBackend(png=True, size=(1344, 768))


@pytest.fixture
def client(tmp_path: Path, backend: FakeKleinBackend) -> Iterator[TestClient]:
    registry = build_registry(backend)
    queue = JobQueue(
        JobStore(tmp_path / "data"),
        registry,
        backend,
        max_waiting=1,
        timeout_factor=2,
        retention_days=7,
    )
    app = create_app(queue, registry, token=None, max_body_bytes=32 << 20)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c
    backend.release.set()


def finish(client: TestClient, job_id: str, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed", "canceled"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_two_images_at_the_exact_size(client: TestClient, backend: FakeKleinBackend) -> None:
    job = client.post(Z_IMAGE, json=BODY | {"count": 2}).json()
    body = finish(client, job["id"])
    assert body["status"] == "succeeded", body
    request = backend.calls[0]
    assert request.model_id == "z-image-turbo" and request.references == ()
    assert (request.seed, request.count) == (9, 2)
    assert [o["seed_used"] for o in body["result"]["outputs"]] == [9, 10]
    assert body["result"]["details"]["generation"] == {"width": 1344, "height": 768}
    png = client.get(f"/v1/jobs/{job['id']}/output", params={"index": 1}).content
    with Image.open(io.BytesIO(png)) as image:
        assert (image.format, image.mode, image.size) == ("PNG", "RGB", (1000, 562))


@pytest.mark.parametrize(
    "extra",
    [
        {"quality": "draft"},  # one recipe: no tiers
        {"reference_images": []},  # text only
        {"steps": 8},  # never client-chosen
        {"count": 5},
        {"seed": -1},
    ],
)
def test_params_outside_the_contract_are_refused(extra: dict) -> None:
    with pytest.raises(ValidationError):
        ZImageTurboParams.model_validate(BODY | extra)


def test_blank_or_control_prompt_is_refused() -> None:
    for prompt in ("   ", "a\x00b"):
        with pytest.raises(ValidationError):
            ZImageTurboParams.model_validate(BODY | {"prompt": prompt})


def test_capabilities_list_both_workflows(client: TestClient) -> None:
    workflows = {w["id"]: w for w in client.get("/v1/capabilities").json()["workflows"]}
    z_image = workflows["z-image-turbo"]
    assert z_image["typical_seconds"] == 34 and z_image["reference_images"] == {"max": 0}
    assert "quality" not in z_image and workflows["flux2-klein"]["quality"] == [
        "draft",
        "standard",
    ]
    smoke = ZImageTurboWorkflow().smoke_params()
    ZImageTurboParams.model_validate(smoke)


def test_a_missing_model_is_503_for_its_workflow_only(
    client: TestClient, backend: FakeKleinBackend
) -> None:
    backend.problems = {"z-image-turbo": "model files missing: z_image_turbo_1.0_i8x.ckpt"}
    refused = client.post(Z_IMAGE, json=BODY)
    assert refused.status_code == 503
    error = refused.json()["error"]
    assert error["code"] == "workflow_unavailable" and error["retryable"] is False
    assert "z_image_turbo_1.0_i8x.ckpt" in error["message"]
    assert not backend.calls
    health = client.get("/v1/health").json()
    assert health["status"] == "ok"
    both = {"available": True, "reason": None}
    assert health["workflows"] == {
        "flux2-klein": {
            "available": True,
            "reason": None,
            "qualities": {"draft": both, "standard": both},
        },
        "z-image-turbo": {
            "available": False,
            "reason": "model files missing: z_image_turbo_1.0_i8x.ckpt",
            "qualities": None,
        },
    }
    accepted = client.post(KLEIN, json=BODY)
    assert accepted.status_code == 202


def test_flux2_klein_is_available_while_one_tier_is(
    client: TestClient, backend: FakeKleinBackend
) -> None:
    backend.problems = {"klein-9b": "model files missing: flux_2_klein_9b_i8x.ckpt"}
    klein = client.get("/v1/health").json()["workflows"]["flux2-klein"]
    missing_9b = "model files missing: flux_2_klein_9b_i8x.ckpt"
    assert klein == {
        "available": True,
        "reason": f"standard: {missing_9b}",
        # per tier (ixa picks the tier from this): draft = 4B files, standard = 9B files
        "qualities": {
            "draft": {"available": True, "reason": None},
            "standard": {"available": False, "reason": missing_9b},
        },
    }
    standard = client.post(KLEIN, json=BODY)
    assert standard.status_code == 503 and "standard tier" in standard.json()["error"]["message"]
    draft = client.post(KLEIN, json=BODY | {"quality": "draft"})
    assert draft.status_code == 202
    backend.problems["klein-4b"] = "draw-things-cli is not installed"
    klein = client.get("/v1/health").json()["workflows"]["flux2-klein"]
    assert klein["available"] is False and "draft: draw-things-cli" in klein["reason"]
    assert klein["qualities"]["draft"] == {
        "available": False,
        "reason": "draw-things-cli is not installed",
    }
