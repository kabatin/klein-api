"""The real ``flux2-klein`` workflow and real Pillow post-processing, against the fake
backend, over HTTP."""

import base64
import io
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from klein_api.api.app import create_app
from klein_api.jobs.queue import JobQueue
from klein_api.jobs.store import JobStore
from klein_api.workflows.base import WorkflowRegistry
from klein_api.workflows.flux2_klein import Flux2KleinParams, Flux2KleinWorkflow
from tests.conftest import make_png
from tests.fakes import FakeKleinBackend

KLEIN = "/v1/workflows/flux2-klein/jobs"


def make_client(tmp_path: Path, backend: FakeKleinBackend) -> TestClient:
    registry = WorkflowRegistry(
        [Flux2KleinWorkflow(max_reference_images=backend.max_reference_images)]
    )
    queue = JobQueue(
        JobStore(tmp_path / "data"),
        registry,
        backend,
        max_waiting=1,
        timeout_factor=2,
        retention_days=7,
    )
    app = create_app(queue, registry, token=None, max_body_bytes=32 << 20)
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def backend() -> FakeKleinBackend:
    return FakeKleinBackend(png=True, size=(1344, 768), max_refs=2)


@pytest.fixture
def client(tmp_path: Path, backend: FakeKleinBackend) -> Iterator[TestClient]:
    with make_client(tmp_path, backend) as c:
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


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_two_images_with_references_over_http(
    client: TestClient, backend: FakeKleinBackend, tmp_path: Path
) -> None:
    main = b64(make_png(tmp_path / "main.png", (640, 360)))
    jpeg = io.BytesIO()
    Image.new("RGB", (300, 200), "green").save(jpeg, format="JPEG")
    job = client.post(
        KLEIN,
        json={
            "prompt": "波と🌊カモメ, the same red tram",
            "output": {"width": 1920, "height": 1080},
            "quality": "draft",
            "seed": 5,
            "count": 2,
            "reference_images": [
                {"data": main, "media_type": "image/png"},
                {"data": b64(jpeg.getvalue()), "media_type": "image/jpeg"},
            ],
        },
        headers={"Idempotency-Key": "shot-1"},
    ).json()
    body = finish(client, job["id"])
    assert body["status"] == "succeeded", body
    assert body["progress"] == 1.0 and body["waiting_for"] is None
    assert body["result"] == {
        "outputs": [
            {"index": 0, "media_type": "image/png", "width": 1920, "height": 1080, "seed_used": 5},
            {"index": 1, "media_type": "image/png", "width": 1920, "height": 1080, "seed_used": 6},
        ],
        "seed_used": 5,
        "details": {
            "generation": {"width": 1344, "height": 768, "quality": "draft", "references": 2}
        },
    }
    request = backend.calls[0]
    assert (request.count, request.seed, request.quality) == (2, 5, "draft")
    assert request.prompt == "波と🌊カモメ, the same red tram"
    assert request.aspect_ratio == pytest.approx(16 / 9)
    assert [p.name for p in request.references] == ["reference-0.png", "reference-1.png"]
    seen = set()
    for index in (0, 1):
        response = client.get(f"/v1/jobs/{job['id']}/output", params={"index": index})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        seen.add(response.content)
        with Image.open(io.BytesIO(response.content)) as img:
            assert (img.format, img.mode, img.size) == ("PNG", "RGB", (1920, 1080))
            assert img.info["Comment"] == f"klein-job:{job['id']}:{index}"
    assert len(seen) == 2  # byte-unique even though the pixels are identical
    assert client.get(f"/v1/jobs/{job['id']}/output?index=2").status_code == 404
    job_dir = tmp_path / "data" / "jobs" / job["id"]
    assert not list(job_dir.glob("raw-*")) and not (job_dir / "inputs").exists()


def test_text_only_job_and_random_seed(client: TestClient, backend: FakeKleinBackend) -> None:
    job = client.post(
        KLEIN, json={"prompt": "a lighthouse", "output": {"width": 768, "height": 1344}}
    ).json()
    body = finish(client, job["id"])
    assert body["status"] == "succeeded", body
    assert len(body["result"]["outputs"]) == 1
    seed = body["result"]["seed_used"]
    assert 0 <= seed < 2**31 and body["result"]["outputs"][0]["seed_used"] == seed
    assert backend.calls[0].references == () and backend.calls[0].quality == "standard"
    assert backend.calls[0].count == 1


def test_alpha_from_the_engine_is_dropped(client: TestClient, backend: FakeKleinBackend) -> None:
    backend.mode = "rgba"
    job = client.post(KLEIN, json={"prompt": "x", "output": {"width": 512, "height": 512}})
    assert finish(client, job.json()["id"])["status"] == "succeeded"
    png = client.get(f"/v1/jobs/{job.json()['id']}/output").content
    with Image.open(io.BytesIO(png)) as img:
        assert img.mode == "RGB"


@pytest.mark.parametrize(
    ("mode", "code", "retryable"),
    [
        ("short", "generation_failed", True),  # an incomplete set is not published
        ("malformed", "invalid_output", True),
        ("no_output", "generation_failed", True),
        ("timeout", "timeout", True),
    ],
)
def test_broken_backend_output_fails_the_job(
    client: TestClient,
    backend: FakeKleinBackend,
    tmp_path: Path,
    mode: str,
    code: str,
    retryable: bool,
) -> None:
    backend.mode = mode
    job = client.post(
        KLEIN, json={"prompt": "x", "output": {"width": 1024, "height": 1024}, "count": 2}
    ).json()
    body = finish(client, job["id"])
    assert body["status"] == "failed"
    assert (body["error"]["code"], body["error"]["retryable"]) == (code, retryable)
    assert client.get(f"/v1/jobs/{job['id']}/output").status_code == 409
    job_dir = tmp_path / "data" / "jobs" / job["id"]
    assert not list(job_dir.glob("output-*.png")) and not list(job_dir.glob("raw-*"))


def test_too_many_references_for_this_engine(client: TestClient, tmp_path: Path) -> None:
    ref = {"data": b64(make_png(tmp_path / "r.png", (128, 128))), "media_type": "image/png"}
    refused = client.post(
        KLEIN,
        json={
            "prompt": "x",
            "output": {"width": 1024, "height": 1024},
            "reference_images": [ref] * 3,
        },
    )
    assert refused.status_code == 422
    assert "at most 2" in refused.json()["error"]["message"]
    caps = client.get("/v1/capabilities").json()["workflows"][0]
    assert caps["reference_images"]["max"] == 2


def test_undecodable_reference_is_422_and_not_echoed(client: TestClient) -> None:
    body = {
        "prompt": "x",
        "output": {"width": 1024, "height": 1024},
        "reference_images": [{"data": "A" * 4000, "media_type": "image/png"}],
    }
    bad = client.post(KLEIN, json=body)
    error = bad.json()["error"]
    assert bad.status_code == 422 and error["code"] == "invalid_params"
    assert "reference_images[0]" in error["message"]
    assert error["details"][0]["loc"] == ["body", "reference_images", "0"]
    assert "A" * 100 not in bad.text  # never echo inputs back


def test_store_inputs_normalises_references(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° CW on display
    buf = io.BytesIO()
    Image.new("RGB", (4000, 1000), "green").save(buf, format="JPEG", exif=exif.tobytes())
    params = Flux2KleinParams.model_validate(
        {
            "prompt": "  padded  ",
            "output": {"width": 1024, "height": 1024},
            "reference_images": [{"data": b64(buf.getvalue()), "media_type": "image/jpeg"}],
        }
    )
    stored = Flux2KleinWorkflow(max_reference_images=1).store_inputs(params, tmp_path)
    assert stored["prompt"] == "padded" and stored["reference_images"] == ["inputs/reference-0.png"]
    with Image.open(tmp_path / stored["reference_images"][0]) as img:
        # upright (EXIF applied) and scaled to the 2048 px cap, aspect kept
        assert (img.format, img.mode, img.size) == ("PNG", "RGB", (512, 2048))
    assert "data" not in str(stored)
    assert stored["count"] == 1 and isinstance(stored["seed"], int)


def test_build_request_refuses_missing_or_escaping_references(tmp_path: Path) -> None:
    from klein_api.workflows.base import WorkflowFailedError

    workflow = Flux2KleinWorkflow(max_reference_images=1)
    params = {
        "prompt": "x",
        "output": {"width": 1024, "height": 1024},
        "quality": "draft",
        "seed": 1,
        "count": 1,
    }
    for ref in ("inputs/reference-0.png", "../../etc/passwd"):
        with pytest.raises(WorkflowFailedError, match="gone"):
            workflow.build_request("job_x", params | {"reference_images": [ref]}, tmp_path)


def test_capabilities_are_consistent_with_validation() -> None:
    caps = Flux2KleinWorkflow().capabilities()
    base = {"prompt": "x", "output": {"width": 1024, "height": 1024}}
    Flux2KleinParams.model_validate(base | {"count": caps["count"]["max"]})
    with pytest.raises(ValueError, match="count"):
        Flux2KleinParams.model_validate(base | {"count": caps["count"]["max"] + 1})
    out = caps["output"]
    Flux2KleinParams.model_validate(
        base | {"output": {"width": out["width"]["max"], "height": out["height"]["min"] * 4}}
    )
    assert caps["reference_images"]["max"] == 0
    assert set(caps["reference_images"]["media_types"]) == {
        "image/png",
        "image/jpeg",
        "image/webp",
    }
