"""A real draw-things-cli generation through the workflow (opt-in: KLEIN_API_REAL_TESTS=1).

Uses the server's own configuration (``$KLEIN_API_CONFIG`` or ~/.config/klein-api/config.toml),
so it runs the engine and model files the server would use. Not part of CI.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from klein_api.backends import build_backend
from klein_api.media import inspect_png
from klein_api.settings import load_settings
from klein_api.workflows import build_registry

pytestmark = pytest.mark.real


@pytest.mark.parametrize("quality", ["draft", "standard"])
def test_real_generation(tmp_path: Path, quality: str) -> None:
    settings = load_settings()
    backend = build_backend(settings)
    failed = [c for c in backend.health() if c.failed]
    assert not failed, failed
    workflow = next(iter(build_registry(backend)))
    params = workflow.params_model.model_validate(
        workflow.smoke_params()
        | {"quality": quality, "count": 2, "output": {"width": 1920, "height": 1080}}
    )
    job_id = "job_" + "1" * 26
    stored = workflow.store_inputs(params, tmp_path)
    request = workflow.build_request(job_id, stored, tmp_path)
    progress: list[float] = []
    raw = backend.generate(
        request, tmp_path, timeout_s=600, cancel=threading.Event(), on_progress=progress.append
    )
    assert raw.ok, f"{raw.describe_failure()} (log: {raw.log_path})"
    assert progress and progress[-1] == 1.0, "no progress lines were parsed"
    output = workflow.finalize(job_id, stored, tmp_path, raw)
    assert len(output.files) == 2
    for file in output.files:
        info = inspect_png(file)
        assert (info.mode, info.width, info.height) == ("RGB", 1920, 1080)
    assert output.result["details"]["generation"]["width"] % 64 == 0
