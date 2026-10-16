from datetime import timedelta
from pathlib import Path

import pytest

from klein_api.jobs.models import JobRecord, JobStatus, new_job_id, utcnow
from klein_api.jobs.store import JobStore, StoreLockedError
from klein_api.settings import SettingsError, is_loopback, load_settings

# -- store -----------------------------------------------------------------------------


def test_ids_are_unique_and_sortable() -> None:
    ids = [new_job_id() for _ in range(200)]
    assert len(set(ids)) == 200
    assert all(i.startswith("job_") and len(i) == 30 for i in ids)


def test_save_get_roundtrip(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    record = store.save(JobRecord(workflow="w", params={"a": 1}))
    assert store.get(record.id) == record
    assert not list(store.job_dir(record.id).glob(".*.tmp"))


@pytest.mark.parametrize("bad", ["", "../x", ".hidden", "a/b", "missing"])
def test_get_unknown_or_unsafe(tmp_path: Path, bad: str) -> None:
    assert JobStore(tmp_path).get(bad) is None


def test_corrupt_record_is_ignored(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    record = store.save(JobRecord(workflow="w", params={}))
    (store.job_dir(record.id) / "job.json").write_text("{broken")
    assert store.get(record.id) is None
    assert store.all() == []


def test_transitions_are_immutable_and_timed() -> None:
    record = JobRecord(workflow="w", params={})
    running = record.started(42.0)
    assert record.status is JobStatus.QUEUED and running.status is JobStatus.RUNNING
    assert running.estimate_seconds == 42.0 and "queue_seconds" in running.timings
    done = running.succeeded({"x": 1}, ["output-0.png"], {"backend_seconds": 3.0})
    assert done.output_files == ["output-0.png"]
    assert done.status.finished and done.finished_at is not None
    assert {"queue_seconds", "backend_seconds", "total_seconds"} <= set(done.timings)
    failed = running.failed("c", "m", retryable=True)
    assert failed.error is not None and failed.error.retryable
    assert not running.cancel_requested and running.with_cancel_requested().cancel_requested


def test_recover_after_restart(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    queued = store.save(JobRecord(workflow="w", params={}))
    running = store.save(JobRecord(workflow="w", params={}).started(1))
    done = store.save(JobRecord(workflow="w", params={}).canceled())
    requeue, interrupted = store.recover()
    assert [r.id for r in requeue] == [queued.id]
    assert [r.id for r in interrupted] == [running.id]
    recovered = store.get(running.id)
    assert recovered is not None and recovered.error is not None
    assert recovered.error.code == "server_restarted" and recovered.error.retryable
    assert store.get(done.id) == done
    # nothing is left "running" forever
    assert all(r.status is not JobStatus.RUNNING for r in store.all())


def test_prune_old_finished_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    old = JobRecord(workflow="w", params={}).canceled()
    old = store.save(old.model_copy(update={"finished_at": utcnow() - timedelta(days=10)}))
    fresh = store.save(JobRecord(workflow="w", params={}).canceled())
    queued = store.save(
        JobRecord(workflow="w", params={}, created_at=utcnow() - timedelta(days=30))
    )
    assert store.prune(7) == [old.id]
    assert store.get(old.id) is None
    assert store.get(fresh.id) is not None and store.get(queued.id) is not None


def test_prune_caps_the_number_of_finished_records(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    finished = [store.save(JobRecord(workflow="w", params={}).canceled()) for _ in range(5)]
    queued = store.save(JobRecord(workflow="w", params={}))
    removed = store.prune(7, max_finished=3)
    assert removed == [r.id for r in finished[:2]]  # the oldest go first
    assert store.get(queued.id) is not None and len(store.all()) == 4


def test_recover_cancels_a_queued_job_whose_cancel_was_requested(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    record = store.save(JobRecord(workflow="w", params={}).with_cancel_requested())
    requeue, interrupted = store.recover()
    assert requeue == [] and [r.status for r in interrupted] == [JobStatus.CANCELED]
    assert store.get(record.id).status is JobStatus.CANCELED  # type: ignore[union-attr]


def test_unreadable_record_is_ignored(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    record = store.save(JobRecord(workflow="w", params={}))
    (store.job_dir(record.id) / "job.json").write_bytes(b"\xff\xfe not utf-8")
    assert store.get(record.id) is None


def test_instance_lock(tmp_path: Path) -> None:
    first, second = JobStore(tmp_path), JobStore(tmp_path)
    first.acquire_instance_lock()
    with pytest.raises(StoreLockedError):
        second.acquire_instance_lock()
    first.release_instance_lock()
    second.acquire_instance_lock()
    second.release_instance_lock()


# -- settings --------------------------------------------------------------------------


def test_defaults_without_file_or_env(tmp_path: Path) -> None:
    s = load_settings(env={}, config_path=tmp_path / "none.toml")
    assert (s.host, s.port, s.max_waiting, s.max_body_mb) == ("127.0.0.1", 8767, 2, 64)
    assert s.token is None and s.retention_days == 7 and s.job_timeout_factor == 3.0
    assert s.coordinate_with == () and s.settle_seconds == 30.0
    assert s.data_dir == Path("~/.local/share/klein-api").expanduser()


def test_env_overrides_file(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        'port = 9000\nmax_waiting = 3\ndata_dir = "~/klein-data"\n'
        '[backends.drawthings]\nmodels_dir = "/m"\n'
    )
    s = load_settings(
        env={"KLEIN_API_PORT": "9100", "KLEIN_API_SETTLE_SECONDS": "5"}, config_path=cfg
    )
    assert s.port == 9100 and s.max_waiting == 3 and s.settle_seconds == 5.0
    assert s.data_dir == Path("~/klein-data").expanduser()
    assert s.backend_options() == {"models_dir": "/m"}
    assert s.backend_options("other") == {}


def test_config_path_from_env(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text("port = 9001\n")
    assert load_settings(env={"KLEIN_API_CONFIG": str(cfg)}).port == 9001


def test_blank_env_values_are_ignored(tmp_path: Path) -> None:
    s = load_settings(env={"KLEIN_API_TOKEN": "  "}, config_path=tmp_path / "x.toml")
    assert s.token is None


def test_non_loopback_with_token(tmp_path: Path) -> None:
    token = "test-token-0123456789-abcdefghijklmnop"
    s = load_settings(
        env={"KLEIN_API_HOST": "0.0.0.0", "KLEIN_API_TOKEN": token}, config_path=tmp_path / "x"
    )
    assert s.token is not None and s.token.get_secret_value() == token
    assert token not in repr(s)


def test_overrides_are_revalidated(tmp_path: Path) -> None:
    s = load_settings(env={}, config_path=tmp_path / "x.toml")
    assert s.with_overrides(port=9999, host=None).port == 9999
    with pytest.raises(SettingsError, match="KLEIN_API_TOKEN"):
        s.with_overrides(host="0.0.0.0")


@pytest.mark.parametrize(
    ("env", "needle"),
    [
        ({"KLEIN_API_PORT": "70000"}, "port"),
        ({"KLEIN_API_SETTLE_SECONDS": "-1"}, "settle_seconds"),
        ({"KLEIN_API_MAX_WAITING": "-1"}, "max_waiting"),
    ],
)
def test_invalid_values_are_reported(tmp_path: Path, env: dict, needle: str) -> None:
    with pytest.raises(SettingsError, match=needle):
        load_settings(env=env, config_path=tmp_path / "x.toml")


def test_broken_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text("port = = 1")
    with pytest.raises(SettingsError, match="cannot read"):
        load_settings(env={}, config_path=cfg)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("192.168.0.5", False),
        ("example.com", False),
    ],
)
def test_is_loopback(host: str, expected: bool) -> None:
    assert is_loopback(host) is expected
