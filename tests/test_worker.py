from app.worker import JobStore, JobStatus


def test_create_job_returns_id_and_pending_status():
    store = JobStore()
    job_id = store.create(
        text="Test text",
        categories=["Bug", "Feature"],
        entities=["Person", "Date"],
    )

    job = store.get(job_id)

    assert job is not None
    assert job.status == JobStatus.PENDING
    assert job.result is None


def test_get_unknown_job_returns_none():
    store = JobStore()

    assert store.get("nonexistent") is None


# --- LLM Processing Tests ---

from unittest.mock import patch, MagicMock
from app.worker import process_job, ClassificationResult, CategoryResult, EntityResult


def test_process_job_sets_status_to_completed():
    store = JobStore()
    job_id = store.create(
        text="Die Deutsche Bahn plant Bauarbeiten in Berlin.",
        categories=["Infrastructure", "Finance"],
        entities=["Organization", "Location"],
    )

    fake_result = ClassificationResult(
        categories=[CategoryResult(label="Infrastructure", confidence=0.95)],
        entities=[
            EntityResult(text="Deutsche Bahn", type="Organization", start=4, end=17),
            EntityResult(text="Berlin", type="Location", start=40, end=46),
        ],
    )

    with patch("app.worker._call_llm", return_value=fake_result):
        process_job(store, job_id)

    job = store.get(job_id)
    assert job.status == JobStatus.COMPLETED
    assert len(job.result.categories) == 1
    assert len(job.result.entities) == 2


def test_process_job_sets_status_to_failed_on_error():
    store = JobStore()
    job_id = store.create(text="Test", categories=["A"], entities=["B"])

    with patch("app.worker._call_llm", side_effect=Exception("API down")):
        process_job(store, job_id)

    job = store.get(job_id)
    assert job.status == JobStatus.FAILED
    assert "API down" in job.error
