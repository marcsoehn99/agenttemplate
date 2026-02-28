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
