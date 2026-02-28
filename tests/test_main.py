from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_post_jobs_returns_job_id_and_pending():
    with patch("app.main.process_job"):
        response = client.post("/jobs", json={
            "text": "Hello world",
            "categories": ["Greeting"],
            "entities": ["Person"],
        })

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


def test_get_job_returns_status():
    with patch("app.main.process_job"):
        create = client.post("/jobs", json={
            "text": "Test",
            "categories": ["A"],
            "entities": ["B"],
        })

    job_id = create.json()["job_id"]
    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["job_id"] == job_id


def test_get_unknown_job_returns_404():
    response = client.get("/jobs/nonexistent")
    assert response.status_code == 404
