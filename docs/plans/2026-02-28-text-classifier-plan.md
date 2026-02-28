# Text Classifier & Entity Extractor - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** FastAPI-Service mit Polling-Architektur der Text kategorisiert und Entitäten extrahiert via LLM (Instructor + OpenAI).

**Architecture:** POST /jobs nimmt Text + gewünschte Kategorien/Entitäten entgegen, startet Background-Processing via Instructor/OpenAI, speichert Ergebnis in In-Memory Dict. GET /jobs/{id} liefert Status + Ergebnis.

**Tech Stack:** Python 3.12+, FastAPI, Uvicorn, Instructor, OpenAI SDK, Pydantic v2, pytest, httpx

---

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `app/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "text-classifier"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "instructor>=1.7.0",
    "openai>=1.60.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "httpx>=0.28.0",
    "pytest-asyncio>=0.25.0",
]
```

**Step 2: Create .env.example**

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
```

**Step 3: Create empty app/__init__.py**

Empty file.

**Step 4: Install dependencies**

Run: `pip install -e ".[dev]"`
Expected: All packages install successfully

**Step 5: Commit**

```bash
git init
git add pyproject.toml .env.example app/__init__.py
git commit -m "chore: project setup with dependencies"
```

---

### Task 2: Worker - Pydantic Models + Job Store

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_worker.py`
- Create: `app/worker.py`

**Step 1: Write test for job store**

```python
# tests/test_worker.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_worker.py -v`
Expected: FAIL - cannot import JobStore

**Step 3: Implement JobStore and models**

```python
# app/worker.py
import uuid
from enum import Enum
from threading import Lock
from pydantic import BaseModel


# --- Status ---

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Result Models ---

class CategoryResult(BaseModel):
    label: str
    confidence: float


class EntityResult(BaseModel):
    text: str
    type: str
    start: int
    end: int


class ClassificationResult(BaseModel):
    categories: list[CategoryResult]
    entities: list[EntityResult]


# --- Job ---

class Job(BaseModel):
    job_id: str
    status: JobStatus
    text: str
    categories: list[str]
    entities: list[str]
    result: ClassificationResult | None = None
    error: str | None = None


# --- Store ---

class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def create(self, text: str, categories: list[str], entities: list[str]) -> str:
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            text=text,
            categories=categories,
            entities=entities,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = status

    def complete(self, job_id: str, result: ClassificationResult) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = JobStatus.COMPLETED
                self._jobs[job_id].result = result

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = JobStatus.FAILED
                self._jobs[job_id].error = error
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_worker.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add app/worker.py tests/
git commit -m "feat: job store with thread-safe in-memory storage"
```

---

### Task 3: Worker - LLM Processing

**Files:**
- Modify: `tests/test_worker.py`
- Modify: `app/worker.py`

**Step 1: Write test for LLM processing (mocked)**

Append to `tests/test_worker.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_worker.py -v`
Expected: FAIL - cannot import process_job

**Step 3: Implement process_job and _call_llm**

Append to `app/worker.py`:

```python
import os
import instructor
from openai import OpenAI


def _create_llm_client() -> instructor.Instructor:
    return instructor.from_openai(OpenAI())


def _build_system_prompt(categories: list[str], entities: list[str]) -> str:
    return (
        "You are a text classifier and entity extractor.\n\n"
        f"Classify the text into these categories: {', '.join(categories)}.\n"
        "For each matching category, provide a confidence score between 0 and 1.\n\n"
        f"Extract these entity types: {', '.join(entities)}.\n"
        "For each entity, provide the exact text, its type, "
        "and the start/end character positions in the original text.\n\n"
        "Only return categories and entities that are actually present in the text."
    )


def _call_llm(text: str, categories: list[str], entities: list[str]) -> ClassificationResult:
    client = _create_llm_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    return client.chat.completions.create(
        model=model,
        response_model=ClassificationResult,
        messages=[
            {"role": "system", "content": _build_system_prompt(categories, entities)},
            {"role": "user", "content": text},
        ],
    )


def process_job(store: JobStore, job_id: str) -> None:
    job = store.get(job_id)
    if job is None:
        return

    store.update_status(job_id, JobStatus.PROCESSING)

    try:
        result = _call_llm(job.text, job.categories, job.entities)
        store.complete(job_id, result)
    except Exception as e:
        store.fail(job_id, str(e))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_worker.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add app/worker.py tests/test_worker.py
git commit -m "feat: LLM processing with instructor and OpenAI"
```

---

### Task 4: FastAPI App + Routes

**Files:**
- Create: `tests/test_main.py`
- Create: `app/main.py`

**Step 1: Write tests for API endpoints**

```python
# tests/test_main.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL - cannot import app

**Step 3: Implement FastAPI app**

```python
# app/main.py
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.worker import JobStore, JobStatus, process_job


# --- Request / Response Models ---

class JobRequest(BaseModel):
    text: str
    categories: list[str]
    entities: list[str]


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: dict | None = None
    error: str | None = None


# --- App ---

app = FastAPI(title="Text Classifier & Entity Extractor")
store = JobStore()


# --- Routes ---

@app.post("/jobs")
def create_job(request: JobRequest) -> JobResponse:
    job_id = store.create(
        text=request.text,
        categories=request.categories,
        entities=request.entities,
    )

    thread = threading.Thread(target=process_job, args=(store, job_id))
    thread.start()

    return JobResponse(job_id=job_id, status=JobStatus.PENDING)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JobResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        result=job.result.model_dump() if job.result else None,
        error=job.error,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: 3 passed

**Step 5: Run all tests**

Run: `pytest -v`
Expected: 7 passed

**Step 6: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: FastAPI app with polling endpoints"
```

---

### Task 5: Smoke Test - Run Server & Test Manually

**Step 1: Create .env file with your API key**

```bash
cp .env.example .env
# Edit .env and add your real OPENAI_API_KEY
```

**Step 2: Start the server**

Run: `uvicorn app.main:app --reload`
Expected: Server starts on http://127.0.0.1:8000

**Step 3: Test with curl**

```bash
# Create a job
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Die Deutsche Bahn plant für März 2026 umfangreiche Bauarbeiten an der Strecke Berlin-München. Projektleiter Thomas Müller rechnet mit Kosten von 50 Millionen Euro.",
    "categories": ["Infrastructure", "Finance", "Operations", "HR"],
    "entities": ["Person", "Organization", "Date", "Location", "Money"]
  }'

# Poll for result (replace JOB_ID)
curl http://localhost:8000/jobs/JOB_ID
```

Expected: Status goes from pending → processing → completed with categories and entities.

**Step 4: Check the OpenAPI docs**

Open: http://127.0.0.1:8000/docs
Expected: Swagger UI with both endpoints documented

**Step 5: Commit**

```bash
git add .env.example
git commit -m "docs: smoke test complete, API working"
```
