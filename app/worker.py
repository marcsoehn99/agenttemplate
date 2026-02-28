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


# --- LLM Processing ---

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
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")

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
