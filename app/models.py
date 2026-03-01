import uuid
from enum import Enum
from threading import Lock
from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CategoryInput(BaseModel):
    name: str
    description: str = ""


class EntityInput(BaseModel):
    name: str
    description: str = ""


class CategoryResult(BaseModel):
    label: str
    confidence: float


class EntityResult(BaseModel):
    text: str
    type: str
    start: int
    end: int


class ClassificationResult(BaseModel):
    reasoning: str
    categories: list[CategoryResult]
    entities: list[EntityResult]


class Job(BaseModel):
    job_id: str
    status: JobStatus
    text: str
    categories: list[CategoryInput] = []
    entities: list[EntityInput] = []
    result: ClassificationResult | None = None
    error: str | None = None


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def create(self, text: str, categories: list[CategoryInput] = [], entities: list[EntityInput] = []) -> str:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, status=JobStatus.PENDING, text=text, categories=categories, entities=entities)
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
