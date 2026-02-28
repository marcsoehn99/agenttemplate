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
