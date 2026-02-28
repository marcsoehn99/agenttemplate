# Architecture - Text Classifier & Entity Extractor

## Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant JobStore
    participant Thread
    participant OpenAI

    Client->>FastAPI: POST /jobs {text, categories, entities}
    FastAPI->>JobStore: create(text, categories, entities)
    JobStore-->>FastAPI: job_id
    FastAPI->>Thread: start(process_job)
    FastAPI-->>Client: {job_id, status: "pending"}

    Thread->>JobStore: update_status("processing")
    Thread->>OpenAI: Instructor call (text + dynamic prompt)
    OpenAI-->>Thread: ClassificationResult (Pydantic)
    Thread->>JobStore: complete(job_id, result)

    Client->>FastAPI: GET /jobs/{job_id}
    FastAPI->>JobStore: get(job_id)
    JobStore-->>FastAPI: Job (status + result)
    FastAPI-->>Client: {status: "completed", result: {...}}
```

## Component Overview

```mermaid
flowchart LR
    subgraph app["app/"]
        main["main.py\n─────────\nFastAPI App\nRoutes\nRequest/Response Models"]
        worker["worker.py\n─────────\nJobStore (In-Memory Dict)\nprocess_job()\n_call_llm() via Instructor\nPydantic Result Models"]
    end

    main -->|"store.create()\nstore.get()"| worker
    main -->|"threading.Thread\nprocess_job()"| worker
    worker -->|"instructor.from_openai()"| openai["OpenAI API"]

    client["HTTP Client"] -->|"POST /jobs\nGET /jobs/{id}"| main
```

## Job Lifecycle

```mermaid
stateDiagram-v2
    [*] --> pending: POST /jobs
    pending --> processing: Thread startet
    processing --> completed: LLM Result OK
    processing --> failed: LLM Error
    completed --> [*]: GET /jobs/{id}
    failed --> [*]: GET /jobs/{id}
```

## Projektstruktur

```
agenttemplate/
├── app/
│   ├── main.py        # FastAPI App, Routes, Request/Response Models
│   └── worker.py      # JobStore, LLM Processing, Result Models
├── tests/
│   ├── test_main.py   # API Endpoint Tests
│   └── test_worker.py # JobStore + Processing Tests
├── docs/
│   ├── architecture.md       # ← Du bist hier
│   └── plans/
│       ├── ...-design.md     # Design Doc
│       └── ...-plan.md       # Implementation Plan
├── pyproject.toml
├── .env.example
└── .env                      # (gitignored) OPENAI_API_KEY
```
