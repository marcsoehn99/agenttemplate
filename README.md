# Text Classifier & Entity Extractor

FastAPI-Service der Text kategorisiert und Entitäten extrahiert. LLM-basiert mit Polling-Architektur.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# .env editieren: OPENAI_API_KEY eintragen
```

## Server starten

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Swagger UI: http://localhost:8000/docs

## API

### Job erstellen

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Die Deutsche Bahn plant für März 2026 Bauarbeiten in Berlin. Thomas Müller leitet das 50 Mio Euro Projekt.",
    "categories": [
      {"name": "Infrastructure", "description": "Bahnhöfe, Strecken, Gleise"},
      {"name": "Finance", "description": "Kosten, Budgets, Investitionen"},
      {"name": "Operations"},
      {"name": "HR"}
    ],
    "entities": [
      {"name": "Person", "description": "Mitarbeiter und Projektleiter"},
      {"name": "Organization"},
      {"name": "Date"},
      {"name": "Location"},
      {"name": "Money"}
    ]
  }'
```

Response:
```json
{"job_id": "uuid", "status": "pending"}
```

### Ergebnis pollen

```bash
curl http://localhost:8000/jobs/<JOB_ID>
```

Response (wenn fertig):
```json
{
  "job_id": "uuid",
  "status": "completed",
  "result": {
    "reasoning": "The text discusses railway construction (Infrastructure) with cost details (Finance)...",
    "categories": [
      {"label": "Infrastructure", "confidence": 0.96}
    ],
    "entities": [
      {"text": "Deutsche Bahn", "type": "Organization", "start": 4, "end": 17}
    ]
  }
}
```

## Kategorien & Entitäten anpassen

Beide Felder sind **optional und unabhängig** voneinander. So kann man nur Kategorisierung, nur Extraktion oder beides machen:

```json
// Nur Kategorisierung
{"text": "...", "categories": [{"name": "Bug"}, {"name": "Feature"}]}

// Nur Entity-Extraktion
{"text": "...", "entities": [{"name": "Person"}, {"name": "Location"}]}

// Beides
{"text": "...", "categories": [...], "entities": [...]}
```

`description` ist optional und gibt dem LLM mehr Kontext:

```json
{"name": "Infrastructure", "description": "Bahnhöfe, Strecken, Gleise, Weichen"}
```

Ohne Description:
```json
{"name": "Infrastructure"}
```

## Tests

```bash
source .venv/bin/activate
pytest -v
```

## Projektstruktur

```
app/
├── main.py      # FastAPI App, Routes, Request/Response Models
└── worker.py    # Job Store, LLM Processing, Pydantic Models
```

## Konfiguration

| Variable | Default | Beschreibung |
|---|---|---|
| `OPENAI_API_KEY` | - | OpenAI API Key (required) |
| `OPENAI_MODEL` | `gpt-5.2` | LLM Model |

## Umbauen auf Azure OpenAI

In `app/worker.py` die `_create_llm_client` Funktion anpassen:

```python
# Vorher (OpenAI)
from openai import OpenAI

def _create_llm_client():
    return instructor.from_openai(OpenAI())

# Nachher (Azure OpenAI)
from openai import AzureOpenAI

def _create_llm_client():
    return instructor.from_openai(AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION", "2024-02-01"),
    ))
```

Und in `.env`:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-01
OPENAI_API_KEY=your-azure-key
```

## Tech Stack

- **FastAPI** + Uvicorn
- **Instructor** + OpenAI SDK (Structured Output via Pydantic)
- **Pydantic v2** (Request/Response Models)
- In-Memory Job Store (thread-safe, erweiterbar auf Redis)
