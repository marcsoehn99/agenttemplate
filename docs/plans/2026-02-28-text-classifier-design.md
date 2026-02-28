# Text Classifier & Entity Extractor - Design Doc

## Ziel

FastAPI-Service der Text kategorisiert und Entitäten extrahiert.
LLM-basiert (Instructor + OpenAI), mit Polling-Architektur für asynchrone Verarbeitung.

## Entscheidungen

- **Framework:** FastAPI
- **LLM:** OpenAI (POC), später Azure OpenAI (Deutsche Bahn)
- **Structured Output:** Instructor + Pydantic v2
- **Job-Queue:** In-Memory Dict (später Redis-Migration möglich)
- **Auth:** Keine (POC)
- **Chunking:** Keins - GPT-4o hat 128k Token Context Window

## API

### POST /jobs

Erstellt einen neuen Klassifikations-Job.

Request:
```json
{
  "text": "Die Deutsche Bahn plant für März 2026...",
  "categories": ["Infrastructure", "Finance", "Operations"],
  "entities": ["Person", "Organization", "Date", "Location"]
}
```

Response:
```json
{
  "job_id": "uuid",
  "status": "pending"
}
```

### GET /jobs/{job_id}

Gibt Status und (wenn fertig) das Ergebnis zurück.

Response (pending/processing):
```json
{
  "job_id": "uuid",
  "status": "processing"
}
```

Response (completed):
```json
{
  "job_id": "uuid",
  "status": "completed",
  "result": {
    "categories": [
      {"label": "Infrastructure", "confidence": 0.92}
    ],
    "entities": [
      {"text": "Deutsche Bahn", "type": "Organization", "start": 4, "end": 17},
      {"text": "März 2026", "type": "Date", "start": 28, "end": 37}
    ]
  }
}
```

Response (failed):
```json
{
  "job_id": "uuid",
  "status": "failed",
  "error": "OpenAI API timeout"
}
```

## Flow

```
POST /jobs → UUID generieren → In-Memory Dict speichern → Background Thread startet
GET  /jobs/{id} → Dict nachschauen → Status + Result zurückgeben
```

## Projektstruktur

```
agenttemplate/
├── app/
│   ├── main.py        # FastAPI App, Routes, Pydantic Models
│   └── worker.py      # Job Store + LLM Processing (Instructor)
├── pyproject.toml
└── .env.example
```

## Tech Stack

| Komponente | Technologie |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| LLM Client | Instructor + OpenAI SDK |
| Models | Pydantic v2 |
| Job Store | In-Memory Dict (threading-safe) |
| Config | .env via pydantic-settings |

## Qualitätsanspruch

- Code lesbar wie ein Buch
- Klare Variablennamen, logische Struktur
- Stabil aber nicht over-engineered

## Späterer Ausbau (nicht im POC)

- Azure OpenAI statt OpenAI
- Redis statt In-Memory
- API-Key Auth
- Chunking für Texte > 128k Tokens
