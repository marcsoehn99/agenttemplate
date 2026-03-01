import os
import instructor
from openai import OpenAI
from app.models import JobStore, JobStatus, CategoryInput, EntityInput, ClassificationResult


def _format_items(items: list[CategoryInput] | list[EntityInput]) -> str:
    return "\n".join(
        f"- {item.name}: {item.description}" if item.description else f"- {item.name}"
        for item in items
    )


def _build_system_prompt(categories: list[CategoryInput], entities: list[EntityInput]) -> str:
    parts = ["You are a text classifier and entity extractor."]

    if categories:
        parts.append(
            f"Classify the text into these categories:\n{_format_items(categories)}\n\n"
            "For each matching category, provide a confidence score between 0 and 1."
        )
    else:
        parts.append("No categories requested. Return an empty categories list.")

    if entities:
        parts.append(
            f"Extract these entity types:\n{_format_items(entities)}\n\n"
            "For each entity, provide the exact text, its type, "
            "and the start/end character positions in the original text."
        )
    else:
        parts.append("No entities requested. Return an empty entities list.")

    parts.append(
        "Only return categories and entities that are actually present in the text.\n\n"
        "Provide a brief reasoning explaining your decisions."
    )
    return "\n\n".join(parts)


def _call_llm(text: str, categories: list[CategoryInput], entities: list[EntityInput]) -> ClassificationResult:
    client = instructor.from_openai(OpenAI())
    return client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
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
