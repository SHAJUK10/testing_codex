from fastapi import FastAPI, File, Form, UploadFile

from app.schemas import AskRequest, AskResponse, UploadResponse
from app.services.ingestion_service import ingest_file
from app.services.retrieval_service import retrieve_relevant_chunks

app = FastAPI(title="Syllabus-grounded AI Tutor Backend")


@app.post("/upload", response_model=UploadResponse)
def upload_file(file: UploadFile = File(...), chapter: str | None = Form(default=None)) -> UploadResponse:
    result = ingest_file(file, chapter=chapter)
    return UploadResponse(**result)


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    chunks = retrieve_relevant_chunks(payload.question, n_results=5)
    return AskResponse(
        question=payload.question,
        chunks=chunks,
        llm_answer_placeholder="TODO: integrate with your preferred LLM using retrieved chunks as context.",
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
