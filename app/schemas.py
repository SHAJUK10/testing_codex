from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class RetrievedChunk(BaseModel):
    text: str
    score: float
    chapter: Optional[str] = None
    page: Optional[int] = None


class AskResponse(BaseModel):
    question: str
    chunks: List[RetrievedChunk]
    llm_answer_placeholder: str


class UploadResponse(BaseModel):
    filename: str
    chapter: str
    chunks_indexed: int
