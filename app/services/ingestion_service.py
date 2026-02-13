import io
import os
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader

from app.services.embedding_service import embed_texts
from app.services.text_splitter import split_text
from app.services.vector_store import upsert_chunks


def _extract_text_from_pdf(content: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(content))
    pages: List[Tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        pages.append((idx, page.extract_text() or ""))
    return pages


def _extract_text_from_txt(content: bytes) -> List[Tuple[int, str]]:
    text = content.decode("utf-8", errors="ignore")
    return [(1, text)]


def ingest_file(upload_file: UploadFile, chapter: str | None = None) -> Dict[str, str | int]:
    ext = Path(upload_file.filename or "").suffix.lower()
    if ext not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    content = upload_file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    chapter_value = chapter or os.path.splitext(upload_file.filename or "uploaded_file")[0]

    if ext == ".pdf":
        page_texts = _extract_text_from_pdf(content)
    else:
        page_texts = _extract_text_from_txt(content)

    chunk_docs: List[str] = []
    chunk_ids: List[str] = []
    chunk_metadatas: List[Dict[str, str | int]] = []

    for page, text in page_texts:
        chunks = split_text(text, chunk_size=1000, overlap=200)
        for chunk_index, chunk in enumerate(chunks):
            chunk_docs.append(chunk)
            chunk_ids.append(str(uuid.uuid4()))
            chunk_metadatas.append(
                {
                    "chapter": chapter_value,
                    "page": page,
                    "chunk_index": chunk_index,
                    "source": upload_file.filename or "unknown",
                }
            )

    if not chunk_docs:
        raise HTTPException(status_code=400, detail="No readable text found in file")

    embeddings = embed_texts(chunk_docs)
    upsert_chunks(ids=chunk_ids, documents=chunk_docs, embeddings=embeddings, metadatas=chunk_metadatas)

    return {
        "filename": upload_file.filename or "unknown",
        "chapter": chapter_value,
        "chunks_indexed": len(chunk_docs),
    }
