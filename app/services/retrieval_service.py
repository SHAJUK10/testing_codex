from typing import Any, Dict, List

from app.services.embedding_service import embed_query
from app.services.vector_store import query_chunks


def retrieve_relevant_chunks(question: str, n_results: int = 5) -> List[Dict[str, Any]]:
    query_vector = embed_query(question)
    query_result = query_chunks(query_embedding=query_vector, n_results=n_results)

    documents = query_result.get("documents", [[]])[0]
    metadatas = query_result.get("metadatas", [[]])[0]
    distances = query_result.get("distances", [[]])[0]

    chunks: List[Dict[str, Any]] = []
    for doc, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": doc,
                "score": float(distance),
                "chapter": metadata.get("chapter") if metadata else None,
                "page": metadata.get("page") if metadata else None,
            }
        )
    return chunks
