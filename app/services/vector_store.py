from functools import lru_cache
from typing import Any, Dict, List

import chromadb
from chromadb.api.models.Collection import Collection


DB_PATH = "./chroma_db"
COLLECTION_NAME = "syllabus_chunks"


@lru_cache(maxsize=1)
def get_collection() -> Collection:
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def upsert_chunks(
    ids: List[str],
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
) -> None:
    collection = get_collection()
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def query_chunks(query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
    collection = get_collection()
    return collection.query(query_embeddings=[query_embedding], n_results=n_results, include=["documents", "metadatas", "distances"])
