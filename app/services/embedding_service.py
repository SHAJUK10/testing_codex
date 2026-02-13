from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    model = get_embedding_model()
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vector.tolist()
