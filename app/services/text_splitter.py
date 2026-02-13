from typing import List


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    for start in range(0, len(normalized), step):
        chunk = normalized[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
    return chunks
