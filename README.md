# Syllabus-grounded AI Tutor Backend

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API

- `POST /upload`
  - multipart form-data:
    - `file`: `.pdf` or `.txt`
    - `chapter` (optional): chapter name override
  - Extracts text, chunks (1000 chars, 200 overlap), embeds with `sentence-transformers/all-MiniLM-L6-v2`, and stores in Chroma with metadata (`chapter`, `page`, `chunk_index`, `source`).

- `POST /ask`
  - body:
    ```json
    {"question": "What is Newton's second law?"}
    ```
  - Retrieves top 5 relevant chunks from Chroma and returns text + similarity score with an LLM integration placeholder.
