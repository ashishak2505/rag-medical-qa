import os

required = [
    "src/__init__.py",
    "src/ingest.py",
    "src/embeddings.py",
    "src/vector_store.py",
    "src/rag_pipeline.py",
]

missing = [f for f in required if not os.path.exists(f)]

if missing:
    raise FileNotFoundError(f"Missing files: {missing}")

print("RAG structure valid ✅")
