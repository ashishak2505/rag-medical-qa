import os
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embedding_model
from src.ingest import load_and_split

VECTOR_DB_PATH = "vector_db"

def create_vector_store_from_pdf(pdf_path):
    chunks = load_and_split(pdf_path)
    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
