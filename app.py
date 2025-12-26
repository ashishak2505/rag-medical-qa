import streamlit as st
import os
from src.vector_store import create_vector_store_from_pdf
from src.rag_pipeline import answer_question
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embedding_model
from src.ingest import load_and_split



st.set_page_config(page_title="Medical RAG QA", layout="wide")

st.title("🧠 Medical RAG Question Answering System")

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# ---- Sidebar: PDF Upload ----
st.sidebar.header("📄 Upload Medical PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing PDF and building vector database..."):
        chunks = load_and_split(pdf_path)
        embeddings = get_embedding_model()
        vectorstore = FAISS.from_documents(chunks, embeddings)

    st.session_state.vectorstore = vectorstore
    st.session_state.db_ready = True


    st.sidebar.success("PDF processed successfully!")
    st.session_state.db_ready = True

# ---- Main Chat Interface ----
st.subheader("💬 Ask Questions")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not st.session_state.db_ready:
        st.warning("Please upload a PDF first.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            answer, docs = answer_question(
                question,
                st.session_state.vectorstore
            )

        # Save to chat history
        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page")
                    }
                    for doc in docs
                ]
            }
        )

# ---- Display Chat History ----
st.divider()

for chat in reversed(st.session_state.chat_history):
    st.markdown("### ❓ Question")
    st.write(chat["question"])

    st.markdown("### ✅ Answer")
    st.write(chat["answer"])

    st.markdown("### 📚 Sources")
    for src in chat["sources"]:
        st.write(f"- Source: {src['source']}, Page: {src['page']}")
    st.divider()

# ---- Disclaimer ----
st.caption(
    "⚠️ This system is for educational and research purposes only. "
    "It does not provide medical diagnosis or treatment advice."
)
