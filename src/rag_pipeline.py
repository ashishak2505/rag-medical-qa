from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Load .env (for local)
load_dotenv()


# -----------------------------
# 🔐 SAFE API KEY LOADING
# -----------------------------
def get_api_key():
    key = os.getenv("HF_API_KEY")

    if key:
        return key

    try:
        import streamlit as st
        return st.secrets["HF_API_KEY"]
    except Exception:
        return None


HF_API_KEY = get_api_key()


# -----------------------------
# 🔍 INTENT DETECTION
# -----------------------------
def is_summary_question(question: str) -> bool:
    summary_keywords = [
        "summary",
        "summarize",
        "what is this pdf about",
        "what is this document about",
        "overview",
        "brief",
        "gist"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in summary_keywords)


# -----------------------------
# 📦 LOAD EMBEDDINGS (SAFE)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 🤖 LOAD HF CLIENT (LAZY)
# -----------------------------
def get_hf_client():
    if not HF_API_KEY:
        return None

    try:
        return InferenceClient(
            model="HuggingFaceH4/zephyr-7b-beta",
            token=HF_API_KEY
        )
    except Exception as e:
        print(f"Error initializing HF client: {e}")
        return None


# -----------------------------
# 🧠 MAIN QA FUNCTION
# -----------------------------
def answer_question(question, vectorstore):
    client = get_hf_client()

    if client is None:
        return "HF API key not configured", []

    # Detect intent
    summary_mode = is_summary_question(question)

    # Retrieval
    k = 8 if summary_mode else 3
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(question)

    if not docs:
        return "I don't know", []

    # Limit context (VERY IMPORTANT)
    context = "\n\n".join(doc.page_content[:500] for doc in docs)
    context = context[:2000]

    # -----------------------------
    # 📝 PROMPTS
    # -----------------------------
    if summary_mode:
        system_prompt = (
            "You are a medical document summarizer.\n"
            "Generate a concise and structured summary using ONLY the provided context.\n\n"
            "Instructions:\n"
            "1. Highlight key medical findings, diagnoses, and important details.\n"
            "2. Avoid unnecessary repetition.\n"
            "3. Use bullet points if helpful.\n"
            "4. Do NOT add information not present in the context.\n"
        )
    else:
        system_prompt = (
            "You are a highly accurate medical assistant.\n"
            "Use ONLY the provided context to answer the question.\n"
            "Do NOT use external knowledge.\n\n"
            "Rules:\n"
            "1. If the answer is not explicitly present, say: 'I don't know'.\n"
            "2. Do not guess or infer beyond the context.\n"
            "3. Keep answers clear, factual, and concise.\n"
            "4. If relevant, cite key phrases from the context.\n"
        )

    # -----------------------------
    # 💬 MESSAGES
    # -----------------------------
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Context:
{context}

Question:
{question}
"""
        },
    ]

    # -----------------------------
    # 🚀 API CALL
    # -----------------------------
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=350 if summary_mode else 300,
            temperature=0.2,
        )

        return response.choices[0].message.content, docs

    except Exception as e:
        print(f"HF API Error: {e}")
        return "Error generating response", docs