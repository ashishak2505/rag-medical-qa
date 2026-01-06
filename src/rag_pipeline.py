from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import os
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


# Load embeddings ONCE (safe)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load HF client ONCE
from huggingface_hub import InferenceClient

HF_API_KEY = os.getenv("HF_API_KEY")

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=HF_API_KEY
)



def answer_question(question, vectorstore):
    # Detect intent
    summary_mode = is_summary_question(question)

    # Use more chunks for summary
    k = 8 if summary_mode else 3
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    if summary_mode:
        system_prompt = (
            "You are a medical document analyst. "
            "Provide a concise, high-level summary of the document "
            "based ONLY on the provided context."
        )
    else:
        system_prompt = (
            "You are a medical assistant. "
            "Answer ONLY using the provided context. "
            "If the answer is not in the context, say 'I don't know'."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}
"""
        },
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=350 if summary_mode else 300,
        temperature=0.2,
    )

    return response.choices[0].message.content, docs
