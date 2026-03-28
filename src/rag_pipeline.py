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

    context = "\n\n".join(doc.page_content[:500] for doc in docs)

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

    messages = [
        {
            "role": "user",
            "content": f"""Context:
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
