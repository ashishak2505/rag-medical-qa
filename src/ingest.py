from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split(pdf_path):
    """
    Load a PDF file and split it into overlapping text chunks.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        list: List of LangChain Document chunks
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    return chunks


# Optional: simple manual test (NO hard dependency)
if __name__ == "__main__":
    test_pdf = input("Enter PDF path for testing: ").strip()

    if test_pdf:
        chunks = load_and_split(test_pdf)
        print(f"Total chunks created: {len(chunks)}")
        print("\nSample chunk:\n")
        print(chunks[0].page_content[:500])
