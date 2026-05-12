import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retriever.vectordb import HybridRetrieverBuilder


def load_and_chunk_pdf():
    pdf_path = PROJECT_ROOT / "data" / "deepseek_docs" / "DeepSeek Technical Report.pdf"

    print("📂 Loading PDF from:", pdf_path)
    print("Exists:", pdf_path.exists())

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    print(f"📄 Loaded {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    docs = splitter.split_documents(pages)

    print(f"🧩 Created {len(docs)} chunks")

    return docs


if __name__ == "__main__":
    docs = load_and_chunk_pdf()

    if not docs:
        raise ValueError("No chunks created from PDF")

    builder = HybridRetrieverBuilder()
    builder.build(docs)

    print("✅ DeepSeek PDF indexed into Chroma DB successfully")