import sys
import time
from pathlib import Path

from langsmith import traceable
from langchain_groq import ChatGroq

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from retriever.vectordb import HybridRetrieverBuilder
from agents.workflow import QAPipeline


_retriever = None
_pipeline = None
_llm = None


def _get_llm():
    global _llm

    if _llm is None:
        _llm = ChatGroq(
            model=settings.MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            max_tokens=1000,
        )

    return _llm


def _ensure_rag_initialized():
    global _retriever, _pipeline

    if _retriever is not None and _pipeline is not None:
        return

    print("🔧 Initializing Agentic RAG pipeline...")

    _retriever = HybridRetrieverBuilder().build()
    _pipeline = QAPipeline()

    print("✅ Agentic RAG pipeline ready.")


def _retrieve_docs(question: str):
    try:
        return _retriever.invoke(question)
    except AttributeError:
        return _retriever.get_relevant_documents(question)


def _debug_documents(documents):
    print("\n" + "=" * 50)
    print("🔍 RETRIEVAL DEBUG")
    print("=" * 50)

    print(f"DOC COUNT: {len(documents)}")

    if not documents:
        print("❌ No documents retrieved.")
        print("=" * 50)
        return

    first_doc = documents[0]

    if hasattr(first_doc, "page_content"):
        preview = first_doc.page_content[:500]
        metadata = getattr(first_doc, "metadata", {})
    elif isinstance(first_doc, dict):
        preview = (
            first_doc.get("page_content")
            or first_doc.get("content")
            or first_doc.get("text")
            or ""
        )[:500]
        metadata = first_doc.get("metadata", {})
    else:
        preview = str(first_doc)[:500]
        metadata = {}

    print("\nFIRST DOC PREVIEW:")
    print(preview)

    print("\nFIRST DOC METADATA:")
    print(metadata)

    print("=" * 50 + "\n")


@traceable(name="gyansetu:llm_only_baseline")
def llm_only_target(inputs: dict) -> dict:
    question = inputs["question"]

    start = time.time()

    prompt = f"""
Answer the following question directly and factually.

Do not use external documents.
Do not mention retrieved context.

Question:
{question}
"""

    response = _get_llm().invoke(prompt)

    latency = time.time() - start

    return {
        "answer": response.content if hasattr(response, "content") else str(response),
        "documents": [],
        "latency": latency,
        "mode": "llm_only",
    }


@traceable(name="gyansetu:agentic_rag_with_docs")
def agentic_rag_target(inputs: dict) -> dict:
    _ensure_rag_initialized()

    question = inputs["question"]

    start = time.time()

    retrieved_docs = _retrieve_docs(question)

    _debug_documents(retrieved_docs)

    result = _pipeline.run(question, _retriever)

    latency = time.time() - start

    answer = result.get("draft_answer") or result.get("answer") or ""

    return {
        "answer": answer,
        "documents": retrieved_docs,
        "relevance_report": result.get("relevance_report", ""),
        "verification_report": result.get("verification_report", ""),
        "latency": latency,
        "mode": "agentic_rag",
    }