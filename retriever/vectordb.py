import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from config.settings import settings

logger = logging.getLogger(__name__)


class HybridRetrieverBuilder:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5",
                                                model_kwargs={"device": "cpu"},
                                                encode_kwargs={"normalize_embeddings": True})

    def build(self, docs) -> EnsembleRetriever:
        try:
            # Semantic search using vector embeddings
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=settings.CHROMA_DB_PATH
            )
            vector_retriever = vectorstore.as_retriever(
                search_kwargs={"k": 10}
            )

            # Keyword search using BM25
            bm25_retriever = BM25Retriever.from_documents(docs)

            # Combine both retrievers
            hybrid = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS
            )

            logger.info("Hybrid retriever built successfully.")
            return hybrid

        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise