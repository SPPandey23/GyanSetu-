import logging
from langchain_groq import ChatGroq
from config.settings import settings
from langchain_core.messages import HumanMessage
logger = logging.getLogger(__name__)


class DocumentRelevanceFilter:
    VALID_LABELS = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}

    def __init__(self):
        self.model = ChatGroq(
            model=settings.MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.2,
        )

    def check(self, question: str, retriever, k=10) -> str:
        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents found. Returning NO_MATCH.")
            return "NO_MATCH"

        context = "\n\n".join(doc.page_content for doc in top_docs[:k])
        label = self._classify(question, context)
        print(f"Relevance result: {label}")
        return label

    def _classify(self, question: str, context: str) -> str:
        prompt = f"""You are an intelligent document relevance evaluator.

Your task is to assess whether the provided document passages contain enough information to address the user's question.

Guidelines:
- Be generous with your classification. If there is any meaningful overlap between the question and the passages, prefer CAN_ANSWER or PARTIAL.
- For general questions such as summaries, overviews, or topic listings, return CAN_ANSWER if the passages contain relevant content.
- Only return NO_MATCH if the passages are entirely unrelated to the question.

Classification Labels:
- CAN_ANSWER: The passages contain sufficient information to fully or mostly answer the question.
- PARTIAL: The passages touch on the topic but lack enough detail for a complete answer.
- NO_MATCH: The passages have no meaningful connection to the question.

User Question: {question}

Document Passages:
{context}

Respond with exactly one label — no explanation, no extra text:"""

        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            label = response.content.strip().upper()
            logger.debug(f"LLM label: {label}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "CAN_ANSWER"  # fail open instead of fail closed

        return label if label in self.VALID_LABELS else "CAN_ANSWER"