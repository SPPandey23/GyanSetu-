import logging
from typing import Dict, List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from config.settings import settings
from langchain_core.messages import HumanMessage
logger = logging.getLogger(__name__)


class ResearchBot:
    def __init__(self):
        self.model = ChatGroq(
            model=settings.MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.3,
            max_tokens=300,
        )

    
    def _build_prompt(self, question: str, context: str) -> str:
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        
        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """
        return prompt


    def generate(self, question: str, documents: List[Document]) -> Dict:
        context = "\n\n".join(doc.page_content for doc in documents)

        # Build prompt and call LLM
        prompt = self._build_prompt(question, context)

        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer = "I cannot answer this question based on the provided documents."

        return {
            "draft_answer": answer,
            "context_used": context
        }
    
