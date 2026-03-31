import logging
from typing import Dict, List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage  
from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_REPORT = {
    "Supported": "YES",        
    "Unsupported Claims": [],
    "Contradictions": [],
    "Relevant": "YES",         
    "Additional Details": ""
}


class AnswerVerifier:

    def __init__(self):
        self.model = ChatGroq(
            model=settings.MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=200,
        )

    def check(self, answer: str, documents: List[Document]) -> Dict:
        context = "\n\n".join(doc.page_content for doc in documents)
        prompt = self._build_prompt(answer, context)

        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            
            return {
                "verification_report": self._format_report(DEFAULT_REPORT),
                "context_used": context
            }

        report = self._parse_response(raw) or DEFAULT_REPORT
        return {
            "verification_report": self._format_report(report),
            "context_used": context
        }

    def _build_prompt(self, answer: str, context: str) -> str:
        prompt = f"""
        You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

        Instructions:
        - Verify the following answer against the provided context.
        - Check for:
        1. Direct/indirect factual support (YES/NO)
        2. Unsupported claims (list any if present)
        3. Contradictions (list any if present)
        4. Relevance to the question (YES/NO)
        - Provide additional details or explanations where relevant.
        - Respond in the exact format specified below without adding any unrelated information.

        Format:
        Supported: YES/NO
        Unsupported Claims: [item1, item2, ...]
        Contradictions: [item1, item2, ...]
        Relevant: YES/NO
        Additional Details: [Any extra information or explanations]

        Answer: {answer}
        Context: {context}

        Respond ONLY with the above format."
        """
        return prompt
    def _parse_response(self, text: str) -> Dict:
        """Parse the LLM's structured response into a dictionary."""
        try:
            report = {}
            for line in text.split('\n'):
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key in ("Unsupported Claims", "Contradictions"):
                    if value.lower() == "none":
                        report[key] = []
                    else:
                        items = value.strip('[]').split(',')
                        report[key] = [i.strip().strip('"\'') for i in items if i.strip()]
                elif key in ("Supported", "Relevant"):
                    report[key] = value.upper()
                elif key == "Additional Details":
                    report[key] = value

            # Fill missing keys with defaults
            for key, default in DEFAULT_REPORT.items():
                report.setdefault(key, default)

            return report
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return None

    def _format_report(self, report: Dict) -> str:
        
        lines = [
            f"Supported: {report.get('Supported', 'YES')}",
            f"Unsupported Claims: {', '.join(report.get('Unsupported Claims', [])) or 'None'}",
            f"Contradictions: {', '.join(report.get('Contradictions', [])) or 'None'}",
            f"Relevant: {report.get('Relevant', 'YES')}",
            f"Additional Details: {report.get('Additional Details', '') or 'None'}",
        ]
        return "\n".join(lines)