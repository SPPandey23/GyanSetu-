import logging
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

from .research_agent import ResearchBot
from .verification_agent import AnswerVerifier
from .relevance_checker import DocumentRelevanceFilter

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: EnsembleRetriever


class QAPipeline:
    def __init__(self):
        self.researcher = ResearchBot()
        self.verifier = AnswerVerifier()
        self.relevance_checker = DocumentRelevanceFilter()
        self.workflow = self._build_workflow()

    def run(self, question: str, retriever: EnsembleRetriever) -> Dict:
        try:
            documents = retriever.invoke(question)
            logger.info(f"Retrieved {len(documents)} documents for: '{question}'")

            initial_state = AgentState(
                question=question,
                documents=documents,
                draft_answer="",
                verification_report="",
                is_relevant=False,
                retriever=retriever
            )

            final_state = self.workflow.invoke(initial_state)

            return {
                "draft_answer": final_state["draft_answer"],
                "verification_report": final_state["verification_report"]
            }
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _build_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("check_relevance", self._check_relevance)
        workflow.add_node("research", self._research)
        workflow.add_node("verify", self._verify)

        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            lambda state: "research" if state["is_relevant"] else END
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            lambda state: "research" if self._needs_retry(state) else END
        )

        return workflow.compile()

    def _check_relevance(self, state: AgentState) -> Dict:
        result = self.relevance_checker.check(
            question=state["question"],
            retriever=state["retriever"],
            k=10  # more context = better judgement
        )

        if result in ("CAN_ANSWER", "PARTIAL"):
            return {"is_relevant": True}

        return {
            "is_relevant": False,
            "draft_answer": "Your question doesn't seem related to the uploaded documents. Please ask something relevant."
        }

    def _research(self, state: AgentState) -> Dict:
        result = self.researcher.generate(state["question"], state["documents"])
        return {"draft_answer": result["draft_answer"]}

    def _verify(self, state: AgentState) -> Dict:
        result = self.verifier.check(state["draft_answer"], state["documents"])
        return {"verification_report": result["verification_report"]}

    def _needs_retry(self, state: AgentState) -> bool:
    
        report = state["verification_report"]
        return "Supported: NO" in report or "Relevant: NO" in report