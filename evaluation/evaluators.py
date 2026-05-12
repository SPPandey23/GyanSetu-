from typing import Any
from typing_extensions import Annotated, TypedDict

from langchain_groq import ChatGroq
from langsmith import traceable

from config.settings import settings


def _get_grader_llm(schema):
    return ChatGroq(
        model=settings.MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
        max_tokens=500,
    ).with_structured_output(schema)


def _docs_to_text(documents: list[Any]) -> str:
    texts = []

    for doc in documents or []:
        if hasattr(doc, "page_content"):
            texts.append(doc.page_content)

        elif isinstance(doc, dict):
            texts.append(
                doc.get("page_content")
                or doc.get("content")
                or doc.get("text")
                or ""
            )

        elif isinstance(doc, str):
            texts.append(doc)

    return "\n\n".join(text for text in texts if text).strip()

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the correctness score"]
    correct: Annotated[bool, ..., "True if the answer is factually correct"]


CORRECTNESS_PROMPT = """
You are evaluating a RAG system answer.

You will be given:
- QUESTION
- REFERENCE ANSWER
- PREDICTED ANSWER

Grade only factual correctness against the reference answer.

Return correct=True if:
- the predicted answer captures the main factual meaning of the reference answer
- it does not contradict the reference answer
- extra information is acceptable only if it is accurate

Return correct=False if:
- the answer is wrong
- the answer misses the main point
- the answer contradicts the reference answer
"""


_correctness_llm = None


def _get_correctness_llm():
    global _correctness_llm

    if _correctness_llm is None:
        _correctness_llm = _get_grader_llm(CorrectnessGrade)

    return _correctness_llm


@traceable(name="eval:correctness")
def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    prompt = f"""
QUESTION:
{inputs.get("question", "")}

REFERENCE ANSWER:
{reference_outputs.get("answer", "")}

PREDICTED ANSWER:
{outputs.get("answer", "")}
"""

    grade = _get_correctness_llm().invoke(
        [
            {"role": "system", "content": CORRECTNESS_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    return bool(grade["correct"])


class AnswerRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the relevance score"]
    relevant: Annotated[bool, ..., "True if the answer addresses the question"]


ANSWER_RELEVANCE_PROMPT = """
You are evaluating whether a RAG system answer addresses the user question.

Return relevant=True if:
- the answer directly addresses the question
- the answer is not off-topic
- the answer is useful to the user

Return relevant=False if:
- the answer is unrelated
- the answer refuses unnecessarily
- the answer does not answer the question
"""


_answer_relevance_llm = None


def _get_answer_relevance_llm():
    global _answer_relevance_llm

    if _answer_relevance_llm is None:
        _answer_relevance_llm = _get_grader_llm(AnswerRelevanceGrade)

    return _answer_relevance_llm


@traceable(name="eval:answer_relevance")
def answer_relevance(inputs: dict, outputs: dict) -> bool:
    prompt = f"""
QUESTION:
{inputs.get("question", "")}

PREDICTED ANSWER:
{outputs.get("answer", "")}
"""

    grade = _get_answer_relevance_llm().invoke(
        [
            {"role": "system", "content": ANSWER_RELEVANCE_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    return bool(grade["relevant"])

class GroundednessGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the groundedness score"]
    grounded: Annotated[bool, ..., "True if the answer is grounded in retrieved documents"]


GROUNDEDNESS_PROMPT = """
You are evaluating whether a RAG answer is grounded in retrieved documents.

You will be given:
- RETRIEVED CONTEXT
- PREDICTED ANSWER

Return grounded=True if:
- the answer is supported by the retrieved context
- the answer does not add unsupported factual claims

Return grounded=False if:
- the answer contains claims not supported by the context
- the answer contradicts the context
- no useful context is provided but the answer makes factual claims
"""


_groundedness_llm = None


def _get_groundedness_llm():
    global _groundedness_llm

    if _groundedness_llm is None:
        _groundedness_llm = _get_grader_llm(GroundednessGrade)

    return _groundedness_llm


@traceable(name="eval:groundedness")
def groundedness(inputs: dict, outputs: dict) -> bool:
    context = _docs_to_text(outputs.get("documents", []))

    prompt = f"""
RETRIEVED CONTEXT:
{context}

PREDICTED ANSWER:
{outputs.get("answer", "")}
"""

    grade = _get_groundedness_llm().invoke(
        [
            {"role": "system", "content": GROUNDEDNESS_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    return bool(grade["grounded"])


class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the retrieval relevance score"]
    relevant: Annotated[bool, ..., "True if retrieved documents are relevant"]


RETRIEVAL_RELEVANCE_PROMPT = """
You are evaluating retrieved context for a RAG system.

Return relevant=True if:
- the retrieved context contains information related to the question
- the context would help answer the question

Return relevant=False if:
- the retrieved context is empty
- the context is unrelated to the question
"""


_retrieval_relevance_llm = None


def _get_retrieval_relevance_llm():
    global _retrieval_relevance_llm

    if _retrieval_relevance_llm is None:
        _retrieval_relevance_llm = _get_grader_llm(RetrievalRelevanceGrade)

    return _retrieval_relevance_llm


@traceable(name="eval:retrieval_relevance")
def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    context = _docs_to_text(outputs.get("documents", []))

    prompt = f"""
QUESTION:
{inputs.get("question", "")}

RETRIEVED CONTEXT:
{context}
"""

    grade = _get_retrieval_relevance_llm().invoke(
        [
            {"role": "system", "content": RETRIEVAL_RELEVANCE_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    return bool(grade["relevant"])





