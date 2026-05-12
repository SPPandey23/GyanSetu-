import os
from dotenv import load_dotenv


def configure_langsmith(project_name: str = "DeepSeek-RAG-Evaluation") -> None:
    load_dotenv()

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "LANGSMITH_API_KEY is not set. Add it to your .env file or environment variables."
        )

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project_name
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    print(f"✅ LangSmith configured — project: {os.environ['LANGSMITH_PROJECT']}")
    print(f"   Tracing: {os.environ['LANGSMITH_TRACING']}")