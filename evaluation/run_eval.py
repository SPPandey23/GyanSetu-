import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-only and Agentic RAG evaluation with LangSmith tracing"
    )

    parser.add_argument(
        "--dataset",
        default="DeepSeek RAG Evaluation",
        help="LangSmith dataset name",
    )

    parser.add_argument(
        "--prefix",
        default=None,
        help="Experiment prefix",
    )

    parser.add_argument(
        "--project",
        default="DeepSeek-RAG-Evaluation",
        help="LangSmith project name",
    )

    parser.add_argument(
        "--mode",
        choices=["llm_only", "rag"],
        default="rag",
        help="Evaluation mode: llm_only or rag",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  DeepSeek Evaluation Runner")
    print("=" * 60)

    from evaluation.eval_config import configure_langsmith

    configure_langsmith(project_name=args.project)

    print("\n📦 Setting up evaluation dataset...")
    from evaluation.dataset import get_or_create_dataset

    dataset_name = get_or_create_dataset(dataset_name=args.dataset)

    print("\n📊 Loading evaluators...")
    from evaluation.evaluators import (
        correctness,
        answer_relevance,
        groundedness,
        retrieval_relevance,
    )

    evaluators = [
        correctness,
        answer_relevance,
        groundedness,
        retrieval_relevance,
    ]

    print("\n🎯 Selecting target pipeline...")
    from evaluation.target import (
        llm_only_target,
        agentic_rag_target,
        _ensure_rag_initialized,
    )

    if args.mode == "llm_only":
        target = llm_only_target
        evaluation_type = "llm_only_baseline"
        default_prefix = "deepseek-llm-only"
        print("   Mode: LLM-only baseline")
    else:
        _ensure_rag_initialized()
        target = agentic_rag_target
        evaluation_type = "agentic_rag"
        default_prefix = "deepseek-agentic-rag"
        print("   Mode: Agentic RAG with documents")

    experiment_prefix = args.prefix or (
        f"{default_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    print("\n🚀 Starting evaluation...")
    print(f"   Dataset:    {dataset_name}")
    print(f"   Prefix:     {experiment_prefix}")
    print(f"   Project:    {args.project}")
    print(f"   Mode:       {args.mode}")
    print(
        "   Evaluators: correctness, answer_relevance, "
        "groundedness, retrieval_relevance"
    )

    from langsmith import Client

    client = Client()

    experiment_results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        metadata={
            "app": "GyanSetu",
            "version": "v1.0",
            "evaluation_type": evaluation_type,
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
        },
    )

    print("\n" + "=" * 60)
    print("  ✅ Evaluation Complete!")
    print("=" * 60)

    try:
        df = experiment_results.to_pandas()

        print("\n📊 Results Summary:")
        print(df.to_string(index=False))

        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        results_path = results_dir / f"{experiment_prefix}.csv"
        df.to_csv(results_path, index=False)

        print(f"\n💾 Results saved to: {results_path}")

    except Exception as e:
        print(f"\n⚠️ Could not save local results: {e}")
        print("   Results are still available in LangSmith.")

    print("\n🔗 View results in LangSmith:")
    print("   https://smith.langchain.com/")
    print(f"   Project: {args.project}")
    print(f"   Experiment: {experiment_prefix}")


if __name__ == "__main__":
    main()