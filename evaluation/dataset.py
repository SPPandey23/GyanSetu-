
from langsmith import Client
EVAL_EXAMPLES = [
    {
        "inputs": {"question": "What is DeepSeek-R1 and what is its primary focus?"},
        "outputs": {
            "answer": (
                "DeepSeek-R1 is a first-generation reasoning model introduced by DeepSeek-AI. "
                "Its primary focus is improving reasoning capability through reinforcement learning, "
                "especially on reasoning-intensive tasks such as mathematics, coding, and scientific reasoning."
            )
        },
    },
    {
        "inputs": {"question": "What architecture does DeepSeek-R1 use?"},
        "outputs": {
            "answer": (
                "DeepSeek-R1 uses a Mixture-of-Experts (MoE) architecture with 37B activated "
                "parameters and 671B total parameters."
            )
        },
    },
    {
        "inputs": {"question": "What is the key innovation introduced in DeepSeek-R1-Zero?"},
        "outputs": {
            "answer": (
                "DeepSeek-R1-Zero demonstrates that strong reasoning capabilities can be incentivized "
                "through large-scale reinforcement learning directly on the base model, without using "
                "supervised fine-tuning as a preliminary step."
            )
        },
    },
    {
        "inputs": {"question": "How does DeepSeek-R1-Zero improve its reasoning ability during training?"},
        "outputs": {
            "answer": (
                "DeepSeek-R1-Zero improves through reinforcement learning, where reasoning behaviors "
                "such as self-verification, reflection, longer chain-of-thought, and exploration of alternative "
                "solutions emerge naturally during training."
            )
        },
    },
    {
        "inputs": {"question": "What training approach is used in DeepSeek-R1?"},
        "outputs": {
            "answer": (
                "DeepSeek-R1 uses a multi-stage training pipeline that includes cold-start data, "
                "reasoning-oriented reinforcement learning, rejection sampling with supervised fine-tuning, "
                "and a final reinforcement learning stage for broader alignment."
            )
        },
    },
    {
        "inputs": {"question": "What reward types are used to train DeepSeek-R1-Zero?"},
        "outputs": {
            "answer": (
                "DeepSeek-R1-Zero uses rule-based rewards, mainly accuracy rewards to check whether "
                "answers are correct and format rewards to enforce the required reasoning and answer format."
            )
        },
    },
    {
        "inputs": {"question": "What types of tasks does DeepSeek-R1 perform strongly on?"},
        "outputs": {
            "answer": (
                "DeepSeek-R1 performs strongly on reasoning tasks, mathematics, coding-related tasks, "
                "knowledge benchmarks such as MMLU and GPQA Diamond, and long-context understanding tasks."
            )
        },
    },
]

def get_or_create_dataset(
    dataset_name: str = "DeepSeek RAG Evaluation",
    description: str = "Golden QA pairs for evaluating the DeepSeek RAG pipeline",
) -> str:
    """
    Creates the dataset in LangSmith if it doesn't exist,
    or returns the existing one. Populates examples on first create.

    Returns:
        dataset_name (str) — the name to pass to `client.evaluate()`
    """
    client = Client()

    # Check if dataset already exists
    existing_datasets = list(client.list_datasets(dataset_name=dataset_name))

    if existing_datasets:
        print(f"📦 Dataset '{dataset_name}' already exists — skipping creation.")
        return dataset_name

    # Create fresh
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=EVAL_EXAMPLES,
    )

    print(f"✅ Created dataset '{dataset_name}' with {len(EVAL_EXAMPLES)} examples.")
    return dataset_name
