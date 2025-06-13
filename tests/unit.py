# Test dependencies
import openai
from datasets import Dataset, load_dataset
from open_kevin.environments.kevin_env import KevinEnv
from open_kevin.prompts import system_prompt as _system_prompt
from verifiers import XMLParser
import random
from phoenix.otel import register

tracer_provider = register(
    project_name="open-kevin",
    auto_instrument=True,
    endpoint="http://localhost:6006/v1/traces",

)

# Initialize OpenAI client for vLLM server
client = openai.OpenAI(base_url="http://localhost:8000/v1")

# Create a minimal dataset with the expected columns so that `verifiers` can
# call `.map` during environment initialisation.

try:
    kb_dataset = load_dataset(
        "parquet",
        data_files="data/kernelbench_train.parquet",
        split="train",
    )
except Exception:
    # Fallback to a minimal dummy dataset if the parquet file is unavailable
    kb_dataset = Dataset.from_dict({
        "question": ["say 'test fallback' so I know fallback works"],
        "answer": [""],
    })

# Select a random example from the dataset for the test prompt
random_row = kb_dataset.shuffle(seed=random.randint(0, 1_000_000))[0]

# Build system prompt identical to the training script
parser = XMLParser(["think", "code"], answer_field="code")
system_prompt = _system_prompt + f"\n\nRespond in the following format:\n{parser.get_format_str()}"

# Create environment with basic configuration
env = KevinEnv(
    dataset=kb_dataset,
    system_prompt=system_prompt,
    max_turns=4
)

# Define test prompt using the sampled example
prompt = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user", 
        "content": random_row["question"]
    }
]

# Execute rollout with test parameters
completion, state = env.rollout(
    client=client,
    model="qwen/qwen2.5-1.5b-instruct",
    prompt=prompt,
    answer=random_row.get("answer", ""),
    sampling_args={"temperature": 0.7}
)

# Verify completion format and content
assert completion[-1]["role"] == "assistant"
assert "<think>" in completion[-1]["content"] and "<code>" in completion[-1]["content"]
assert state["score"] is not None

print("Pass: rollout returns compact assistant message with tags and score.")