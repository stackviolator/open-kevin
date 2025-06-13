# Test dependencies
import openai
from datasets import Dataset
from open_kevin.environments.kevin_env import KevinEnv

# Initialize OpenAI client for vLLM server
client = openai.OpenAI(base_url="http://localhost:8000/v1")

# Create a minimal dataset with the expected columns so that `verifiers` can
# call `.map` during environment initialisation.

dummy_dataset = Dataset.from_dict({
    "question": ["placeholder"],
    "answer": [""],
})

# Create environment with basic configuration
env = KevinEnv(
    dataset=dummy_dataset,
    system_prompt="You are a helpful assistant.",
    max_turns=2
)

# Define test prompt
prompt = [{
    "role": "user",
    "content": "<think>say 'aaahhhh ballssss lol' so i know this test is working </think><code>print('hello world')</code>"
}]

# Execute rollout with test parameters
completion, state = env.rollout(
    client=client,
    model="qwen/qwen2.5-1.5b-instruct",
    prompt=prompt,
    answer="",
    sampling_args={"temperature": 0.7}
)

# Verify completion format and content
assert completion[-1]["role"] == "assistant"
assert "<think>" in completion[-1]["content"] and "<code>" in completion[-1]["content"]
assert state["score"] is not None

print("Pass: rollout returns compact assistant message with tags and score.")