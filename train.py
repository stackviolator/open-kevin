from dotenv import load_dotenv

load_dotenv()

import verifiers as vf
import datasets

MODEL_NAME="qwen/qwen2.5-0.5b"

parser = vf.XMLParser(['think', 'answer'])

# Reward function
def compute_reward(prompt, completion, answer, **kwargs):
    return 1.0 if completion == answer else 0.0

# Define the rubric
rubric = vf.Rubric(
    funcs=[compute_reward, parser.get_format_reward_func()],
    weights=[1.0, 0.2]
)

# Define dataset
dataset = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")
dataset = dataset.map(lambda x: {"question": x["Problem"], "answer": x["Answer"]})

print(dataset)

# Define the environment
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=f"Fortnite balls I'm gay, I like boys"
)

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=vf.grpo_defaults(run_name="le-epic-test")
)

# trainer.train()