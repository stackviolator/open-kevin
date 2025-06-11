from dotenv import load_dotenv

load_dotenv()

import verifiers as vf
import datasets
from kevin_reward import compilation_reward, correctness_reward, performance_reward
from system_prompt import system_prompt

"""
Training on 4x A40s. 1 for inference, 3 for training.
CUDA_VISIBLE_DEVICES=0 vf-vllm --model qwen/qwen2.5-1.5b-instruct

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 --config-file ~/verifiers/configs/zero3.yaml train.py
"""
MODEL_NAME="qwen/qwen2.5-1.5b-instruct"

parser = vf.XMLParser(['think', 'code'], answer_field="code")

# Define the rubric
rubric = vf.Rubric(
    funcs=[compilation_reward, correctness_reward, performance_reward, parser.get_format_reward_func()],
    weights=[0.2, 0.3, 0.5, 0.2]
)

# Define dataset
train_dataset = datasets.load_dataset("parquet", data_files="data/kernelbench_train.parquet", split="train")
eval_dataset = datasets.load_dataset("parquet", data_files="data/kernelbench_holdout.parquet", split="train")

system_prompt += f'''\n\nRespond in the following format:
{parser.get_format_str()}'''

# Define the environment
vf_env = vf.SingleTurnEnv(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric
)

args = vf.grpo_defaults(run_name="le-epic-test")
args.num_iterations = 10
args.per_device_train_batch_size = 10
args.num_generations = 10
args.gradient_accumulation_steps = 4
args.eval_strategy = "steps"
args.eval_steps = 10
args.max_steps = 5

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args
)

trainer.train()