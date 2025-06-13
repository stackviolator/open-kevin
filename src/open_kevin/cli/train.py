from __future__ import annotations

"""Command-line entry point that replicates the original *train.py* while
organising it under the `open_kevin` namespace.

CUDA_VISIBLE_DEVICES=0 vf-vllm --model 'Qwen/Qwen2.5-1.5B-Instruct'
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 --config-file ~/verifiers/configs/zero3.yaml -m open_kevin.cli.train
"""

from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

import datasets
import verifiers as vf

from open_kevin.rewards import compute_score_modular
from open_kevin.prompts import system_prompt as _system_prompt
from open_kevin.environments.kevin_env import KevinEnv

from phoenix.otel import register

tracer_provider = register(
    project_name="open-kevin",
    auto_instrument=True,
    endpoint="http://localhost:6006/v1/traces",

)

MODEL_NAME = "qwen/qwen2.5-1.5b-instruct"


def build_trainer() -> vf.GRPOTrainer:  # type: ignore[name-defined]
    """Factory that constructs and returns a configured `vf.GRPOTrainer`."""

    parser = vf.XMLParser(["think", "code"], answer_field="code")

    # ---------------------------------------------------------------------
    # Reward rubric
    # ---------------------------------------------------------------------
    rubric = vf.Rubric(
        funcs=[
            compute_score_modular,
            parser.get_format_reward_func(),
        ],
        weights=[1.0, 0.2],
    )

    # ---------------------------------------------------------------------
    # Datasets – paths identical to the original script
    # ---------------------------------------------------------------------
    train_dataset = datasets.load_dataset(
        "parquet", data_files="data/kernelbench_train.parquet", split="train"
    )
    eval_dataset = datasets.load_dataset(
        "parquet", data_files="data/kernelbench_holdout.parquet", split="train"
    )

    # Extend the system prompt with the parser format string
    system_prompt = _system_prompt + f"\n\nRespond in the following format:\n{parser.get_format_str()}"

    vf_env = KevinEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        max_turns=8,
        parser=parser,
        rubric=rubric,
    )

    args = vf.grpo_defaults(run_name=f"kevin-multi-turn-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{vf_env.max_turns}")
    args.num_iterations = 20
    args.per_device_train_batch_size = 20
    args.num_generations = 5
    args.gradient_accumulation_steps = 2
    args.eval_strategy = "steps"
    args.eval_steps = 10
    args.num_train_epochs= 1

    model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

    return vf.GRPOTrainer(  # type: ignore[return-value]
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=args,
    )


def main() -> None:  # pragma: no cover
    """Entry-point used by `python -m open_kevin.cli.train` or CLI wrapper."""
    trainer = build_trainer()
    try:
        trainer.train()
    finally:
        if hasattr(trainer, "async_generator"):
            trainer.async_generator.stop()


if __name__ == "__main__":  # pragma: no cover
    main() 