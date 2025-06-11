from __future__ import annotations

"""Command-line entry point that replicates the original *train.py* while
organising it under the `open_kevin` namespace.

Example execution – using the same environment variables as before:

```
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file ~/verifiers/configs/zero3.yaml \
    -m open_kevin.cli.train
```
"""

from dotenv import load_dotenv

load_dotenv()

import datasets
import verifiers as vf

from open_kevin.rewards import (
    compilation_reward,
    correctness_reward,
    performance_reward,
)
from open_kevin.prompts import system_prompt as _system_prompt


MODEL_NAME = "qwen/qwen2.5-1.5b-instruct"


def build_trainer() -> vf.GRPOTrainer:  # type: ignore[name-defined]
    """Factory that constructs and returns a configured `vf.GRPOTrainer`."""

    parser = vf.XMLParser(["think", "code"], answer_field="code")

    # ---------------------------------------------------------------------
    # Reward rubric
    # ---------------------------------------------------------------------
    rubric = vf.Rubric(
        funcs=[
            compilation_reward,
            correctness_reward,
            performance_reward,
            parser.get_format_reward_func(),
        ],
        weights=[0.2, 0.3, 0.5, 0.2],
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

    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    args = vf.grpo_defaults(run_name="le-epic-test")
    args.num_iterations = 10
    args.per_device_train_batch_size = 10
    args.num_generations = 10
    args.gradient_accumulation_steps = 4
    args.eval_strategy = "steps"
    args.eval_steps = 10
    args.max_steps = 100

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
    trainer.train()


if __name__ == "__main__":  # pragma: no cover
    main() 