#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
DATA_PATH=./

python3 -m verl.trainer.main_ppo \
   data.train_files=$DATA_PATH/data/kernelbench_train.parquet \
   data.val_files=$DATA_PATH/data/kernelbench_holdout.parquet \
   data.prompt_key=prompt \
   data.train_batch_size=16 \
   data.max_prompt_length=1024 \
   data.max_response_length=1024 \
   actor_rollout_ref.model.path=$MODEL_PATH \
   actor_rollout_ref.model.lora_rank=16 \
   actor_rollout_ref.model.lora_alpha=16 \
   actor_rollout_ref.model.target_modules=all-linear \
   actor_rollout_ref.rollout.name=vllm \
   actor_rollout_ref.rollout.load_format=safetensors \
   actor_rollout_ref.rollout.layered_summon=True \
   actor_rollout_ref.rollout.multi_turn.enable=True \
   actor_rollout_ref.rollout.multi_turn.format=chatml \
   actor_rollout_ref.rollout.multi_turn.tool_config_path=tools/kernel_tool.yaml \
   actor_rollout_ref.rollout.temperature=0.7 \
   actor_rollout_ref.rollout.top_p=0.9 \
   actor_rollout_ref.rollout.n=1 \
   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
   actor_rollout_ref.actor.loss_agg_mode=token-mean \
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
   actor_rollout_ref.actor.fsdp_config.param_offload=True \
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
   actor_rollout_ref.actor.use_kl_loss=True \
   actor_rollout_ref.actor.kl_loss_coef=0.001 \
   actor_rollout_ref.actor.kl_loss_type=low_var_kl \
   actor_rollout_ref.actor.entropy_coeff=0 \
   actor_rollout_ref.actor.strategy=fsdp2 \
   algorithm.adv_estimator=grpo \
   algorithm.use_kl_in_reward=False \
   algorithm.kl_penalty=kl \
   algorithm.kl_ctrl.type=fixed \
   algorithm.kl_ctrl.kl_coef=0.02 \
   algorithm.kl_ctrl.target_kl=0.1 \
   custom_reward_function.path=kevin_reward.py \
   custom_reward_function.name=compute_score \
   trainer.project_name=kevin-grpo \
   trainer.experiment_name=kevin-grpo-$MODEL_PATH-lora \
   trainer.total_epochs=4 \
   trainer.nnodes=1 \
   trainer.n_gpus_per_node=1 \
   trainer.save_freq=20 \
   trainer.test_freq=5 \
   trainer.logger=['console','wandb'] $@ 