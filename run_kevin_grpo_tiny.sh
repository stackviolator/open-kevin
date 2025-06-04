#!/bin/bash

set -x

python3 -m verl.trainer.main_ppo \
   data.train_files=data/kernelbench_train.parquet \
   data.val_files=data/kernelbench_holdout.parquet \
   data.prompt_key=prompt \
   data.train_batch_size=16 \
   data.max_prompt_length=1024 \
   data.max_response_length=1024 \
   actor_rollout_ref.model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
   actor_rollout_ref.model.lora_rank=16 \
   actor_rollout_ref.model.lora_alpha=16 \
   actor_rollout_ref.model.target_modules=all-linear \
   actor_rollout_ref.model.use_shm=True \
   actor_rollout_ref.rollout.name=vllm \
   actor_rollout_ref.rollout.load_format=safetensors \
   actor_rollout_ref.rollout.layered_summon=True \
   actor_rollout_ref.rollout.multi_turn=False \
   actor_rollout_ref.rollout.format=chatml \
   actor_rollout_ref.rollout.tool_config_path=tools/kernel_tool.yaml \
   actor_rollout_ref.rollout.temperature=0.7 \
   actor_rollout_ref.rollout.top_p=0.9 \
   actor_rollout_ref.rollout.n=1 \
   actor_rollout_ref.ref.model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
   actor_rollout_ref.actor.loss_agg_mode=token-mean \
   actor_rollout_ref.actor.fsdp_config.param_offload=True \
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
   algorithm.adv_estimator=grpo \
   algorithm.use_kl_in_reward=False \
   algorithm.kl_penalty=kl \
   algorithm.kl_ctrl.type=fixed \
   algorithm.kl_ctrl.kl_coef=0.02 \
   algorithm.kl_ctrl.target_kl=0.1 \
   reward_model.module_path=kevin_reward.compute_score \
   trainer.project_name=kevin-grpo \
   trainer.experiment_name=kevin-grpo-qwen0.5b-lora \
   trainer.total_epochs=4 \
   trainer.micro_batch_size_per_gpu=1 \
   trainer.global_batch_size=16 \
   trainer.default_hdfs_dir=hdfs://user/verl/experiments/kevin_grpo_lora/ \
   trainer.logger=['console','wandb'] $@ 