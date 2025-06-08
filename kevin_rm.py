# tools/kevin_rm.py
from __future__ import annotations
import json, torch
from typing import Any
from verl.workers.reward_manager import NaiveRewardManager

GAMMA, BASE_BONUS = 0.4, 0.3

def _speed_score(rt_ms: float, best: float | None) -> float:
    if rt_ms <= 0:
        return 0.0
    return (1.0 if best is None else best) / rt_ms

def compute_score(prompt: str,
                  response: str | list[str],
                  *,
                  turn_id: int,
                  tool_result: str | None = None,
                  **_) -> float:
    if tool_result is None:
        return 0.0
    t = json.loads(tool_result)
    if not t.get("ok") or t.get("timeout") or t.get("rejected"):
        return 0.0
    reward = BASE_BONUS + _speed_score(t["runtime_ms"], t.get("best_runtime_ms"))
    return reward / (1 - GAMMA)

class KevinRewardManager(NaiveRewardManager):
    """Rule‑based RM that needs no ground‑truth column."""
    def __init__(self, tokenizer, num_examine: int = 0, compute_score=compute_score):
        super().__init__(tokenizer=tokenizer,
                         num_examine=num_examine,
                         compute_score=compute_score)

    def __call__(self, data_proto, return_dict: bool = False):
        rewards = []
        for item in data_proto:
            print(data_proto.keys())
            print(item)
            r = compute_score(
                tool_result=item.non_tensor_batch.get("tool_result"),
            )
            rewards.append(r)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        if return_dict:
            return reward_tensor, {"mean_reward": reward_tensor.mean().item()}
        return reward_tensor
