# kevin_reward_manager.py
from __future__ import annotations
from typing import Dict, Any, List

from verl.workers.reward_manager.base import BaseRewardManager

# ---------------------------------------------------------------------
# Helper: your existing reward fn, unchanged
# ---------------------------------------------------------------------
import json, math

GAMMA = 0.4
BASE_BONUS = 0.3

def _speed_score(runtime_ms: float, best_so_far: float | None) -> float:
    if runtime_ms <= 0:
        return 0.0
    if best_so_far is None:
        return 1.0 / runtime_ms
    return best_so_far / runtime_ms

def compute_score(prompt: str,
                  responses: List[str] | str,
                  *,
                  turn_id: int,
                  tool_result: str | None = None,
                  **kwargs) -> float:
    if tool_result is None:
        return 0.0
    data = json.loads(tool_result)
    if not data.get("ok") or data.get("timeout") or data.get("rejected"):
        return 0.0
    rt = data["runtime_ms"]
    best = data.get("best_runtime_ms")
    reward = BASE_BONUS + _speed_score(rt, best)
    return reward / (1 - GAMMA)

# ---------------------------------------------------------------------
# The custom RewardManager
# ---------------------------------------------------------------------
class KevinRewardManager(BaseRewardManager):
    """
    RewardManager that **does not expect** a ground‑truth field.
    It simply calls `compute_score` and reports the mean reward as a metric.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg or {})
        # track cumulative reward and count for val metric
        self._val_cum   = 0.0
        self._val_count = 0

    # ----------  training loop  ----------
    def __call__(self, data_item, return_dict: bool = False):
        reward = compute_score(
            prompt       = data_item.prompt,
            responses    = data_item.response,
            turn_id      = data_item.turn_id,
            tool_result  = data_item.non_tensor_batch.get("tool_result"),
        )
        # you may add extra shaping, clipping, etc. here
        if return_dict:
            return {"reward": reward}
        return reward

    # ----------  validation loop  ----------
    def reset_metrics(self):
        self._val_cum   = 0.0
        self._val_count = 0

    def add_val_result(self, reward: float):
        self._val_cum   += reward
        self._val_count += 1

    def gather_val_metrics(self) -> Dict[str, float]:
        """
        Called by Verl after each validation epoch.
        Return whatever scalars you want logged to WandB / console.
        """
        mean_reward = (self._val_cum / self._val_count) if self._val_count else math.nan
        return {"mean_reward": mean_reward}
