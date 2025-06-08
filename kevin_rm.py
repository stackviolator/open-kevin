# tools/kevin_rm.py
from __future__ import annotations
from typing import Dict, Any
import math, torch, json

from verl.workers.reward_manager.base import BaseRewardManager

# ────────────────────────────────────────────────────────────
# your existing reward helper
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
                  **_unused) -> float:
    if tool_result is None:
        return 0.0
    t = json.loads(tool_result)
    if not t.get("ok") or t.get("timeout") or t.get("rejected"):
        return 0.0
    reward = BASE_BONUS + _speed_score(t["runtime_ms"], t.get("best_runtime_ms"))
    return reward / (1 - GAMMA)

# ────────────────────────────────────────────────────────────
class KevinRewardManager(BaseRewardManager):
    """
    Minimal RM that needs no ground‑truth.
    Logs `mean_reward` every validation epoch.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg or {})
        self.reset_metrics()          # <- init accumulators

    # called for *every* training example
    def __call__(self, item, return_dict: bool = False):
        r = compute_score(
            prompt      = item.prompt,
            response    = item.response,
            turn_id     = item.turn_id,
            tool_result = item.non_tensor_batch.get("tool_result"),
        )
        if return_dict:
            return {"reward": r}
        return r

    # ── validation bookkeeping ──
    def reset_metrics(self):
        self._cum, self._cnt = 0.0, 0

    def add_val_result(self, reward: float):
        self._cum += reward
        self._cnt += 1

    def gather_val_metrics(self) -> Dict[str, float]:
        return {"mean_reward": self._cum / self._cnt if self._cnt else math.nan}
