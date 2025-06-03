"""Discounted, step‑wise reward identical in spirit to Kevin (γ = 0.4)."""

import json
GAMMA = 0.4
BASE_BONUS = 0.3  # matches Kevin paper


def _speed_score(runtime_ms: float, best_so_far: float | None) -> float:
    """Higher is better; >1 if faster than previous best."""
    if runtime_ms <= 0:
        return 0.0
    if best_so_far is None:
        return 1.0 / runtime_ms
    return best_so_far / runtime_ms


def compute_score(prompt, responses, *, turn_id: int, tool_result: str | None = None, **_):
    """Signature: only `prompt, responses` positional → kw‑only for extras."""
    if tool_result is None:
        return 0.0
    data = json.loads(tool_result)
    if not data.get("ok"):
        return 0.0  # failed compile → no reward

    rt = data["runtime_ms"]
    best = data.get("best_runtime_ms")
    reward = BASE_BONUS + _speed_score(rt, best)
    # one‑step discounted total so PPO sees ∑ future
    return reward / (1 - GAMMA)
