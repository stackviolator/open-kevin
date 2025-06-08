"""Kevin-style discounted reward with extra checks for sandbox flags."""
import json
GAMMA = 0.4
BASE_BONUS = 0.3

def _speed_score(runtime_ms: float, best_so_far: float | None) -> float:
    if runtime_ms <= 0:
        return 0.0
    if best_so_far is None:
        return 1.0 / runtime_ms
    return best_so_far / runtime_ms

def compute_score(tool_result: str | None = None, **_):
    if tool_result is None:
        return 0.0
    data = json.loads(tool_result)
    if not data.get("ok") or data.get("timeout") or data.get("rejected"):
        return 0.0
    rt = data["runtime_ms"]
    best = data.get("best_runtime_ms")
    reward = BASE_BONUS + _speed_score(rt, best)
    return reward / (1 - GAMMA)
