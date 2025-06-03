import json
from kevin_reward import compute_score

GAMMA = 0.4


def test_reward_positive():
    tool_res = {"ok": True, "runtime_ms": 10.0, "best_runtime_ms": 20.0}
    expected = (0.3 + 2.0) / (1 - GAMMA)  # 20/10 = 2
    out = compute_score("p", ["r"], turn_id=0, tool_result=json.dumps(tool_res))
    assert abs(out - expected) < 1e-6


def test_reward_zero_on_failure():
    for res in [
        {"ok": False},
        {"ok": True, "timeout": True},
        {"ok": True, "rejected": True},
    ]:
        assert compute_score("p", ["r"], turn_id=0, tool_result=json.dumps(res)) == 0.0
