from __future__ import annotations

"""open_kevin.rewards.base

Houses the underlying implementation of reward components that evaluate
CUDA-accelerated kernels produced by a language-model.

Any new, higher-level convenience imports should live in
`open_kevin.rewards.__init__`, *not* here.
"""

from typing import Dict, Tuple
import sys
from pathlib import Path

import verifiers as vf

# ---------------------------------------------------------------------------
# Local import of kernelbench (kept identical to original kevin_reward.py)
# ---------------------------------------------------------------------------
_current_dir = Path(__file__).resolve()
# The project root is three levels up: <root>/src/open_kevin/rewards/base.py
_project_root = _current_dir.parents[3]
_kernelbench_path = _project_root / "kernelbench"
if _kernelbench_path.exists() and str(_kernelbench_path) not in sys.path:
    sys.path.insert(0, str(_kernelbench_path))

from kernelbench.src.eval import eval_kernel_against_ref, KernelExecResult  # type: ignore
from kernelbench.scripts.generate_baseline_time import measure_program_time  # type: ignore

# ---------------------------------------------------------------------------
# Simple, in-memory cache to avoid recompilation when the same code pair is
# evaluated repeatedly during RL training.
# ---------------------------------------------------------------------------
_KB_RESULT_CACHE: Dict[Tuple[str, str, int, int], "KernelExecResult"] = {}


def _get_kernel_result(
    prompt: str,
    answer: str,
    completion: str,
    *,
    correct_trials: int = 5,
    perf_trials: int = 100,
    **kwargs,
) -> KernelExecResult:
    """Internal helper that runs kernelbench evaluation & caches results."""
    parser = vf.XMLParser(["think", "code"])
    parsed = parser.parse(completion[-1]["content"])
    custom_code = parsed.code

    ref_code = prompt[-1]["content"]

    cache_key = (ref_code, custom_code, correct_trials, perf_trials)
    if cache_key in _KB_RESULT_CACHE:
        return _KB_RESULT_CACHE[cache_key]

    kb_result = eval_kernel_against_ref(
        original_model_src=ref_code,
        custom_model_src=custom_code,
        num_correct_trials=correct_trials,
        num_perf_trials=perf_trials,
        measure_performance=True,
    )

    # If kernelbench fails with a lock-file or transient error, synthesise a
    # failure result to keep downstream code robust.
    if kb_result is None:
        kb_result = KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={"error": "eval_kernel_against_ref returned None"},
        )

    _KB_RESULT_CACHE[cache_key] = kb_result
    return kb_result


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

def compilation_reward(prompt: str, completion: str, answer: str = "", **kwargs) -> float:  # noqa: D401
    """0/1 reward indicating whether the generated CUDA code compiled."""
    kb_result = _get_kernel_result(prompt, answer, completion, **kwargs)
    return 1.0 if kb_result.compiled else 0.0


def correctness_reward(prompt: str, completion: str, answer: str = "", **kwargs) -> float:
    """0/1 reward indicating functional correctness of the kernel."""
    kb_result = _get_kernel_result(prompt, answer, completion, **kwargs)
    return 1.0 if kb_result.correctness else 0.0


def performance_reward(
    prompt: str,
    completion: str,
    answer: str = "",
    *,
    perf_trials: int = 100,
    **kwargs,
) -> float:
    """Continuous reward in \[0,1\] mapping runtime speed-up to a bounded score."""
    kb_result = _get_kernel_result(prompt, answer, completion, perf_trials=perf_trials, **kwargs)
    if not kb_result.compiled or not kb_result.correctness:
        return 0.0

    eager_stats = measure_program_time(
        ref_arch_name="ref",
        ref_arch_src=prompt,
        num_trials=perf_trials,
        use_torch_compile=False,
    )

    speedup = eager_stats["mean"] / kb_result.runtime
    if speedup <= 1.0:
        return 0.0

    # Map 1× → 0.0, 10× → 1.0
    return min((speedup - 1) / 9, 1.0)


# ---------------------------------------------------------------------------
# Combined reward helper (kept API-compatible with legacy `compute_score`)
# ---------------------------------------------------------------------------

_default_weights: Dict[str, float] = {
    "compile": 0.2,
    "correct": 0.3,
    "performance": 0.5,
}


def compute_score_modular(
    prompt: str,
    completion: str,
    answer: str,
    *,
    perf_trials: int = 100,
    correct_trials: int = 5,
    weights: Dict[str, float] | None = None,
) -> float:
    """Vectorised reward function combining individual metrics with weights."""

    if weights is None:
        weights = _default_weights

    kwargs = {"perf_trials": perf_trials, "correct_trials": correct_trials}

    scores = {
        "compile": compilation_reward(prompt, completion, answer, **kwargs),
        "correct": correctness_reward(prompt, completion, answer, **kwargs),
        "performance": performance_reward(prompt, completion, answer, perf_trials=perf_trials, **kwargs),
    }

    return sum(weights[k] * v for k, v in scores.items())


# Legacy alias expected by `tests/test_reward.py`

def compute_score(prompt: str, completion: str, answer: str = "", **kwargs) -> float:  # noqa: D401
    """Backwards-compatibility shim → delegates to :pyfunc:`compute_score_modular`."""
    return compute_score_modular(prompt, completion, answer, **kwargs) 