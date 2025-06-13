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
import re

import verifiers as vf

# ---------------------------------------------------------------------------
# Local import of kernelbench (kept identical to original kevin_reward.py)
# ---------------------------------------------------------------------------
_current_dir = Path(__file__).resolve()
# The project root is three levels up: <root>/src/open_kevin/rewards/base.py
_project_root = _current_dir.parents[3]

# Add the *project root* to ``sys.path`` so that the interpreter can resolve
# the ``kernelbench`` package (located at <project_root>/kernelbench) in the
# standard "package on the PYTHONPATH" manner.  Adding the sub-directory
# itself (i.e. ``<project_root>/kernelbench``) would cause the import machinery
# to look for ``kernelbench.kernelbench`` which fails.  Therefore, we insert
# the parent directory instead.

_kernelbench_dir = _project_root / "kernelbench"
if _kernelbench_dir.exists() and str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Some helper scripts inside ``kernelbench/scripts`` perform relative imports
# such as ``from src.eval import ...``.  For these to resolve, we additionally
# need the *kernelbench* directory itself on the path so that ``import src``
# finds ``<kernelbench>/src``.  Insert this **after** the project root to avoid
# masking the top-level ``kernelbench`` package import earlier.

if _kernelbench_dir.exists() and str(_kernelbench_dir) not in sys.path:
    sys.path.append(str(_kernelbench_dir))

from kernelbench.src.eval import eval_kernel_against_ref, KernelExecResult  # type: ignore
from kernelbench.scripts.generate_baseline_time import measure_program_time  # type: ignore

# ---------------------------------------------------------------------------
# Simple, in-memory cache to avoid recompilation when the same code pair is
# evaluated repeatedly during RL training.
#
# IMPORTANT:
# 1. `KevinEnv.env_response` may call `compute_score_modular` **inside the rollout**
#    to obtain live feedback (compile success, runtime, etc.) and store these
#    numbers in the `state` dict that is passed back to the model.
# 2. Later, the GRPO training loop asks the rubric to score the very same
#    rollout; it will invoke `compute_score_modular` **again**.  Thanks to this
#    cache, the second call is an O(1) dictionary lookup and no expensive
#    compilation / execution is repeated.
#
# The key includes the reference code, candidate code, and evaluation settings
# so that different prompts/trial counts are cached independently.
# ---------------------------------------------------------------------------
_KB_RESULT_CACHE: Dict[Tuple[str, str, int, int], "KernelExecResult"] = {}

# ---------------------------------------------------------------------------
# Torch.nn usage guard
# ---------------------------------------------------------------------------

_ALLOWED_NN_ATTRS = {
    "Parameter",
    "ParameterList",
    "ModuleList",
    "Sequential",
    "ModuleDict",
    "init",  # sub-module torch.nn.init.*
    "Module",  # base container class
}


_NN_USAGE_PATTERN = re.compile(r"\b(?:torch\.nn|nn)\.(\w+)")


def _uses_disallowed_torch_nn(completion: str) -> bool:
    """Return True if the provided <code> block uses torch.nn in a disallowed way.

    The system prompt forbids any use of ``torch.nn`` except for a small
    whitelist (``Parameter``, container classes, and ``init`` utilities). This
    helper performs a *best-effort* static check by scanning the user-provided
    code for attribute access patterns such as ``torch.nn.Conv2d`` or
    ``nn.functional.relu`` that are **not** on the whitelist. If any such usage
    is detected, the function returns ``True``.
    """

    # The ``completion`` argument may be either a raw string containing the
    # <code> tag or a list of chat messages (as produced by the OpenAI
    # ChatCompletion API). Normalise it into the textual payload that contains
    # the candidate program.
    if isinstance(completion, str):
        completion_text = completion
    else:  # assume list[dict[str, str]] ala OpenAI messages
        try:
            completion_text = completion[-1]["content"]
        except Exception:
            return True  # malformed structure -> reject

    try:
        parser = vf.XMLParser(["think", "code"])
        parsed = parser.parse(completion_text)
        code_src = parsed.code
    except Exception:
        # If structured parsing fails, fall back to inspecting the raw text.
        code_src = completion_text

    for match in _NN_USAGE_PATTERN.finditer(code_src):
        attr = match.group(1)
        if attr not in _ALLOWED_NN_ATTRS:
            return True

    return False


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
    # Normalise *prompt* and *completion* into their textual payloads.
    if isinstance(completion, str):
        completion_text = completion
    else:
        completion_text = completion[-1]["content"]  # type: ignore[index]

    if isinstance(prompt, str):
        prompt_text = prompt
    else:
        prompt_text = prompt[-1]["content"]  # type: ignore[index]

    parser = vf.XMLParser(["think", "code"])
    try:
        parsed = parser.parse(completion_text)
        custom_code = parsed.code
    except Exception as parse_err:
        # Bail out early if the assistant response does not contain the
        # expected XML tags or cannot be parsed. This prevents downstream
        # kernelbench evaluation from crashing on malformed inputs.
        dummy_result = KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={"error": f"Malformed assistant message: {parse_err}"},
        )
        # Cache the dummy result so repeated calls are cheap.
        _KB_RESULT_CACHE[(prompt_text, completion_text, correct_trials, perf_trials)] = dummy_result
        return dummy_result

    # Edge-case: missing or empty <code> block → treat as malformed, bail early.
    if not custom_code or custom_code.strip() == "":
        dummy_result = KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={"error": "Assistant message lacked <code> content."},
        )
        _KB_RESULT_CACHE[(prompt_text, completion_text, correct_trials, perf_trials)] = dummy_result
        return dummy_result

    ref_code = prompt_text

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
    """Continuous reward in [0,1] mapping runtime speed-up to a bounded score."""
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

    # -------------------------------------------------------------------
    # Guard against disallowed ``torch.nn`` usage before any heavy eval.
    # -------------------------------------------------------------------
    if _uses_disallowed_torch_nn(completion):
        return 0.0

    if weights is None:
        weights = _default_weights

    kwargs = {"perf_trials": perf_trials, "correct_trials": correct_trials}

    scores = {
        "compile": compilation_reward(prompt, completion, answer, **kwargs),
        "correct": correctness_reward(prompt, completion, answer, **kwargs),
        "performance": performance_reward(prompt, completion, answer, **kwargs),
    }

    return sum(weights[k] * v for k, v in scores.items())


# Legacy alias expected by `tests/test_reward.py`

def compute_score(prompt: str, completion: str, answer: str = "", **kwargs) -> float:  # noqa: D401
    """Backwards-compatibility shim → delegates to :pyfunc:`compute_score_modular`."""
    return compute_score_modular(prompt, completion, answer, **kwargs) 