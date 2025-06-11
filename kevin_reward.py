# reward_modular.py - Example of split approach
from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
import re

# === kernelbench imports ===
import sys
import os
from pathlib import Path

import verifiers as vf

# Add kernelbench to Python path to handle submodule imports
_current_dir = Path(__file__).parent
_kernelbench_path = _current_dir / "kernelbench"
if _kernelbench_path.exists() and str(_kernelbench_path) not in sys.path:
    sys.path.insert(0, str(_kernelbench_path))

from kernelbench.src.eval import eval_kernel_against_ref, KernelExecResult
from kernelbench.scripts.generate_baseline_time import measure_program_time

# --------------------------------------------
# Utility functions
# --------------------------------------------

def _get_kernel_result(prompt: str, answer: str, completion: str,
                      correct_trials: int = 5, perf_trials: int = 100, **kwargs) -> KernelExecResult:
    parser = vf.XMLParser(['think', 'code'])
    print(f"completion: {completion}")
    parsed = parser.parse(completion[-1]['content'])
    custom_code = parsed.code

    ref_code = prompt[-1]['content']

    # Call kernelbench eval; it may return None if it thinks there was a transient lock-file error.
    kb_result = eval_kernel_against_ref(
        original_model_src=ref_code,
        custom_model_src=custom_code,
        num_correct_trials=correct_trials,
        num_perf_trials=perf_trials,
        measure_performance=True,
    )

    # Robustness: if eval returns None, synthesize a failure result instead of propagating None.
    if kb_result is None:
        # Create a minimal KernelExecResult-like object so downstream code does not crash.
        # We import here to avoid circular import issues.
        from kernelbench.src.eval import KernelExecResult

        kb_result = KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={"error": "eval_kernel_against_ref returned None (likely lock-file or build issue)"},
        )

    return kb_result

# --------------------------------------------
# Individual reward components
# --------------------------------------------

def compilation_reward(prompt: str, completion: str, answer: str, **kwargs) -> float:
    """Reward for successful compilation (0.0 or 1.0)"""
    kb_result = _get_kernel_result(prompt, answer, completion, **kwargs)
    return 1.0 if kb_result.compiled else 0.0

def correctness_reward(prompt: str, completion: str, answer: str, **kwargs) -> float:
    """Reward for correctness (0.0 or 1.0)"""
    kb_result = _get_kernel_result(prompt, answer, completion, **kwargs)
    return 1.0 if kb_result.correctness else 0.0

def performance_reward(prompt: str, completion: str, answer: str, 
                      perf_trials: int = 100, **kwargs) -> float:
    """Reward based on speedup (0.0 to 1.0)"""
    kb_result = _get_kernel_result(prompt, answer, completion, **kwargs)
    if not kb_result.compiled or not kb_result.correctness:
        return 0.0
    
    # Get baseline timing
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

# --------------------------------------------
# Combined reward function (weighted sum)
# --------------------------------------------

def compute_score_modular(
    reference_code: str,
    cuda_src: str,
    *,
    perf_trials: int = 100,
    correct_trials: int = 5,
    weights: Dict[str, float] = None
) -> float:
    """
    Modular reward function with configurable weights
    """
    if weights is None:
        # Default weights
        weights = {
            'compile': 0.2,     # Must compile  
            'correct': 0.3,     # Must be correct
            'performance': 0.5  # Performance matters most
        }
    
    kwargs = {'perf_trials': perf_trials, 'correct_trials': correct_trials}
    
    scores = {
        'compile': compilation_reward(reference_code, cuda_src, **kwargs),
        'correct': correctness_reward(reference_code, cuda_src, **kwargs),
        'performance': performance_reward(reference_code, cuda_src, **kwargs)
    }
    
    # Weighted sum
    total_score = sum(weights[component] * score 
                     for component, score in scores.items())
    
    return total_score

# For backward compatibility
def compute_score(prompt: str, completion: str, answer: str, **kwargs) -> float:
    """
    Take original code from dataset and response from model
    Args:
    reference_code: str
    response: str
    **kwargs: dict
    """
    return compute_score_modular(prompt, answer, **kwargs)