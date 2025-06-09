# reward.py
from __future__ import annotations
import re
from typing import Dict, Any
from pathlib import Path

# === kernelbench imports ===
import sys
import os
from pathlib import Path

# Add kernelbench to Python path to handle submodule imports
_current_dir = Path(__file__).parent
_kernelbench_path = _current_dir / "kernelbench"
if _kernelbench_path.exists() and str(_kernelbench_path) not in sys.path:
    sys.path.insert(0, str(_kernelbench_path))

from kernelbench.src.eval import eval_kernel_against_ref, KernelExecResult
from kernelbench.scripts.generate_baseline_time import measure_program_time

# --------------------------------------------
# utility: extract the cuda text from <code>…</code> tags
# --------------------------------------------
_CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.S)

def _extract_cuda(src: str) -> str | None:
    m = _CODE_TAG_RE.search(src)
    return m.group(1).strip() if m else None


# --------------------------------------------
# the single public entry‑point
# --------------------------------------------
def compute_score(                           # ← same signature as before
    reference_code: str,                     # PyTorch baseline (KernelBench “original”)
    response: str,
    *,
    perf_trials: int = 100,
    correct_trials: int = 5,

) -> float:
    """
    score = 0.0 … 1.0
    
    R0  0.0  no <code> tag / empty
    R1  0.1  fails to compile (KernelBench compiled == False)
    R2  0.2  compiled but crashes or wrong answer (correctness == False)
    R3  0.4  correct but slower/equal to eager baseline
    R4  0.4–1.0 correct *and* faster – linearly mapped up to 10× speed‑up
    """
    # ---------------- step 1: formatting ----------------
    cuda_src = _extract_cuda(response)
    if not cuda_src:
        return 0.0

    # ---------------- step 2/3: compilation + correctness ----------------
    kb_result: KernelExecResult = eval_kernel_against_ref(
        original_model_src=reference_code,
        custom_model_src=cuda_src,
        num_correct_trials=correct_trials,
        num_perf_trials=perf_trials,
        measure_performance=True,
    )

    if not kb_result.compiled:
        return 0.1                          # R1

    if not kb_result.correctness:
        return 0.2                          # R2

    kernel_ms: float = kb_result.runtime    # mean of perf_trials

    # ---------------- step 4: baseline timings ----------------
    eager_stats: Dict[str, Any] = measure_program_time(
        ref_arch_name="ref",
        ref_arch_src=reference_code,
        num_trials=perf_trials,
        use_torch_compile=False,
    )
    eager_ms: float = eager_stats["mean"]

    if kernel_ms >= eager_ms:
        return 0.4                          # R3 (correct but not faster)

    # ---------------- step 5: reward shaping by speed‑up ----------------
    speedup = eager_ms / kernel_ms
    # map 1× ↦ 0.4 … 10× ↦ 1.0   (cap anything ≥10×)
    return 0.4 + 0.6 * min((speedup - 1) / 9, 1.0)
