# reward.py
from __future__ import annotations
import re
import subprocess
import sys
import tempfile
import pickle
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


def _run_eval_in_subprocess(reference_code: str, cuda_src: str, correct_trials: int, perf_trials: int) -> KernelExecResult | None:
    """Run the evaluation in a subprocess to handle segfaults gracefully."""
    # Create a temporary script to run the evaluation
    script_content = f'''
import sys
from pathlib import Path

# Add kernelbench to Python path
_current_dir = Path(__file__).parent
_kernelbench_path = _current_dir / "kernelbench"
if _kernelbench_path.exists() and str(_kernelbench_path) not in sys.path:
    sys.path.insert(0, str(_kernelbench_path))

from kernelbench.src.eval import eval_kernel_against_ref
import pickle

reference_code = {reference_code!r}
cuda_src = {cuda_src!r}
correct_trials = {correct_trials}
perf_trials = {perf_trials}

try:
    result = eval_kernel_against_ref(
        original_model_src=reference_code,
        custom_model_src=cuda_src,
        num_correct_trials=correct_trials,
        num_perf_trials=perf_trials,
        measure_performance=True,
    )
    # Serialize the result to stdout
    print("RESULT_START")
    print(pickle.dumps(result).hex())
    print("RESULT_END")
except Exception as e:
    print(f"EXCEPTION: {{type(e).__name__}}: {{e}}")
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run the script in a subprocess with a timeout
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
            cwd=str(_current_dir)
        )
        
        if result.returncode != 0:
            # Process crashed (segfault) or had other error
            return None
            
        # Try to extract the pickled result
        output_lines = result.stdout.strip().split('\n')
        if 'RESULT_START' in output_lines and 'RESULT_END' in output_lines:
            start_idx = output_lines.index('RESULT_START')
            end_idx = output_lines.index('RESULT_END')
            if start_idx + 1 < end_idx:
                hex_data = output_lines[start_idx + 1]
                try:
                    result_data = pickle.loads(bytes.fromhex(hex_data))
                    return result_data
                except:
                    pass
        
        return None
        
    except subprocess.TimeoutExpired:
        # Timeout - treat as runtime error
        return None
    except Exception:
        return None
    finally:
        # Clean up the temporary script
        try:
            os.unlink(script_path)
        except:
            pass


# --------------------------------------------
# the single public entry‑point
# --------------------------------------------
def compute_score(                           # ← same signature as before
    reference_code: str,                     # PyTorch baseline (KernelBench "original")
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
    kb_result: KernelExecResult | None = _run_eval_in_subprocess(
        reference_code, cuda_src, correct_trials, perf_trials
    )
    
    if kb_result is None:
        # Subprocess crashed (segfault) or timed out - treat as runtime error
        return 0.2

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
