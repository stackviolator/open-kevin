"""Kevin-style discounted reward with extra checks for sandbox flags."""
from __future__ import annotations
import os, re, subprocess, tempfile, time, multiprocessing as mp
from pathlib import Path
from typing import Dict
import numpy as np

CUDA_RESET_CMD = os.getenv("GPU_RESET_CMD", "nvidia-smi --gpu-reset -i 0")
TIMEOUT_S = 3
GRID_LIMIT = 1024            # reject blocks > 1024 threads in any dim

def _compile_kernel(code: str, tmp_dir_path: str) -> Dict:
    """Compiles the given CUDA code using nvcc."""
    cu_path = Path(tmp_dir_path) / "kern.cu"
    cu_path.write_text(code)
    out_path = Path(tmp_dir_path) / "kern.out"

    proc = subprocess.run(
        ["nvcc", "-O3", str(cu_path), "-o", str(out_path)],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        return {"ok": False, "log": proc.stderr}

    return {"ok": True, "executable_path": str(out_path), "log": proc.stderr}

def _run_compiled_kernel(executable_path: str) -> Dict:
    """Runs a compiled CUDA executable and captures its output and runtime."""
    t0 = time.time()
    run_proc = subprocess.run(
        [executable_path], capture_output=True, text=True,
    )
    runtime_ms = (time.time() - t0) * 1e3

    if run_proc.returncode != 0:
        return {"ok": False, "log": run_proc.stderr, "stdout": run_proc.stdout}

    return {
        "ok": True,
        "log": run_proc.stderr,
        "stdout": run_proc.stdout,
        "runtime_ms": runtime_ms,
    }

def _safe_exec(target_func, *args) -> Dict:
    """Executes a target function in a sandboxed process with a timeout."""
    q: mp.Queue = mp.Queue()
    def _worker(queue, *args):
        queue.put(target_func(*args))

    p = mp.Process(target=_worker, args=(q,) + args)
    p.start()
    p.join(TIMEOUT_S)
    if p.is_alive():
        p.terminate()
        _maybe_gpu_reset()
        return {"ok": False, "log": "timeout", "timeout": True}
    return q.get()

_GRID_RE = re.compile(r"<<<\((.*?)\)>>>", re.S)

def _grid_too_large(src: str) -> bool:
    """Crude static check to abort insane launches before nvcc."""
    m = _GRID_RE.search(src)
    if not m:
        return False
    dims = [int(d.strip() or 1) for d in m.group(1).split(",")]
    return any(d > GRID_LIMIT for d in dims if d)

def _maybe_gpu_reset():
    """Try to reset the GPU so the next rollout isn't wedged."""
    try:
        subprocess.run(CUDA_RESET_CMD.split(), timeout=5)
    except Exception:
        pass  # ignore if reset unavailable

def _instrument_pytorch_code(code: str) -> str | None:
    """Instruments PyTorch code with a timing decorator for precise measurement."""
    # Find the function definition to apply the decorator to.
    m_def = re.search(r"def\s+(\w+)\s*\(.*\):", code, re.DOTALL)
    if not m_def:
        return None  # No function definition found to instrument.

    def_statement = m_def.group(0)
    # Apply the decorator to the function definition.
    code_with_decorator = code.replace(def_statement, f"@time_this\n{def_statement}")

    # The decorator and its helper variables.
    wrapper_code = f"""
import time
import sys

__start_time = None
__end_time = None

def time_this(func):
    def wrapper(*args, **kwargs):
        global __start_time, __end_time
        # For GPU timing, we need to synchronize before starting the timer.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, AttributeError):
            pass  # Torch not available or no CUDA.
        __start_time = time.perf_counter()
        result = func(*args, **kwargs)
        # Synchronize again after execution before stopping the timer.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, AttributeError):
            pass
        __end_time = time.perf_counter()
        return result
    return wrapper
"""

    # Combine the wrapper, the decorated code, and the final print statement.
    final_code = wrapper_code + "\\n" + code_with_decorator + """
if __start_time is not None and __end_time is not None:
    runtime_ms = (__end_time - __start_time) * 1000
    print(str(runtime_ms), file=sys.stderr, flush=True)
"""
    return final_code

def _run_pytorch_kernel(code: str) -> Dict[str, float | str | bool]:
    """
    Runs the original PyTorch kernel to get a baseline. It first tries to
    instrument the code for precise timing and falls back to timing the
    entire process if instrumentation fails.
    """
    instrumented_code = _instrument_pytorch_code(code)

    if instrumented_code is None:
        # Fallback to old method if instrumentation is not possible.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            py_path = f.name
        try:
            start = time.time()
            proc = subprocess.run(
                ["python", py_path],
                capture_output=True, text=True, timeout=TIMEOUT_S
            )
            runtime_ms = (time.time() - start) * 1e3
            if proc.returncode != 0:
                return {"ok": False, "log": proc.stderr}
            return {"ok": True, "stdout": proc.stdout, "runtime_ms": runtime_ms}
        except subprocess.TimeoutExpired:
            return {"ok": False, "log": "timeout", "timeout": True}
        finally:
            os.unlink(py_path)

    # Run the instrumented code.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(instrumented_code)
        py_path = f.name

    try:
        proc = subprocess.run(
            ["python", py_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )

        if proc.returncode != 0:
            return {"ok": False, "log": proc.stderr}

        # The runtime is now in stderr, and the kernel output in stdout.
        try:
            runtime_ms = float(proc.stderr.strip())
        except (ValueError, IndexError):
            return {"ok": False, "log": "Failed to parse runtime from instrumented code: " + proc.stderr}

        return {
            "ok": True,
            "stdout": proc.stdout,
            "runtime_ms": runtime_ms,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "log": "timeout", "timeout": True}
    finally:
        os.unlink(py_path)

def extract_code(generation: str) -> str | None:
    match = re.search(r"<code>(.*?)</code>", generation, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def check_correctness(kernel_output: str, reference_output: str) -> bool:
    """Compare kernel output with reference output.

    Assumes outputs are space-separated lists of numbers.
    """
    print(f"Kernel output: {kernel_output}")
    print(f"Reference output: {reference_output}")
    try:
        kernel_vals = [float(x) for x in kernel_output.strip().split()]
        ref_vals = [float(x) for x in reference_output.strip().split()]
        if len(kernel_vals) != len(ref_vals):
            return False
        return np.allclose(kernel_vals, ref_vals, atol=1e-5, rtol=1e-5)
    except (ValueError, IndexError):
        return False

def compute_score(original_code: str, response: str, **_) -> float:
    """Computes a reward score for a generated CUDA kernel."""
    # R0: Format checks
    code = extract_code(response)
    if not code:
        return 0.0 # Bad format
    
    # R1: Code Compiles
    if _grid_too_large(code):
        return 0.1 
    with tempfile.TemporaryDirectory() as tmp:
        compile_result = _safe_exec(_compile_kernel, code, tmp)
        if not compile_result.get("ok"):
            return 0.1

        # R2: No runtime errors
        executable_path = compile_result["executable_path"]
        run_result = _safe_exec(_run_compiled_kernel, executable_path)
        if not run_result.get("ok"):
            print(f"CUDA kernel failed: {run_result}")
            return 0.2
        pytorch_result = _run_pytorch_kernel(original_code)
        if not pytorch_result.get("ok"):
            print(f"PyTorch kernel failed: {pytorch_result}")
            return 0.2

        # R3: Correct output
        pytorch_runtime_ms = pytorch_result["runtime_ms"]
        reference_output = pytorch_result["stdout"]
        kernel_output = run_result.get("stdout", "")
        if not check_correctness(kernel_output, reference_output):
            return 0.3

        # R4: Correct but not faster than baseline
        runtime_ms = run_result["runtime_ms"]
        if runtime_ms >= pytorch_runtime_ms:
            return 0.4

        # R5: Correct and faster
        speedup = pytorch_runtime_ms / runtime_ms
        # Scale reward between 0.4 and 1.0 for speedups up to 10x
        return 0.4 + 0.6 * min((speedup - 1) / 9, 1)
