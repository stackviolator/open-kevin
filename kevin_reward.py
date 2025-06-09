"""Kevin-style discounted reward with extra checks for sandbox flags."""
from __future__ import annotations
import os, re, subprocess, tempfile, time, multiprocessing as mp
from pathlib import Path
from typing import Dict
import numpy as np

CUDA_RESET_CMD = os.getenv("GPU_RESET_CMD", "nvidia-smi --gpu-reset -i 0")
TIMEOUT_S = 3
GRID_LIMIT = 1024            # reject blocks > 1024 threads in any dim

def _nvcc_compile_run(code: str) -> Dict[str, float | str | bool]:
    """Helper that *must* run inside a child proc; compiles & launches kernel."""
    with tempfile.TemporaryDirectory() as tmp:
        cu_path = Path(tmp) / "kern.cu"
        cu_path.write_text(code)

        # ---------- compile ----------
        start = time.time()
        proc = subprocess.run(
            ["nvcc", "-O3", str(cu_path), "-o", str(Path(tmp) / "kern.out")],
            capture_output=True,
            text=True,
        )
        compile_ms = (time.time() - start) * 1e3
        if proc.returncode != 0:
            return {"ok": False, "log": proc.stderr, "compile_ms": compile_ms}

        # ---------- run ----------
        t0 = time.time()
        run_proc = subprocess.run(
            [str(Path(tmp) / "kern.out")], capture_output=True, text=True,
        )
        runtime_ms = (time.time() - t0) * 1e3
        return {
            "ok": run_proc.returncode == 0,
            "log": proc.stderr + run_proc.stderr,
            "compile_ms": compile_ms,
            "runtime_ms": runtime_ms,
            "stdout": run_proc.stdout,
        }

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

def _safe_launch(code: str) -> Dict[str, float | str | bool]:
    """Fork a subprocess; kill it if it exceeds TIMEOUT_S."""
    if _grid_too_large(code):
        return {"ok": False, "log": "grid too large", "rejected": True}

    q: mp.Queue = mp.Queue()

    def _worker(queue):
        queue.put(_nvcc_compile_run(code))

    p = mp.Process(target=_worker, args=(q,))
    p.start()
    p.join(TIMEOUT_S)
    if p.is_alive():
        p.terminate()
        _maybe_gpu_reset()
        return {"ok": False, "log": "timeout", "timeout": True}
    return q.get()

def _run_pytorch_kernel(code: str) -> Dict[str, float | str | bool]:
    """Runs the original pytorch kernel to get a baseline."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        py_path = f.name

    try:
        start = time.time()
        proc = subprocess.run(
            ["python", py_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        runtime_ms = (time.time() - start) * 1e3

        if proc.returncode != 0:
            return {"ok": False, "log": proc.stderr}

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

    # R0: Bad format
    code = extract_code(response)
    if not code:
        return 0.0

    result = _safe_launch(code)

    # R1: Doesn't compile
    if "runtime_ms" not in result:
        return 0.1

    # R2: Runtime error (e.g. timeout, crash)
    if not result.get("ok"):
        return 0.2

    # CUDA kernel is valid, now run original PyTorch kernel to get baseline
    pytorch_result = _run_pytorch_kernel(original_code)
    if not pytorch_result.get("ok"):
        # If the reference implementation fails, we can't score.
        # Even though the submission compiled, we can't verify it.
        return 0.2

    pytorch_runtime_ms = pytorch_result["runtime_ms"]
    reference_output = pytorch_result["stdout"]

    # R3: Incorrect output
    kernel_output = result.get("stdout", "")
    if not check_correctness(kernel_output, reference_output):
        return 0.3

    runtime_ms = result["runtime_ms"]

    # R4: Correct but not faster than baseline
    if runtime_ms >= pytorch_runtime_ms:
        return 0.4

    # R5: Correct and faster
    speedup = pytorch_runtime_ms / runtime_ms
    # Scale reward between 0.4 and 1.0 for speedups up to 10x
    reward = 0.4 + 0.6 * min((speedup - 1) / 9, 1)
    return reward
