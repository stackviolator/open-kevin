from __future__ import annotations
import json, os, re, subprocess, tempfile, time, multiprocessing as mp
from pathlib import Path
from typing import Dict

from verl.tools.base_tool import BaseTool

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
            [str(Path(tmp) / "kern.out")], capture_output=True
        )
        runtime_ms = (time.time() - t0) * 1e3
        return {
            "ok": run_proc.returncode == 0,
            "log": proc.stderr + run_proc.stderr.decode(),
            "compile_ms": compile_ms,
            "runtime_ms": runtime_ms,
        }


class KernelBenchTool(BaseTool):
    """Compile + run LLM‑generated CUDA kernels with sandboxing & watchdog."""

    name = "kernel_bench"

    def __init__(self):
        super().__init__()
        self.best_runtime_ms: float | None = None

    # ------------------------------------------------------------------
    # safety guards -----------------------------------------------------
    # ------------------------------------------------------------------
    _GRID_RE = re.compile(r"<<<\((.*?)\)>>>", re.S)

    @staticmethod
    def _grid_too_large(src: str) -> bool:
        """Crude static check to abort insane launches before nvcc."""
        m = KernelBenchTool._GRID_RE.search(src)
        if not m:
            return False
        dims = [int(d.strip()) for d in m.group(1).split(",") if d.strip()]
        return any(d > GRID_LIMIT for d in dims)

    def _safe_launch(self, code: str) -> Dict[str, float | str | bool]:
        """Fork a subprocess; kill it if it exceeds TIMEOUT_S."""
        if self._grid_too_large(code):
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

    # ------------------------------------------------------------------
    # public tool interface --------------------------------------------
    # ------------------------------------------------------------------
    def __call__(self, query: str, **_) -> str:
        res = self._safe_launch(query)
        # best‑runtime bookkeeping
        if res.get("ok") and res.get("runtime_ms") is not None:
            rt = res["runtime_ms"]
            if self.best_runtime_ms is None or rt < self.best_runtime_ms:
                self.best_runtime_ms = rt
            res["best_runtime_ms"] = self.best_runtime_ms
        return json.dumps(res)


# ----------------------------------------------------------------------
# util -----------------------------------------------------------------
# ----------------------------------------------------------------------

def _maybe_gpu_reset():
    """Try to reset the GPU so the next rollout isn’t wedged."""
    try:
        subprocess.run(CUDA_RESET_CMD.split(), timeout=5)
    except Exception:
        pass  # ignore if reset unavailable
