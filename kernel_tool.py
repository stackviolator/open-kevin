import json, subprocess, tempfile, time
from pathlib import Path
from typing import Dict

from verl.tools.base_tool import BaseTool

class KernelBenchTool(BaseTool):
    """Compile + run a CUDA kernel and return timing / compile log.

    The instance is a long‑lived singleton inside each rollout worker, so we can
    keep best‑so‑far numbers as internal state.
    """

    name = "kernel_bench"

    def __init__(self):
        super().__init__()
        self.best_runtime_ms: float | None = None

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _compile_run(self, code: str) -> Dict[str, float | str | bool]:
        """Return dict with compile status and timing in ms."""
        with tempfile.TemporaryDirectory() as tmp:
            cu_path = Path(tmp) / "kern.cu"
            cu_path.write_text(code)
            start = time.time()
            proc = subprocess.run(
                ["nvcc", "-O3", str(cu_path), "-o", str(Path(tmp) / "kern.out")],
                capture_output=True,
                text=True,
            )
            compile_ms = (time.time() - start) * 1e3
            if proc.returncode != 0:
                return {"ok": False, "log": proc.stderr, "compile_ms": compile_ms}

            t0 = time.time()
            run_proc = subprocess.run([str(Path(tmp) / "kern.out")], capture_output=True)
            runtime_ms = (time.time() - t0) * 1e3
            return {
                "ok": run_proc.returncode == 0,
                "log": proc.stderr + run_proc.stderr.decode(),
                "compile_ms": compile_ms,
                "runtime_ms": runtime_ms,
            }

    # ------------------------------------------------------------------
    # public tool entry‑point
    # ------------------------------------------------------------------
    def __call__(self, query: str, **_) -> str:
        """`query` is expected to be raw CUDA source code from the LLM."""
        res = self._compile_run(query)

        # track best runtime per conversation
        if res.get("ok") and res.get("runtime_ms") is not None:
            rt = res["runtime_ms"]
            if self.best_runtime_ms is None or rt < self.best_runtime_ms:
                self.best_runtime_ms = rt
            res["best_runtime_ms"] = self.best_runtime_ms
        return json.dumps(res)
