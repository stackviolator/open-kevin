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
    final_code = wrapper_code + "\n" + code_with_decorator + """
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

def _generate_test_data(pytorch_code: str):
    """Generate test data by executing get_inputs() and get_init_inputs() from PyTorch code."""
    # Extract and execute the PyTorch code to get the data generation functions
    test_data = _extract_and_run_data_generators(pytorch_code)
    return test_data

def _extract_and_run_data_generators(pytorch_code: str) -> Dict:
    """Extract get_inputs() and get_init_inputs() functions and execute them."""
    import tempfile
    import pickle
    
    # Create a temporary script that runs the data generators and saves results
    data_gen_script = f"""
import torch
import numpy as np
import pickle
import sys

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

{pytorch_code}

# Execute data generation functions
try:
    init_inputs = get_init_inputs() if 'get_init_inputs' in globals() else []
    inputs = get_inputs()
    
    # Convert tensors to numpy for serialization
    def tensor_to_numpy(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().numpy()
        elif isinstance(obj, list):
            return [tensor_to_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(tensor_to_numpy(item) for item in obj)
        elif isinstance(obj, dict):
            return {{k: tensor_to_numpy(v) for k, v in obj.items()}}
        else:
            return obj
    
    # Save the data
    data = {{
        'init_inputs': tensor_to_numpy(init_inputs),
        'inputs': tensor_to_numpy(inputs),
        'success': True
    }}
    
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(data, f)
        
except Exception as e:
    data = {{'success': False, 'error': str(e)}}
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(data, f)
"""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "generate_data.py"
        script_path.write_text(data_gen_script)
        
        data_path = Path(tmp_dir) / "test_data.pkl"
        
        # Run the script
        proc = subprocess.run(
            ["python", str(script_path)],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S
        )
        
        if proc.returncode != 0 or not data_path.exists():
            raise RuntimeError(f"Failed to extract test data: {proc.stderr}")
        
        # Load the generated data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        if not data.get('success', False):
            raise RuntimeError(f"Data generation failed: {data.get('error', 'Unknown error')}")
        
        return data

def _write_test_data_files(test_data: Dict, tmp_dir: str):
    """Write test data to binary files and create a metadata file for CUDA."""
    files_created = []
    metadata = {
        'init_inputs': test_data.get('init_inputs', []),
        'input_files': [],
        'input_shapes': [],
        'input_dtypes': []
    }
    
    # Write each input tensor to a separate binary file
    inputs = test_data.get('inputs', [])
    for i, input_data in enumerate(inputs):
        if isinstance(input_data, np.ndarray):
            file_path = Path(tmp_dir) / f"input_{i}.bin"
            input_data.astype(np.float32).tofile(str(file_path))
            files_created.append(str(file_path))
            
            metadata['input_files'].append(f"input_{i}.bin")
            metadata['input_shapes'].append(list(input_data.shape))
            metadata['input_dtypes'].append('float32')
    
    # Write metadata as JSON
    metadata_path = Path(tmp_dir) / "metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    return files_created, str(metadata_path)

def _run_compiled_kernel_with_data(executable_path: str, test_data: Dict) -> Dict:
    """Runs a compiled CUDA executable with test data and captures output."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_files, metadata_path = _write_test_data_files(test_data, tmp_dir)
        output_path = Path(tmp_dir) / "output.bin"
        
        # CUDA kernel gets: executable metadata_file output_file
        t0 = time.time()
        run_proc = subprocess.run(
            [executable_path, metadata_path, str(output_path)],
            capture_output=True, text=True,
        )
        runtime_ms = (time.time() - t0) * 1e3

        if run_proc.returncode != 0:
            return {"ok": False, "log": run_proc.stderr, "stdout": run_proc.stdout}

        # Read output and convert to string format for comparison
        if output_path.exists():
            output_array = np.fromfile(str(output_path), dtype=np.float32)
            # Format output for comparison (first 100 values)
            output_str = " ".join(f"{x:.6f}" for x in output_array[:100])
        else:
            output_str = ""

        return {
            "ok": True,
            "log": run_proc.stderr,
            "stdout": output_str,
            "runtime_ms": runtime_ms,
        }

def _run_pytorch_kernel_with_data(code: str, test_data: Dict) -> Dict[str, float | str | bool]:
    """Run PyTorch kernel with test data."""
    # Modify the code to use our test data
    instrumented_code = _instrument_pytorch_code_with_data(code, test_data)
    
    if instrumented_code is None:
        return {"ok": False, "log": "Failed to instrument PyTorch code"}

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

        try:
            runtime_ms = float(proc.stderr.strip())
        except (ValueError, IndexError):
            return {"ok": False, "log": "Failed to parse runtime from instrumented code: " + proc.stderr}

        return {
            "ok": True,
            "stdout": proc.stdout.strip(),
            "runtime_ms": runtime_ms,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "log": "timeout", "timeout": True}
    finally:
        os.unlink(py_path)

def _instrument_pytorch_code_with_data(code: str, test_data: Dict) -> str | None:
    """Instrument PyTorch code to use our test data and measure timing."""
    import json
    
    # Create the wrapper and test execution code
    wrapper_code = f"""
import time
import sys
import torch
import numpy as np
import json

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

__start_time = None
__end_time = None

def time_this(func):
    def wrapper(*args, **kwargs):
        global __start_time, __end_time
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, AttributeError):
            pass
        __start_time = time.perf_counter()
        result = func(*args, **kwargs)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, AttributeError):
            pass
        __end_time = time.perf_counter()
        return result
    return wrapper

# Test data (loaded from generated data)
test_data = {json.dumps(test_data)}
"""

    # Find Model class and forward method to instrument
    model_match = re.search(r'class\s+Model\s*\([^)]*\):', code, re.DOTALL)
    if model_match:
        # Instrument the forward method
        forward_match = re.search(r'def\s+forward\s*\([^)]*\):', code, re.DOTALL)
        if forward_match:
            forward_def = forward_match.group(0)
            instrumented_code = code.replace(forward_def, f"@time_this\n    {forward_def}")
        else:
            instrumented_code = code
    else:
        # Look for standalone function to instrument
        func_match = re.search(r"def\s+(\w+)\s*\(.*\):", code, re.DOTALL)
        if func_match:
            func_def = func_match.group(0)
            instrumented_code = code.replace(func_def, f"@time_this\n{func_def}")
        else:
            instrumented_code = code

    # Create execution code
    exec_code = """
# Reconstruct numpy arrays from test data
def numpy_from_list(data):
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        return np.array(data, dtype=np.float32)
    return data

# Convert test data back to tensors
init_inputs = test_data.get('init_inputs', [])
inputs = test_data.get('inputs', [])

# Convert numpy arrays back from lists
inputs = [torch.from_numpy(numpy_from_list(inp)) if isinstance(inp, list) else torch.tensor(inp) for inp in inputs]

# Create and run the model
if 'Model' in globals():
    model = Model(*init_inputs)
    model.eval()
    with torch.no_grad():
        result = model(*inputs)
else:
    # Assume there's a function we can call directly
    result = forward(*inputs) if 'forward' in globals() else None

# Format output for comparison (first 100 values)
if result is not None:
    if hasattr(result, 'numpy'):
        result_np = result.detach().numpy()
    else:
        result_np = np.array(result)
    
    # Flatten and take first 100 values
    flat_result = result_np.flatten()
    output_str = " ".join(f"{x:.6f}" for x in flat_result[:100])
    print(output_str)

if __start_time is not None and __end_time is not None:
    runtime_ms = (__end_time - __start_time) * 1000
    print(str(runtime_ms), file=sys.stderr, flush=True)
"""

    return wrapper_code + "\n" + instrumented_code + "\n" + exec_code

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

        # Generate test data from PyTorch code
        test_data = _generate_test_data(original_code)
        
        # R2: No runtime errors (test with random data)
        executable_path = compile_result["executable_path"]
        run_result = _safe_exec(_run_compiled_kernel_with_data, executable_path, test_data)
        if not run_result.get("ok"):
            return 0.2
        pytorch_result = _run_pytorch_kernel_with_data(original_code, test_data)
        if not pytorch_result.get("ok"):
            return 0.2

        # R3: Correct output (on random data)
        pytorch_runtime_ms = pytorch_result["runtime_ms"]
        reference_output = pytorch_result["stdout"]
        kernel_output = run_result.get("stdout", "")
        if not check_correctness(kernel_output, reference_output):
            return 0.3

        # R4: Correct but not faster than baseline
        runtime_ms = run_result["runtime_ms"]
        print(f"CUDA runtime: {runtime_ms:.3f}ms, PyTorch runtime: {pytorch_runtime_ms:.3f}ms")
        if runtime_ms >= pytorch_runtime_ms:
            return 0.4

        # R5: Correct and faster
        speedup = pytorch_runtime_ms / runtime_ms
        # Scale reward between 0.4 and 1.0 for speedups up to 10x
        return 0.4 + 0.6 * min((speedup - 1) / 9, 1)
