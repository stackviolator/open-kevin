# Open-Kevin

Open-Kevin is a small research framework for **reinforcement-learning from code execution**.
It uses the [verifiers](https://github.com/willccbb/verifiers) library to train
language-models that generate *CUDA-accelerated* PyTorch kernels.

The default task comes from **[KernelBench](https://github.com/scalingintelligence/KernelBench)**:
given a baseline PyTorch implementation, the model must output an equivalent
CUDA kernel that (1) compiles, (2) is numerically correct and (3) runs faster.

---

## Table of Contents
1. [Quick start](#quick-start)
2. [Installation](#installation)
3. [Project layout](#project-layout)
4. [Training](#training)
5. [Testing](#testing)
6. [Configuration](#configuration)
7. [License](#license)

---

## Quick start
```bash
# Clone the repo (with submodules) and create a Python 3.11 env …
git clone --recurse-submodules https://github.com/stackviolator/open-kevin.git
cd open-kevin
python -m venv .venv && source .venv/bin/activate

# Install in editable mode (CUDA toolkit + GPU drivers are required)
pip install -e .

# Run the unit-tests
pytest -q

# Launch a single-GPU training run
python -m open_kevin.cli.train
```

> **GPU requirements**  
> The default reward uses `torch.utils.cpp_extension.load_inline`, so you need
> a CUDA-capable GPU (tested on A100 and RTX 40-series) with a matching toolkit.

---

## Installation
Open-Kevin is packaged with PEP-517 metadata and lives under `src/`.
If you simply want to *use* the framework inside another project:
```bash
pip install git+https://github.com/stackviolator/open-kevin.git
```
Developers should use editable installs:
```bash
# Inside the repo root
pip install -e .[test]
```

---

## Project layout
```
open-kevin/
├── src/open_kevin/        # importable Python package
│   ├── cli/               # entry-points (train, utilities, …)
│   ├── prompts/           # system + template prompts
│   └── rewards/           # compilation / correctness / speedup metrics
├── kernelbench/           # Git submodule containing the evaluation harness
├── data/                  # Parquet datasets for train & hold-out
├── tests/                 # pytest suite
├── configs/               # YAML configs (hyper-params, cluster args, …)
└── README.md              # you are here
```

---

## Training
The canonical entry-point is `open_kevin.cli.train`.
It wraps `verifiers.GRPOTrainer` and accepts the same command-line flags.

### Single GPU
```bash
python -m open_kevin.cli.train \
    --max-steps 100 \
    --run-name local-test
```

### Multi-GPU (3 training GPUs + 1 inference GPU)
```bash
# GPU-0: vLLM inference, GPUs 1-3: GRPO training
CUDA_VISIBLE_DEVICES=0 vf-vllm --model qwen/qwen2.5-1.5b-instruct &
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file ~/verifiers/configs/zero3.yaml \
    -m open_kevin.cli.train
```

Hyper-parameters can also be supplied via `configs/*.yaml` files or
environment variables (see **Configuration**).

---

## Testing
```bash
pytest            # full suite
pytest -q tests/  # quiet mode
```
The reward tests exercise all major failure modes: bad format, compile error,
incorrect output, slow kernel and fast kernel.

---

## Configuration
Open-Kevin relies on `pydantic-settings`, so any field in the config can be
overridden by an environment variable with the `OK_` prefix:
```bash
OK_NUM_ITERATIONS=50 python -m open_kevin.cli.train
```
Check `open_kevin/cli/train.py` for default values.

---

## License
This project is released under the MIT License.  See [LICENSE](LICENSE).