import pytest
from kevin_reward import compute_score

# Baseline correct PyTorch implementation for vector addition
# This will be used as the reference by the scoring function.
PYTORCH_ADD_VECTORS = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b

def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
"""

# A correct CUDA implementation that should be faster
CUDA_CORRECT_FAST = """
<code>
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
</code>
"""

# Correct logic but implemented inefficiently to be slower
CUDA_CORRECT_SLOW = CUDA_CORRECT_FAST.replace(
    "int main(int argc, char* argv[]) {",
    "int main(int argc, char* argv[]) { for(volatile int i=0; i<100000000; ++i); "
)

# Compiles but produces incorrect output (adds a to itself)
CUDA_INCORRECT_OUTPUT = CUDA_CORRECT_FAST.replace("c[idx] = a[idx] + b[idx];", "c[idx] = a[idx] + a[idx];")

# This should cause a segmentation fault on the host
CUDA_RUNTIME_ERROR = CUDA_CORRECT_FAST.replace(
    "int main(int argc, char* argv[]) {",
    "int main(int argc, char* argv[]) { float* p = NULL; *p = 1.0f;"
)

# Code that will not compile
CUDA_COMPILE_ERROR = "<code> int main() { ERROR SYNTAX; } </code>"

# Not wrapped in code tags
NO_CODE_TAGS = "int main() { return 0; }"


def test_bad_format():
    """R0: Bad format -> 0.0"""
    assert compute_score(PYTORCH_ADD_VECTORS, NO_CODE_TAGS) == 0.0

def test_compile_error():
    """R1: Doesn't compile -> 0.1"""
    assert compute_score(PYTORCH_ADD_VECTORS, CUDA_COMPILE_ERROR) == pytest.approx(0.1)

def test_runtime_error():
    """R2: Runtime error -> 0.2"""
    assert compute_score(PYTORCH_ADD_VECTORS, CUDA_RUNTIME_ERROR) == pytest.approx(0.2)

def test_incorrect_output():
    """R3: Incorrect output -> 0.3"""
    assert compute_score(PYTORCH_ADD_VECTORS, CUDA_INCORRECT_OUTPUT) == pytest.approx(0.3)

def test_correct_but_slower():
    """R4: Correct but not faster -> 0.4"""
    # This test is sensitive to system load. It might be flaky.
    # It assumes the pytorch version is faster than the serially-launched kernel.
    assert compute_score(PYTORCH_ADD_VECTORS, CUDA_CORRECT_SLOW) == pytest.approx(0.4)

def test_correct_and_faster():
    """R5: Correct and faster -> > 0.4"""
    reward = compute_score(PYTORCH_ADD_VECTORS, CUDA_CORRECT_FAST)
    assert reward > 0.4 