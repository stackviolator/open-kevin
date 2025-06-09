import pytest
from kevin_reward import compute_score

# Baseline correct PyTorch implementation for vector addition
# This will be used as the reference by the scoring function.
PYTORCH_ADD_VECTORS = """
import torch
import torch.nn as nn

class Model(nn.Module):
    '''
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    '''
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        '''
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        '''
        return torch.diag(A) @ B

M = 4096
N = 4096

def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
"""

# A correct CUDA implementation that should be faster
CUDA_CORRECT_FAST = """
<code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_matmul_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(
    const float* diag,
    const float* mat,
    float* out,
    const int N,
    const int M) {
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        out[row * M + col] = diag[row] * mat[row * M + col];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor diag, torch::Tensor mat) {
    const int N = mat.size(0);
    const int M = mat.size(1);
    auto out = torch::empty_like(mat);

    const dim3 threads(16, 16);
    const dim3 blocks((M + threads.x - 1) / threads.x,
                      (N + threads.y - 1) / threads.y);

    diag_matmul_kernel<<<blocks, threads>>>(
        diag.data_ptr<float>(),
        mat.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        M
    );

    return out;
}
'''

diag_matmul_cpp = "torch::Tensor diag_matmul_cuda(torch::Tensor diag, torch::Tensor mat);"

diag_matmul = load_inline(
    name='diag_matmul',
    cpp_sources=[diag_matmul_cpp],
    cuda_sources=[diag_matmul_source],
    functions=['diag_matmul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = diag_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.op.diag_matmul_cuda(A, B)
</code>
"""

# Correct logic but implemented inefficiently to be slower
CUDA_CORRECT_SLOW = CUDA_CORRECT_FAST.replace(
    "__global__ void diag_matmul_kernel(",
    "__global__ void diag_matmul_kernel(\n    for(volatile int i=0; i<1000000; ++i); // artificial delay\n    ("  # noqa: E501
)

# Compiles but produces incorrect output (multiplies by diag twice)
CUDA_INCORRECT_OUTPUT = CUDA_CORRECT_FAST.replace("out[row * M + col] = diag[row] * mat[row * M + col];", "out[row * M + col] = diag[row] * diag[row] * mat[row * M + col];")

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


def test_incorrect_output():
    """R2: Incorrect output -> 0.2"""
    assert compute_score(PYTORCH_ADD_VECTORS, CUDA_INCORRECT_OUTPUT) == pytest.approx(0.2)


def test_correct_but_slower():
    """R3: Correct but not faster -> 0.3"""
    # This test is sensitive to system load. It might be flaky.
    # It assumes the pytorch version is faster than the serially-launched kernel.
    assert compute_score(PYTORCH_ADD_VECTORS, CUDA_CORRECT_SLOW) == pytest.approx(0.3)


def test_correct_and_faster():
    """R4: Correct and faster -> > 0.4"""
    reward = compute_score(PYTORCH_ADD_VECTORS, CUDA_CORRECT_FAST)
    assert reward > 0.4