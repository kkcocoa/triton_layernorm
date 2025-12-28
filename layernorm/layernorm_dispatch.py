import torch

from layernorm.layernorm_triton import layernorm_triton as layernorm_v1
from layernorm.layernorm_triton_v2 import layernorm_triton_v2 as layernorm_v2

def is_power_of_2(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0

def layernorm_triton_dispatch(x, w, b, eps=1e-5):
    """
    Dispatch between:
      - v1: fast single-pass fused (best for power-of-two N)
      - v2: general two-pass (works for any N)
    Returns: (y, path_name)
    """
    M, N = x.shape

    # You can tune this threshold later; keep conservative now
    if is_power_of_2(N) and N <= 8192:
        # v1 chooses BLOCK_N = next_power_of_2(N), effectively N itself
        y = layernorm_v1(x, w, b, eps=eps)
        return y, "fast_v1"
    else:
        # v2 needs explicit BLOCK_N; use best config from Day4
        y = layernorm_v2(x, w, b, eps=eps, BLOCK_N=512, num_warps=4)
        return y, "general_v2"
