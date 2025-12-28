import torch
import triton
import triton.language as tl

# -----------------------------
# Pass 1: compute mean and rstd
# -----------------------------
@triton.jit
def layernorm_stats_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    stride_xm, stride_xn,
    stride_mean, stride_rstd,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute mean and rstd for each row.
    """
    m = tl.program_id(0)

    # accumulate sum and sumsq in fp32
    # Use 0-d scalars so tl.store writes a scalar value.
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    # loop over tiles of columns
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        offs = m * stride_xm + cols * stride_xn
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    rstd = tl.rsqrt(var + eps)

    tl.store(mean_ptr + m * stride_mean, mean)
    tl.store(rstd_ptr + m * stride_rstd, rstd)


# -----------------------------
# Pass 2: normalize + affine
# -----------------------------
@triton.jit
def layernorm_fwd_kernel_v2(
    x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_mean, stride_rstd,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Normalize and apply affine transform for each row.
    """
    m = tl.program_id(0)

    mean = tl.load(mean_ptr + m * stride_mean).to(tl.float32)
    rstd = tl.load(rstd_ptr + m * stride_rstd).to(tl.float32)

    start = tl.program_id(1) * BLOCK_N
    cols = start + tl.arange(0, BLOCK_N)
    mask = cols < N

    offs_x = m * stride_xm + cols * stride_xn
    x = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd
    y = y * w + b

    offs_y = m * stride_ym + cols * stride_yn
    tl.store(y_ptr + offs_y, y, mask=mask)


def layernorm_triton_v2(x, weight, bias, eps=1e-5, BLOCK_N=1024, num_warps=4):
    """
    Multi-pass LayerNorm forward:
      - pass1 compute mean, rstd
      - pass2 normalize + affine (tiled)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, N = x.shape
    y = torch.empty_like(x)

    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    # ---- pass1 grid: one program per row
    grid1 = (M,)
    layernorm_stats_kernel[grid1](
        x, mean, rstd,
        x.stride(0), x.stride(1),
        mean.stride(0), rstd.stride(0),
        N=N, eps=eps, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    # ---- pass2 grid: (M, ceil_div(N, BLOCK_N))
    grid2 = (M, triton.cdiv(N, BLOCK_N))
    layernorm_fwd_kernel_v2[grid2](
        x, y, weight, bias, mean, rstd,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        mean.stride(0), rstd.stride(0),
        N=N, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return y


# -----------------------------
# Correctness Test
# -----------------------------
def test_correctness():
    torch.manual_seed(0)
    device = "cuda"

    shapes = [
        (1024, 1024),
        (4096, 1024),
        (4096, 4096),
        (4096, 8192),
        (4096, 5000),   # 非 2^k 形状，用来验证通用性
    ]

    for M, N in shapes:
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        w = torch.randn(N, device=device, dtype=torch.float32)
        b = torch.randn(N, device=device, dtype=torch.float32)

        ref = torch.nn.functional.layer_norm(x, (N,), w, b, eps=1e-5)
        out = layernorm_triton_v2(x, w, b, eps=1e-5, BLOCK_N=1024, num_warps=4)

        max_abs = (ref - out).abs().max().item()
        max_rel = ((ref - out).abs() / (ref.abs() + 1e-6)).max().item()

        print(f"[M={M}, N={N}] max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")

        assert max_abs < 5e-3 or max_rel < 5e-3, "Too much error!"

    print("✅ correctness passed (v2)")


if __name__ == "__main__":
    test_correctness()
