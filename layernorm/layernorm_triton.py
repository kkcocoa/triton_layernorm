import torch
import triton
import triton.language as tl

# -----------------------------
# Triton LayerNorm Kernel
# -----------------------------
@triton.jit
def layernorm_fwd_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    x: [M, N]
    y: [M, N]
    w,b: [N]  (gamma, beta)
    each program handles one row m
    """
    m = tl.program_id(0)

    # offsets for the row
    cols = tl.arange(0, BLOCK_N)
    offs = m * stride_xm + cols * stride_xn
    mask = cols < N

    # load x
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # mean
    mean = tl.sum(x, axis=0) / N

    # variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N

    # inv std
    rstd = tl.rsqrt(var + eps)

    # normalize
    xhat = x_centered * rstd

    # affine
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xhat * w + b

    # store
    y_offs = m * stride_ym + cols * stride_yn
    tl.store(y_ptr + y_offs, y, mask=mask)


def layernorm_triton(x, weight, bias, eps=1e-5):
    """
    x: [M, N] fp32
    weight/bias: [N] fp32
    returns y fp32
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, N = x.shape
    y = torch.empty_like(x)

    # choose block size: first correct, later tune
    BLOCK_N = triton.next_power_of_2(N)
    # avoid too large block
    BLOCK_N = min(BLOCK_N, 8192)

    grid = (M,)

    layernorm_fwd_kernel[grid](
        x, y, weight, bias,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        N=N,
        eps=eps,
        BLOCK_N=BLOCK_N,
        num_warps=4,
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
    ]

    for M, N in shapes:
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        w = torch.randn(N, device=device, dtype=torch.float32)
        b = torch.randn(N, device=device, dtype=torch.float32)

        # reference
        ref = torch.nn.functional.layer_norm(x, (N,), w, b, eps=1e-5)

        # triton
        out = layernorm_triton(x, w, b, eps=1e-5)

        max_abs = (ref - out).abs().max().item()
        max_rel = ((ref - out).abs() / (ref.abs() + 1e-6)).max().item()

        print(f"[M={M}, N={N}] max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")

        # loose threshold for first pass
        assert max_abs < 5e-3 or max_rel < 5e-3, "Too much error!"

    print("✅ correctness passed")


if __name__ == "__main__":
    test_correctness()
