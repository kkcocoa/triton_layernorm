import torch
import numpy as np
from layernorm.layernorm_triton import layernorm_fwd_kernel

device="cuda"
torch.manual_seed(0)

@torch.no_grad()
def bench_kernel(x, w, b, y, BLOCK_N, num_warps, iters=200, warmup=50):
    M, N = x.shape
    grid = (M,)

    # warmup
    for _ in range(warmup):
        layernorm_fwd_kernel[grid](
            x, y, w, b,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            N=N, eps=1e-5, BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )

    starter = torch.cuda.Event(True)
    ender   = torch.cuda.Event(True)

    times=[]
    for _ in range(iters):
        starter.record()
        layernorm_fwd_kernel[grid](
            x, y, w, b,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            N=N, eps=1e-5, BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))
    return float(np.median(times))

def main():
    M, N = 4096, 4096
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    w = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    BLOCKS = [512, 1024, 2048, 4096, 8192]
    WARPS  = [2, 4, 8]

    results=[]
    for bn in BLOCKS:
        for nw in WARPS:
            t = bench_kernel(x, w, b, y, bn, nw)
            results.append((bn, nw, t))
            print(f"BLOCK_N={bn:5d}, warps={nw:2d} -> {t:.3f} ms")

    best = min(results, key=lambda x: x[2])
    print("\nBEST CONFIG:", best)

if __name__ == "__main__":
    main()
