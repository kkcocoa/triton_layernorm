import torch
import numpy as np
from layernorm.layernorm_triton_v2 import layernorm_triton_v2

device="cuda"
torch.manual_seed(0)

@torch.no_grad()
def bench_fn(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    starter = torch.cuda.Event(True)
    ender   = torch.cuda.Event(True)
    times=[]
    for _ in range(iters):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))
    return float(np.median(times))

def main():
    M, N = 4096, 4096
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    w = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)

    BLOCKS = [256, 512, 1024, 2048]
    WARPS  = [2, 4, 8]

    results=[]
    for bn in BLOCKS:
        for nw in WARPS:
            fn = lambda: layernorm_triton_v2(x, w, b, eps=1e-5, BLOCK_N=bn, num_warps=nw)
            t = bench_fn(fn)
            results.append((bn, nw, t))
            print(f"BLOCK_N={bn:4d}, warps={nw:2d} -> {t:.3f} ms")

    best = min(results, key=lambda x: x[2])
    print("\nBEST CONFIG:", best)

if __name__ == "__main__":
    main()
