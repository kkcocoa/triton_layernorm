import torch
import numpy as np
from layernorm.layernorm_dispatch import layernorm_triton_dispatch

device="cuda"
torch.manual_seed(0)

@torch.no_grad()
def bench(fn, iters=200, warmup=50):
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
    # include both pow2 and non-pow2 N
    shapes = [
        (1024, 1024),
        (4096, 1024),
        (4096, 4096),
        (4096, 8192),
        (4096, 5000),
        (8192, 4096),
    ]

    print("LayerNorm Dispatch Benchmark (fast_v1 vs general_v2)")
    print("-" * 100)
    print(f"{'shape':>14} | {'torch(ms)':>10} {'triton(ms)':>10} | {'speedup':>8} | {'path':>10}")
    print("-" * 100)

    for M, N in shapes:
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        w = torch.randn(N, device=device, dtype=torch.float32)
        b = torch.randn(N, device=device, dtype=torch.float32)

        torch_fn = lambda: torch.nn.functional.layer_norm(x, (N,), w, b, eps=1e-5)

        # we want path info from dispatch
        def triton_fn():
            y, _path = layernorm_triton_dispatch(x, w, b, eps=1e-5)
            return y

        # also capture path separately (not inside benchmark loop)
        _, path = layernorm_triton_dispatch(x, w, b, eps=1e-5)

        t_torch  = bench(torch_fn)
        t_triton = bench(triton_fn)

        speedup = t_torch / t_triton

        print(f"{(M,N)!s:>14} | {t_torch:10.3f} {t_triton:10.3f} | {speedup:8.2f} | {path:>10}")

if __name__ == "__main__":
    main()
