import torch
import numpy as np
from layernorm.layernorm_triton_v2 import layernorm_triton_v2

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
    M, N = 4096, 4096
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    w = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)

    torch_fn  = lambda: torch.nn.functional.layer_norm(x, (N,), w, b, eps=1e-5)
    triton_fn = lambda: layernorm_triton_v2(x, w, b, eps=1e-5, BLOCK_N=1024, num_warps=4)

    t_torch  = bench(torch_fn)
    t_triton = bench(triton_fn)

    print(f"torch layer_norm:   {t_torch:.3f} ms")
    print(f"triton layer_norm:  {t_triton:.3f} ms")
    print(f"speedup: {t_torch/t_triton:.2f}x")

if __name__ == "__main__":
    main()
