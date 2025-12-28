import torch 
import numpy as np
from layernorm.layernorm_torch import TorchLayerNorm

torch.manual_seed(0)
device = 'cuda'

@torch.no_grad()
def bench(fn, x, iters = 200, warmup = 50):
    for _ in range(warmup):
        _ = fn(x)
    
    starter = torch.cuda.Event(enable_timing = True)
    ender = torch.cuda.Event(enable_timing = True)

    times = []
    for _ in range(iters):
        starter.record()
        _ = fn(x)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))
    
    med = float(np.median(times))
    avg = float(np.mean(times))
    return med, avg

def run_case(batch, hidden, dtype=torch.float32):
    x = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    model = TorchLayerNorm(hidden).eval()

    if dtype == torch.float32:
        fn = lambda x : model(x)
    else:
        fn = lambda x : torch.autocast(device_type="cuda", dtype=dtype).__enter__ or model()
    
    if dtype == torch.float32:
        med, avg = bench(fn, x)
    else:
        def fn_bf16(x):
            with torch.autocast("cuda", dtype=dtype):
                return model(x)
        med, avg = bench(fn_bf16, x)
    throughput = batch / (med / 1000.0)

    return  med, throughput

def main():
    shapes = [(1024, 1024), (4096, 1024), (4096, 4096)]
    print("LayerNorm baseline benchmark (RTX 3060 Laptop)")
    print("-"*80)
    print(f"{'shape':>14} | {'fp32(ms)':>10} {'bf16(ms)':>10} | {'speedup':>8} | {'fp32 thr':>12} {'bf16 thr':>12}")
    print("-"*80)
    for batch, hidden in shapes:
        fp32_t, fp32_thr = run_case(batch, hidden, torch.float32)
        bf16_t, bf16_thr = run_case(batch, hidden, torch.bfloat16)
        speedup = fp32_t / bf16_t
        print(f"{(batch, hidden)!s:>14} | {fp32_t:10.3f} {bf16_t:10.3f} | {speedup:8.2f} | {fp32_thr:12,.0f} {bf16_thr:12,.0f}")
if __name__ == "__main__":
    main()