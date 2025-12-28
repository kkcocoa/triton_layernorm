import torch
from torch.profiler import profile, ProfilerActivity
from layernorm.layernorm_triton import layernorm_triton

device="cuda"
M,N = 4096,4096

x = torch.randn(M,N, device=device)
w = torch.randn(N, device=device)
b = torch.randn(N, device=device)

# warmup
for _ in range(20):
    _ = layernorm_triton(x,w,b)

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        _ = layernorm_triton(x,w,b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
