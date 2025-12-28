import torch
from torch.profiler import profile, ProfilerActivity
from layernorm.layernorm_torch import TorchLayerNorm

device="cuda"
torch.manual_seed(0)

batch, hidden = 4096 * 4, 4096
x = torch.randn(batch, hidden, device=device)
model = TorchLayerNorm(hidden).eval()

# warmup
for _ in range(20):
    _ = model(x)

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        _ = model(x)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
