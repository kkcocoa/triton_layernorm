import torch
from layernorm.layernorm_dispatch import layernorm_triton_dispatch

device="cuda"
torch.manual_seed(0)

shapes = [
    (1024,1024),
    (4096,4096),
    (4096,5000),
    (4096,8192),
]

for M,N in shapes:
    x = torch.randn(M,N, device=device)
    w = torch.randn(N, device=device)
    b = torch.randn(N, device=device)

    ref = torch.nn.functional.layer_norm(x, (N,), w, b, eps=1e-5)
    out, path = layernorm_triton_dispatch(x, w, b, eps=1e-5)

    max_abs = (ref - out).abs().max().item()
    max_rel = ((ref - out).abs() / (ref.abs() + 1e-6)).max().item()

    print(f"[{(M,N)}] path={path:10s} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")

print("✅ dispatch correctness passed")
