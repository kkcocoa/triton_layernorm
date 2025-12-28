We benchmarked torch.nn.LayerNorm under fp32 vs bf16:

bf16 did not outperform fp32 (speedup ≤ 1.0x)

This suggests LayerNorm is memory-bandwidth bound rather than compute-bound

Unlike GEMM, LayerNorm consists of reductions and elementwise ops, which do not benefit significantly from Tensor Cores

Therefore the optimization target is to fuse mean/var + normalize + affine into a single kernel to reduce memory traffic and kernel launch overhead.