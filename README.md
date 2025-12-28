day 1
We benchmarked torch.nn.LayerNorm under fp32 vs bf16:

bf16 did not outperform fp32 (speedup ≤ 1.0x)

This suggests LayerNorm is memory-bandwidth bound rather than compute-bound

Unlike GEMM, LayerNorm consists of reductions and elementwise ops, which do not benefit significantly from Tensor Cores

Therefore the optimization target is to fuse mean/var + normalize + affine into a single kernel to reduce memory traffic and kernel launch overhead.

day 2
Correctness

We validate the Triton fused LayerNorm output against PyTorch torch.nn.functional.layer_norm across multiple shapes.

Shape (M,N)	Max Abs Error	Max Rel Error
(1024, 1024)	1.43e-06	2.99e-02
(4096, 1024)	1.91e-06	1.39e-02
(4096, 4096)	1.91e-06	4.41e-02
(4096, 8192)	1.91e-06	1.03e-01

Note: Relative error can appear large for elements where the reference output is close to zero. Max absolute error remains at ~1e-6, indicating high numerical accuracy.

got speedup: X1.79