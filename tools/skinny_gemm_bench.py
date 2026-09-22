"""Skinny GEMM (M<=32) BF16: cuBLAS (torch.matmul / F.linear) vs Triton split-K, dla ksztaltow GLM-5.3 @TP4 (decode M=6).
Wagi [N, K] jak w nn.Linear (y = x @ W^T)."""
import sys, time, torch, triton, triton.language as tl

dev = torch.device("cuda:0")


@triton.jit
def _skinny_splitk(x_ptr, w_ptr, y_ptr, M, N, K, stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_per = K // SPLIT_K
    k0 = pid_k * k_per
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, k_per, BLOCK_K):
        offs_k = k0 + k + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, mask=(offs_m[:, None] < M), other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=(offs_n[:, None] < N), other=0.0)
        acc += tl.dot(x, tl.trans(w))
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.atomic_add(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def skinny(x, w, split_k, block_n=64, block_k=64):
    M, K = x.shape; N = w.shape[0]
    y = torch.zeros((M, N), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, block_n), split_k)
    _skinny_splitk[grid](x, w, y, M, N, K, x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1),
                         BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k, BLOCK_M=16)
    return y.to(torch.bfloat16)


def bench(fn, iters=200):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


M = int(sys.argv[1]) if len(sys.argv) > 1 else 6
# (K, N, opis) - warstwa GLM-5.3, TP4: kv_a 6144->576 (repl), indexer wk 6144->128, weights_proj 6144->32, wq_b 2048->4096 (BF16, repl?),
# q_a 6144->2048 (NVFP4 w modelu; tu bf16 dla referencji), o_proj (16 glow*256=4096 -> 6144), shared gate_up 6144->2*3072? (TP4: 12288/4*2=6144)
shapes = [(6144, 576, "kv_a_proj_with_mqa"), (6144, 128, "indexer.wk"), (6144, 32, "indexer.weights_proj"), (2048, 4096, "indexer.wq_b"),
          (6144, 736, "fused kv_a+wk+wproj"), (6144, 2048, "q_a_proj (bf16 ref)"), (2048, 3072, "q_b_proj/4 (bf16 ref)"), (4096, 6144, "o_proj (bf16 ref)")]
print(f"M={M}   us: cuBLAS(F.linear) | triton split-K best (split,blockN)")
for K, N, name in shapes:
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev); w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02
    ref = torch.nn.functional.linear(x, w)
    t_cublas = bench(lambda: torch.nn.functional.linear(x, w))
    best = (1e9, None)
    for split in (1, 2, 4, 8, 16, 32):
        if K % (split * 64) != 0: continue
        for bn in (32, 64, 128):
            if bn > N and N >= 32: continue
            try:
                y = skinny(x, w, split, bn)
                err = (y.float() - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-6)
                if err > 2e-2: continue
                t = bench(lambda: skinny(x, w, split, bn))
                if t < best[0]: best = (t, (split, bn))
            except Exception as e:
                pass
    print(f"  {name:28s} K={K:5d} N={N:5d}: {t_cublas:7.1f} | {best[0]:7.1f} {best[1]}   ({(t_cublas - best[0]):+.1f} us, bytes W={K*N*2/2**20:.1f} MiB)")
