"""BF16 F.linear vs FP8 (torch._scaled_mm, per-tensor, dynamiczna kwantyzacja aktywacji) dla projekcji GLM-5.3 @TP4, M=6 i M=512.
Mierzy tez sam koszt kwantyzacji aktywacji (vllm ops.scaled_fp8_quant)."""
import sys, time, torch
from vllm import _custom_ops as ops

dev = torch.device("cuda:0")


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


shapes = [(6144, 2048, "q_a_proj"), (2048, 3072, "q_b_proj/TP4"), (6144, 576, "kv_a_proj_with_mqa"), (4096, 6144, "o_proj/TP4"),
          (2048, 4096, "indexer.wq_b"), (6144, 6144, "shared gate_up/TP4 (2x3072)"), (3072, 6144, "shared down/TP4"), (6144, 154880 // 4, "lm_head/TP4")]
for M in [int(m) for m in (sys.argv[1] if len(sys.argv) > 1 else "6,512").split(",")]:
    print(f"\nM={M}   us: bf16 | fp8 W8A8 (_scaled_mm) | +act quant | act quant alone | speedup (bf16 / (fp8+quant))")
    tot_bf, tot_fp = 0.0, 0.0
    for K, N, name in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        w = (torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02)
        w_fp8, w_scale = ops.scaled_fp8_quant(w)  # per-tensor
        wt = w_fp8.t()  # [K, N] col-major dla _scaled_mm
        t_bf = bench(lambda: torch.nn.functional.linear(x, w))
        x_fp8, x_scale = ops.scaled_fp8_quant(x)
        def f_mm():
            return torch._scaled_mm(x_fp8, wt, scale_a=x_scale, scale_b=w_scale, out_dtype=torch.bfloat16)
        def f_q():
            return ops.scaled_fp8_quant(x)
        def f_all():
            xq, xs = ops.scaled_fp8_quant(x)
            return torch._scaled_mm(xq, wt, scale_a=xs, scale_b=w_scale, out_dtype=torch.bfloat16)
        try:
            t_mm = bench(f_mm); t_all = bench(f_all); t_q = bench(f_q)
        except Exception as e:
            print(f"  {name:28s} K={K:5d} N={N:6d}: bf16 {t_bf:7.1f} | fp8 ERR {str(e)[:80]}"); continue
        tot_bf += t_bf; tot_fp += t_all
        print(f"  {name:28s} K={K:5d} N={N:6d}: {t_bf:7.1f} | {t_mm:7.1f} | {t_all:7.1f} | {t_q:5.1f} | x{t_bf / t_all:4.2f}")
    print(f"  suma (bez lm_head liczone razem): bf16 {tot_bf:.0f} us vs fp8 {tot_fp:.0f} us")
