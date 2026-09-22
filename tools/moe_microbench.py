"""Mikrobenchmark MoE per warstwa: B12X static (launch_sm120_moe) vs FlashInfer CUTLASS (cutlass_fused_moe + kwantyzacja wejscia).
Ksztalt: E=256, K=hidden, N=intermediate/TP, top_k=8. Uzycie: moe_microbench.py [N=256] [M list]"""
import sys, time, torch
from flashinfer import nvfp4_quantize
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import _get_weight_views, allocate_sm120_moe_workspace, launch_sm120_moe

dev = torch.device("cuda:0")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
Ms = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '--dyn' else "1,2,4,6,12,24,48,80,96,192,512,2048").split(",")]
SKEW = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 0.0  # 0 = rownomierne; >0 = Zipf(1/(r+1)^SKEW)
E, K, TOPK = 256, 6144, 8
torch.manual_seed(0)
w13 = torch.randint(0, 255, (E, 2 * N, K // 2), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 255, (E, K, N // 2), dtype=torch.uint8, device=dev)
w13_sf = torch.randint(100, 120, (E, 2 * N, K // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
w2_sf = torch.randint(100, 120, (E, K, N // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
alpha1 = torch.rand(E, device=dev) * 0.01 + 0.005; alpha2 = torch.rand(E, device=dev) * 0.01 + 0.005; ones = torch.ones(E, device=dev)
a1_gs = torch.full((E,), 1.0 / 0.01, device=dev); a2_gs = torch.full((E,), 1.0 / 0.02, device=dev)
g1 = (alpha1 / a1_gs).contiguous(); g2 = (alpha2 / a2_gs).contiguous()
w13_mma = convert_sf_to_mma_layout(w13_sf.reshape(E * 2 * N, K // 16), m=2 * N, k=K, num_groups=E)
w2_mma = convert_sf_to_mma_layout(w2_sf.reshape(E * K, N // 16), m=K, k=N, num_groups=E)
MAXT = max(m for m in Ms if m * TOPK <= 640) if any(m * TOPK <= 640 for m in Ms) else 80
ws = allocate_sm120_moe_workspace(state_E=E, weight_E=E, max_rows=MAXT * TOPK, k=K, n=N, num_topk=TOPK, device=dev, quant_mode="nvfp4", backend="static", activation="silu")
views = _get_weight_views(w1_fp4=w13, w1_blockscale=w13_mma, w2_fp4=w2, w2_blockscale=w2_mma, w1_alphas=alpha1, w2_alphas=alpha2, n=N, k=K, activation_precision="fp4", quant_mode="nvfp4")


def bench(fn, iters=50):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


print(f"E={E} K={K} N={N} topk={TOPK} skew={SKEW}  (us na warstwe, CUDA graph replay)")
print(f"{'M':>5} {'routed':>7} {'B12X st/dyn':>12} {'CUTLASS(+quant)':>16} {'CUTLASS gemm':>13}")
for M in Ms:
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    if SKEW > 0:
        pr = 1.0 / torch.arange(1, E + 1, device=dev).float() ** SKEW
        ids = torch.multinomial(pr.expand(M, E), TOPK, replacement=False).to(torch.int32)
    else:
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    wts = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
    out = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
    b12x = None
    if M * TOPK > 640 and "--dyn" in sys.argv:
        ws_dyn = allocate_sm120_moe_workspace(state_E=E, weight_E=E, routed_rows=M * TOPK, k=K, n=N, num_topk=TOPK, device=dev, quant_mode="nvfp4", backend="dynamic", activation="silu")
        def f_bd():
            launch_sm120_moe(a=x, topk_ids=ids, topk_weights=wts, w1_weight=w13, w1_weight_sf=w13_mma, w1_alpha=alpha1, fc2_input_scale=ones, input_global_scale=ones,
                             w2_weight=w2, w2_weight_sf=w2_mma, w2_alpha=alpha2, num_experts=E, top_k=TOPK, num_local_experts=E, scatter_output=out,
                             activation="silu", activation_precision="fp4", quant_mode="nvfp4", _workspace=ws_dyn, _weight_views=views)
        try:
            b12x = bench(f_bd)
        except Exception as ex:
            print("b12x dynamic err:", str(ex)[:160]); b12x = float("nan")
        del ws_dyn
    if M * TOPK <= 640:
        def f_b():
            launch_sm120_moe(a=x, topk_ids=ids, topk_weights=wts, w1_weight=w13, w1_weight_sf=w13_mma, w1_alpha=alpha1, fc2_input_scale=ones, input_global_scale=ones,
                             w2_weight=w2, w2_weight_sf=w2_mma, w2_alpha=alpha2, num_experts=E, top_k=TOPK, num_local_experts=E, scatter_output=out,
                             activation="silu", activation_precision="fp4", quant_mode="nvfp4", _workspace=ws, _weight_views=views)
        b12x = bench(f_b)
    a1q, a1q_sf = nvfp4_quantize(x, a1_gs[:1], sf_vec_size=16)
    def f_c_full():
        q, sf = nvfp4_quantize(x, a1_gs[:1], sf_vec_size=16)
        cutlass_fused_moe(input=q, token_selected_experts=ids, token_final_scales=wts, fc1_expert_weights=w13.view(torch.long), fc2_expert_weights=w2.view(torch.long),
                          output_dtype=torch.bfloat16, output=out, quant_scales=[a1_gs, w13_sf.view(torch.int32), g1, a2_gs, w2_sf.view(torch.int32), g2], input_sf=sf,
                          tp_size=8, tp_rank=0, ep_size=1, ep_rank=0, activation_type=ActivationType.Swiglu)
    def f_c_gemm():
        cutlass_fused_moe(input=a1q, token_selected_experts=ids, token_final_scales=wts, fc1_expert_weights=w13.view(torch.long), fc2_expert_weights=w2.view(torch.long),
                          output_dtype=torch.bfloat16, output=out, quant_scales=[a1_gs, w13_sf.view(torch.int32), g1, a2_gs, w2_sf.view(torch.int32), g2], input_sf=a1q_sf,
                          tp_size=8, tp_rank=0, ep_size=1, ep_rank=0, activation_type=ActivationType.Swiglu)
    try:
        c_full = bench(f_c_full); c_gemm = bench(f_c_gemm)
    except Exception as ex:
        c_full = c_gemm = float("nan"); print("cutlass err:", str(ex)[:200])
    print(f"{M:>5} {M*TOPK:>7} {b12x if b12x is not None else float('nan'):>12.1f} {c_full:>16.1f} {c_gemm:>13.1f}")
