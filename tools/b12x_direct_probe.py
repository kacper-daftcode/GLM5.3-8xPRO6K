"""Sprawdza, ze bezposrednie launch_sm120_moe(_workspace wspolny, _weight_views prebudowane, scatter_output=out)
daje to samo co B12xMoEWrapper.run (ksztalt GLM-5.3 @TP8) i mierzy pamiec."""
import torch, time
from flashinfer.fused_moe import B12xMoEWrapper
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import _get_weight_views, allocate_sm120_moe_workspace, launch_sm120_moe

dev = torch.device("cuda:0")
E, K, N, TOPK, MAXT = 256, 6144, 256, 8, 80
torch.manual_seed(0)
w13 = torch.randint(0, 255, (E, 2 * N, K // 2), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 255, (E, K, N // 2), dtype=torch.uint8, device=dev)
w13_sf = torch.randint(100, 120, (E, 2 * N, K // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
w2_sf = torch.randint(100, 120, (E, K, N // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
alpha1 = torch.rand(E, device=dev) * 0.01 + 0.005; alpha2 = torch.rand(E, device=dev) * 0.01 + 0.005; ones = torch.ones(E, device=dev)
w13_mma = convert_sf_to_mma_layout(w13_sf.reshape(E * 2 * N, K // 16), m=2 * N, k=K, num_groups=E)
w2_mma = convert_sf_to_mma_layout(w2_sf.reshape(E * K, N // 16), m=K, k=N, num_groups=E)

wr = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=K, intermediate_size=N, use_cuda_graph=True, max_num_tokens=MAXT, num_local_experts=E, activation="silu")
m0 = torch.cuda.memory_allocated()
ws = allocate_sm120_moe_workspace(state_E=E, weight_E=E, max_rows=MAXT * TOPK, k=K, n=N, num_topk=TOPK, device=dev, quant_mode="nvfp4", backend="static", activation="silu")
print(f"shared static workspace ({MAXT*TOPK} rows): {(torch.cuda.memory_allocated()-m0)/2**20:.0f} MiB")
m0 = torch.cuda.memory_allocated()
views = _get_weight_views(w1_fp4=w13, w1_blockscale=w13_mma, w2_fp4=w2, w2_blockscale=w2_mma, w1_alphas=alpha1, w2_alphas=alpha2, n=N, k=K, activation_precision="fp4", quant_mode="nvfp4")
print(f"weight views: {(torch.cuda.memory_allocated()-m0)/2**20:.1f} MiB (skale bez kopii => ~0)")

for M in (1, 3, 6, 24, 48, 80):
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    wts = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
    ref = wr.run(x=x, w1_weight=w13, w1_weight_sf=w13_mma, w2_weight=w2, w2_weight_sf=w2_mma, token_selected_experts=ids, token_final_scales=wts,
                 w1_alpha=alpha1, w2_alpha=alpha2, fc2_input_scale=ones, input_global_scale=ones).clone()
    out = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
    launch_sm120_moe(a=x, topk_ids=ids, topk_weights=wts, w1_weight=w13, w1_weight_sf=w13_mma, w1_alpha=alpha1, fc2_input_scale=ones, input_global_scale=ones,
                     w2_weight=w2, w2_weight_sf=w2_mma, w2_alpha=alpha2, num_experts=E, top_k=TOPK, num_local_experts=E, scatter_output=out,
                     activation="silu", activation_precision="fp4", quant_mode="nvfp4", _workspace=ws, _weight_views=views)
    torch.cuda.synchronize()
    # drugi raz (workspace juz uzyty) - sprawdza reuse
    out2 = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
    launch_sm120_moe(a=x, topk_ids=ids, topk_weights=wts, w1_weight=w13, w1_weight_sf=w13_mma, w1_alpha=alpha1, fc2_input_scale=ones, input_global_scale=ones,
                     w2_weight=w2, w2_weight_sf=w2_mma, w2_alpha=alpha2, num_experts=E, top_k=TOPK, num_local_experts=E, scatter_output=out2,
                     activation="silu", activation_precision="fp4", quant_mode="nvfp4", _workspace=ws, _weight_views=views)
    torch.cuda.synchronize()
    d = (out.float() - ref.float()).abs().max().item(); d2 = (out2.float() - out.float()).abs().max().item()
    print(f"M={M:>3}: max|direct-wrapper|={d:.3e} max|run2-run1|={d2:.3e} ref_absmax={ref.float().abs().max().item():.3e} finite={bool(torch.isfinite(out).all())}")

# CUDA graph capture test na sciezce bezposredniej
M = 6
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev); ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32); wts = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
out = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    launch_sm120_moe(a=x, topk_ids=ids, topk_weights=wts, w1_weight=w13, w1_weight_sf=w13_mma, w1_alpha=alpha1, fc2_input_scale=ones, input_global_scale=ones, w2_weight=w2, w2_weight_sf=w2_mma, w2_alpha=alpha2, num_experts=E, top_k=TOPK, num_local_experts=E, scatter_output=out, activation="silu", activation_precision="fp4", quant_mode="nvfp4", _workspace=ws, _weight_views=views)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    launch_sm120_moe(a=x, topk_ids=ids, topk_weights=wts, w1_weight=w13, w1_weight_sf=w13_mma, w1_alpha=alpha1, fc2_input_scale=ones, input_global_scale=ones, w2_weight=w2, w2_weight_sf=w2_mma, w2_alpha=alpha2, num_experts=E, top_k=TOPK, num_local_experts=E, scatter_output=out, activation="silu", activation_precision="fp4", quant_mode="nvfp4", _workspace=ws, _weight_views=views)
ref = wr.run(x=x, w1_weight=w13, w1_weight_sf=w13_mma, w2_weight=w2, w2_weight_sf=w2_mma, token_selected_experts=ids, token_final_scales=wts, w1_alpha=alpha1, w2_alpha=alpha2, fc2_input_scale=ones, input_global_scale=ones).clone()
out.zero_(); g.replay(); torch.cuda.synchronize()
print(f"cuda-graph replay: max|graph-wrapper|={(out.float()-ref.float()).abs().max().item():.3e}")
x.copy_(torch.randn(M, K, dtype=torch.bfloat16, device=dev)); ref = wr.run(x=x, w1_weight=w13, w1_weight_sf=w13_mma, w2_weight=w2, w2_weight_sf=w2_mma, token_selected_experts=ids, token_final_scales=wts, w1_alpha=alpha1, w2_alpha=alpha2, fc2_input_scale=ones, input_global_scale=ones).clone()
g.replay(); torch.cuda.synchronize()
print(f"cuda-graph replay (nowe x): max|graph-wrapper|={(out.float()-ref.float()).abs().max().item():.3e}")
t0 = time.perf_counter()
for _ in range(50): g.replay()
torch.cuda.synchronize(); print(f"graph replay M=6: {(time.perf_counter()-t0)/50*1e6:.0f} us/warstwa")
