"""Pomiar pamieci B12X (FlashInfer 0.6.18) dla ksztaltu GLM-5.3 @TP8: E=256, k=6144, n=256, top_k=8."""
import torch, time
from flashinfer.fused_moe import B12xMoEWrapper
from flashinfer import nvfp4_quantize
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

dev = torch.device("cuda:0")
E, K, N, TOPK = 256, 6144, 256, 8


def mem(tag):
    torch.cuda.synchronize()
    print(f"{tag:>40}: allocated={torch.cuda.memory_allocated() / 2**30:6.2f} GiB reserved={torch.cuda.memory_reserved() / 2**30:6.2f} GiB", flush=True)


mem("start")
# wagi FP4 jak w vLLM: w13 [E, 2N, K/2] uint8, w2 [E, K, N/2]; skale swizzlowane [E, 2N, K/16], [E, K, N/16]
w13 = torch.randint(0, 255, (E, 2 * N, K // 2), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 255, (E, K, N // 2), dtype=torch.uint8, device=dev)
w13_sf = torch.randint(100, 120, (E, 2 * N, K // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
w2_sf = torch.randint(100, 120, (E, K, N // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
alpha1 = torch.full((E,), 0.01, device=dev); alpha2 = torch.full((E,), 0.01, device=dev); ones = torch.ones(E, device=dev)
mem("po wagach (1 warstwa)")
w13_mma = convert_sf_to_mma_layout(w13_sf.reshape(E * 2 * N, K // 16), m=2 * N, k=K, num_groups=E)
w2_mma = convert_sf_to_mma_layout(w2_sf.reshape(E * K, N // 16), m=K, k=N, num_groups=E)
print("w13_mma contiguous:", w13_mma.is_contiguous(), "same storage:", w13_mma.data_ptr() == w13_sf.data_ptr())
mem("po convert_sf_to_mma (widoki)")

for max_tok in (80, 2048):
    torch.cuda.empty_cache(); base = torch.cuda.memory_allocated()
    wr = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=K, intermediate_size=N, use_cuda_graph=True, max_num_tokens=max_tok, num_local_experts=E, activation="silu")
    mem(f"wrapper(max_num_tokens={max_tok})")
    for M in (1, 6, 24, 80):
        if M > max_tok: break
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        wts = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
        t0 = time.time()
        out = wr.run(x=x, w1_weight=w13, w1_weight_sf=w13_mma, w2_weight=w2, w2_weight_sf=w2_mma, token_selected_experts=ids, token_final_scales=wts,
                     w1_alpha=alpha1, w2_alpha=alpha2, fc2_input_scale=ones, input_global_scale=ones)
        torch.cuda.synchronize()
        mem(f"  po run M={M} ({time.time() - t0:.1f}s, out finite={bool(torch.isfinite(out).all())})")
    if max_tok >= 2048:
        for M in (256, 2048):
            x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
            ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
            wts = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
            t0 = time.time()
            out = wr.run(x=x, w1_weight=w13, w1_weight_sf=w13_mma, w2_weight=w2, w2_weight_sf=w2_mma, token_selected_experts=ids, token_final_scales=wts,
                         w1_alpha=alpha1, w2_alpha=alpha2, fc2_input_scale=ones, input_global_scale=ones)
            torch.cuda.synchronize()
            mem(f"  po run M={M} ({time.time() - t0:.1f}s)")
    print(f"  => narzut wrappera {max_tok}: {(torch.cuda.memory_allocated() - base) / 2**20:.0f} MiB (x75 warstw = {(torch.cuda.memory_allocated() - base) * 75 / 2**30:.1f} GiB)")
    del wr
