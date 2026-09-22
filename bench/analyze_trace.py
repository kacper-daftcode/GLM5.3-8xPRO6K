#!/usr/bin/env python3
"""Rozklad czasu GPU per kategoria kerneli ze sladu torch.profiler (Chrome trace JSON/.gz) z vLLM.
Uzycie: analyze_trace.py <trace.json[.gz]> [--rank 0] [--top 25]
Kategorie: nccl (komunikacja), moe_fp4 (CUTLASS NVFP4 MoE), moe_other, attn_mla (sparse MLA decode/prefill),
indexer (DSA top-k/indexer), kv_nvfp4 (wasz cache ext), gemm (cuBLAS/CUTLASS dense), norm/act/elementwise,
sampler/topk, memcpy/memset, other. Liczy tez luki bezczynnosci GPU miedzy kernelami (CPU/launch overhead).
"""
import json, sys, gzip, re, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--min-gap-us", type=float, default=20.0)
a = ap.parse_args()

op = gzip.open if a.trace.endswith(".gz") else open
with op(a.trace, "rt") as f:
    data = json.load(f)
events = data["traceEvents"] if isinstance(data, dict) else data

CATS = [
    ("nccl", re.compile(r"ncclDevKernel|ncclKernel|AllReduce|AllGather|ReduceScatter|SendRecv|nccl", re.I)),
    ("kv_nvfp4", re.compile(r"nvfp4_cache|nvfp4_expand|concat_and_cache|nvfp4.*kv|kv.*nvfp4|reshape_and_cache", re.I)),
    ("indexer", re.compile(r"indexer|topk|top_k|fp8_paged_mqa_logits|lightning|sparse_attn_indexer|flatten", re.I)),
    ("attn_mla", re.compile(r"mla|BatchDecode|BatchPrefill|flash_attn|fmha|attention|sparse", re.I)),
    ("moe_fp4", re.compile(r"fp4.*(moe|grouped)|moe.*fp4|cutlass.*moe|fused_moe|grouped_gemm|MoeGemm|expert", re.I)),
    ("moe_other", re.compile(r"moe|router|gating|permute|unpermute|topk_softmax|silu_and_mul", re.I)),
    ("gemm", re.compile(r"gemm|cublas|cutlass|Kernel.*sm\d+|nvjet|ampere|xmma|wgmma|tcgen", re.I)),
    ("norm_act_ew", re.compile(r"norm|rope|rotary|act|elementwise|vectorized|copy_|fill|arange|index|gather|scatter|cat|add|mul|convert|cast", re.I)),
    ("sampler", re.compile(r"sampl|rejection|softmax|argmax|multinomial|logprob|penalt|cumsum|sort|radix", re.I)),
    ("memcpy", re.compile(r"Memcpy|Memset|memcpy|memset", re.I)),
]
def cat(name):
    for c, rx in CATS:
        if rx.search(name):
            return c
    return "other"

kern = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset", "Kernel")]
if not kern:
    kern = [e for e in events if e.get("ph") == "X" and "dur" in e and str(e.get("args", {}).get("stream", "")) != ""]
print(f"kerneli GPU: {len(kern)}")
by_cat = collections.Counter(); by_name = collections.Counter(); cnt_name = collections.Counter()
for e in kern:
    d = e.get("dur", 0.0); n = e.get("name", "")
    by_cat[cat(n)] += d; by_name[n] += d; cnt_name[n] += 1
total = sum(by_cat.values())
kern.sort(key=lambda e: e["ts"])
span = (kern[-1]["ts"] + kern[-1]["dur"] - kern[0]["ts"]) if kern else 0
# luki (GPU idle) - liczone na osi czasu po scaleniu wszystkich streamow
gaps = 0.0; big_gaps = 0; cur_end = kern[0]["ts"] if kern else 0
for e in kern:
    if e["ts"] > cur_end:
        g = e["ts"] - cur_end
        gaps += g
        if g > a.min_gap_us: big_gaps += 1
    cur_end = max(cur_end, e["ts"] + e["dur"])
print(f"okno sladu: {span/1000:.1f} ms | GPU zajete: {(span-gaps)/1000:.1f} ms ({(span-gaps)/span*100:.0f}%) | GPU idle (luki): {gaps/1000:.1f} ms, luk >{a.min_gap_us:.0f}us: {big_gaps}")
print("\n== czas per kategoria (suma po kernelach; overlap streamow moze dac >100% zajetosci) ==")
for c, d in by_cat.most_common():
    print(f"  {c:<12} {d/1000:8.1f} ms  {d/total*100:5.1f}%")
print(f"\n== top {a.top} kerneli ==")
for n, d in by_name.most_common(a.top):
    print(f"  {d/1000:8.1f} ms  n={cnt_name[n]:<6} avg={d/cnt_name[n]:7.1f} us  [{cat(n)}] {n[:110]}")
