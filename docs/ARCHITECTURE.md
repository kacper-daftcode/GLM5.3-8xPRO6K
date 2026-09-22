# ARCHITECTURE — what is patched and why

Target: GLM-5.3 (`glm_moe_dsa`, 78 layers + MTP layer 78, 256 routed experts / top-8, hidden 6144, MLA with DeepSeek sparse attention
(indexer top-k 2048), 743B total / ~40B active) as `incoai/GLM-5.3-NVFP4` (ModelOpt NVFP4 routed experts; everything else BF16, incl. the MTP layer),
served on 8× RTX PRO 6000 Blackwell (SM120, 96 GB, PCIe 5.0, no NVLink, GPUs 0–3 on NUMA0 and 4–7 on NUMA1).

## 1. Why the stock stack does not work / is slow on this hardware

- Upstream vLLM's SM120 sparse-MLA backend (`FLASHINFER_MLA_SPARSE_SM120`) has KV only as `fp8_ds_mla` and no decode-context-parallel (DCP);
  with `max-model-len 750k` the KV pool would be ~0.6M tokens.
- All collectives go over PCIe (~40 GB/s bus bandwidth): TP8 all-reduce is 38–56 µs per call, ×2 per layer, and 8-GPU cross-NUMA collectives dominate.
- The checkpoint leaves ~221 MB/layer/GPU (@TP4) of BF16 weights outside the experts (attention projections, indexer, shared experts, lm_head, the
  whole MTP layer). Decode is memory-bandwidth bound: those bytes were ~30% of the step.
- A long prefill (100k–700k tokens) monopolizes the engine: with 2048-token chunks other users got 4.8 tok/s and a new request waited 39.5 s.

## 2. Team layer (`image/patches/team/`, image `vllm-nightly-fi614:nvfp4`)

| component | files | effect |
|---|---|---|
| **NVFP4 KV cache for sparse-MLA** — new FlashInfer `ModelType::GLM_NSA_NVFP4`, 352 B/token (packed FP4 KV + per-tile scales), decode/prefill kernels unpack in smem; write kernel `concat_and_cache_nvfp4_ds_mla` (`nvfp4_cache_ext.cu`, built to `/opt/nvfp4ext`); vLLM: `--kv-cache-dtype nvfp4` for the SM120 backend, `MLAAttentionSpec.real_page_size_bytes` | `patch_flashinfer.py` (25 anchors in `attention/sparse_mla_sm120/*`, `mla/_core.py`, `mla/_sparse_mla_sm120.py`), `patch_vllm.py` (`config/vllm.py`, `mla_attention.py`, `flashinfer_mla_sparse.py`, `flashinfer_mla_sparse_sm120.py`, `kv_cache_interface.py`), `vllm_nvfp4_cache.py` | KV pool ×1.9 vs FP8 at ~0.5% of step time |
| **DCP for SM120** — decode-context-parallel in `flashinfer_mla_sparse_sm120.py` (LSE return, DCP index filtering) | `dcp_sm120.patch` | pool ×DCP at +19–28% (DCP2) / +25–37% (DCP4) step time; superseded in the candidate by PP2 (DCP1) |
| custom all-reduce over PCIe (IPC one-shot/two-shot) | `patch_custom_ar_pcie.py` | **disabled** (`VLLM_CUSTOM_AR_PCIE_MAX_SIZE=0`): wrong results for >2 GPUs without NVLink, 4–40× slower with SysMem staging |

Base build steps are recorded verbatim in `BASE-BUILD-STEPS.txt` (the upstream nightly tag has been pruned; see RUNBOOK §1).

## 3. This repository's layer (`image/patches/ours/`)

### 3.1 FlashInfer 0.6.14 → 0.6.18.post1 (Dockerfile steps 1–6)
Python + cubin + AOT jit-cache in one version; the AOT `sparse_mla_sm120` module is deleted (AOT beats JIT, would shadow the team's patched sources);
JIT baked with `FLASHINFER_CUDA_ARCH_LIST=12.0f` into the `120f` directory the runtime actually looks up; `quack-kernels` 0.6.5 (0.5.0 breaks on
cutlass-dsl 4.7.1 and is imported by `dcp_indexer_cutedsl.py`). Net effect alone: quality identical, decode −3–4%, prefill −1–3%; mainly a base for the rest.

### 3.2 `patch_pp_mtp.py` — MTP speculative decoding under pipeline parallelism (async scheduling)
Motivation: TP4×PP2 keeps every all-reduce inside one NUMA node (14 µs instead of 38–56 µs), doubles prefill throughput and enlarges the KV pool,
but upstream refused MTP drafts under PP. Seven fixes (`vllm/config/speculative.py`, `v1/worker/gpu_model_runner.py`, `v1/worker/gpu_worker.py`,
`v1/spec_decode/llm_base_proposer.py`, `v1/attention/backends/mla/indexer.py`):
1. `SpeculativeConfig._verify_args`: the draft does not need `SupportsPP` (it runs whole on the last stage).
2. `hasattr(self, "drafter")` guards on non-last stages (6 places).
3. Async scheduling: the last stage broadcasts the speculation state **after** the drafter as int32 `[num_reqs, 2+k]` = `[next_token, valid_count, drafts…]`;
   stage 0 receives it in `sample_tokens` and sets `prev_sampled_token_ids`, `valid_sampled_token_count` (GPU + CPU copy) and `_draft_token_ids`.
4. `output_token_ids` aligned to the scheduler's `num_output_tokens` on **all** ranks (upstream only did it on the last one) — otherwise stage 0
   treated the request as chunked-prefill after a few steps, skipped the broadcast receive and deadlocked the GPUs (diagnosed with py-spy + `VLLM_TQ_PP_DEBUG=1`).
5. The draft on the last stage loads `model.embed_tokens.weight` from the checkpoint (`_tq_load_target_embed_tokens`): it cannot share the target's
   embeddings (stage 0) and the checkpoint has no copy under layer 78 → random embeddings gave acceptance 1.3.
6. `indexer.py`: `expanded_block_table` buffer sized to the block-table width (KV group block 128 vs kernel block 64 → 11719 ≠ 11720 → error with mixed decode lengths).
7. Sync scheduling + PP + speculation is rejected (drafts applied out of order → garbage). PP+MTP requires `--async-scheduling`.
Upstream is converging on the same path (vllm#46994, #53575 — items 1, 2, 5); items 3–4 are the candidates for upstreaming.

### 3.3 `patch_sched_fair.py` — long-prefill chunking only when shared
`--long-prefill-token-threshold N` is applied only when `running + waiting > 1` (`VLLM_LONG_PREFILL_THRESHOLD_SHARED_ONLY=1`, default). A lone long
prefill keeps full 2048-token chunks (7.3k tok/s @131k); when others are present it is cut to N (=512) tokens per step, so decoders get a token every
~170 ms and a new request gets its first token in 0.4–0.5 s. Numbers and the trade-off: `FAIRNESS.md`.

### 3.4 `patch_fp8_excluded.py` — online FP8 W8A8 for excluded linear layers
`ModelOptQuantConfigBase.get_quant_method` returns `TqOnlineFp8LinearMethod` for excluded `LinearBase`/`ParallelLMHead` layers whose prefix matches
`VLLM_TQ_FP8_LINEARS` (substring list). Weights load as BF16 and are quantized in `process_weights_after_loading` (`Fp8LinearMethod` in this vLLM does
not quantize online — it creates an FP8 weight and waits for a checkpoint scale → NaN), then the standard CUTLASS W8A8 path with per-token dynamic
activation scales. Applied to `fused_qkv_a_proj, q_b_proj, o_proj, shared_experts, indexer.wq_b, lm_head, eh_proj` (target and MTP draft; `kv_b_proj` is
absorbed into W_UK/W_UV at load, `indexer.wk/weights_proj` too small). Effect: decode step −13%, KV +8.6%, prefill +5%, CE unchanged. Marlin W8A16
(`VLLM_TQ_FP8_MARLIN=1`) is slower — do not use; `VLLM_TEST_FORCE_FP8_MARLIN` is global and would also move the MoE to Marlin.

### 3.5 `patch_b1s2.py` — MTP draft experts in online NVFP4, per-channel FP8, draft `eh_proj`
- Excluded `RoutedExperts` matching `VLLM_TQ_NVFP4_MOE` get `TqOnlineNvFp4MoEMethod(ModelOptNvFp4FusedMoE)`: BF16 weights load as usual; at
  `process_weights_after_loading` each expert is quantized into the **ModelOpt checkpoint layout** (FP4 e2m1 packed, even element in the low nibble;
  E4M3 block scales per 16 in linear layout; `weight_scale_2 = amax/(6·448)` shared by gate+up) and the unchanged base method does the swizzle/reorder
  and kernel setup (FLASHINFER_CUTLASS, same as the target). Verified bit-exact against a checkpoint expert round-trip and 99.9–100% byte-identical to
  `scaled_fp4_quant` (`tools/nvfp4_quant_unit.py`). Activation global scales are static: `input_scale` of the last target layer × `VLLM_TQ_NVFP4_MOE_AMAX_MARGIN`
  (2.0) or explicit `VLLM_TQ_NVFP4_MOE_ASCALE=a13,a2`. Draft quantization cannot change the output distribution (rejection sampling) — only acceptance.
  Effect: draft pass 341 → ~110 µs, step −1.5 ms, 3.3 GB/GPU freed on the last stage.
- `VLLM_TQ_FP8_CHANNEL=1`: `kFp8StaticChannelSym` weight key, scale `[N,1]` from `scaled_fp8_quant(..., use_per_token_if_dynamic=True)`, weight stored `(K,N)`.
- `eh_proj` (`nn.Linear` 6144×12288 BF16, ~100 µs/pass) becomes a `ReplicatedLinear` when its prefix matches `VLLM_TQ_FP8_LINEARS`, so the FP8 method applies.

### 3.6 `patch_b12x_hybrid.py` — B12X MoE with MTP and a B12X/CUTLASS hybrid (disabled)
Pure `--moe-backend flashinfer_b12x` allocates ~540 MiB workspace per layer (OOM at 75 layers) and rejects the BF16 draft MoE. The hybrid uses one shared
workspace and routes by row count (`VLLM_B12X_HYBRID_MAX_TOKENS`). Result: correct, but 6–10% slower at 1–4 streams with MTP (B12X only wins for ≤4 tokens
per step; every MTP stream contributes 4–6 tokens). Kept for experiments without speculation.

## 4. Configuration rationale (`CAND=1`)

| knob | value | why |
|---|---|---|
| TP4×PP2, DCP1 | stage 0 = NUMA0 GPUs, stage 1 = NUMA1 GPUs + draft | intra-NUMA all-reduce; prefill ×2.1; no DCP overhead; KV pool still +25% thanks to FP8/NVFP4 online |
| `VLLM_PP_LAYER_PARTITION=41,37` | stage 0 is the memory bottleneck | 42/36 → 1.65M, **41/37 → 1.81M**, 40/38 → 1.76M (stage 1 becomes the bottleneck) |
| MTP3 (was MTP5) | equal single-stream, +4–7% at 4–32 streams, KV +1% | with the cheap NVFP4 draft, draft tokens 4–5 (acceptance 0.32/0.26) no longer pay for the larger verification batch |
| `--long-prefill-token-threshold 512` + shared-only | policy | see `FAIRNESS.md` |
| `--max-num-batched-tokens 2048`, `--max-num-seqs 32`, GMU 0.90, `--async-scheduling`, NCCL `P2P_LEVEL=SYS` | as the previous production | measured earlier: P2P+SYS −30% all-reduce latency; 2048 avoids 5 s steps |
| 300 W power cap | site decision (PSU transients) | 450 W measured +0.5–2% prefill only |

## 5. Where the time goes (single stream, TP4×PP2, 25 ms/step with MTP5 before B1.2: 30.6 ms)
Stage 0 (~13 ms): dense GEMMs 4.9 ms (now FP8), MoE NVFP4 4.6 ms (42 × 110 µs), NCCL 1.2 ms, attention/indexer 1.3 ms. Stage 1 (~15.5 → ~12 ms): 36 layers
+ draft passes (5 × ~1 ms after NVFP4/FP8, was 5 ms) + lm_head/sampler 0.5 ms. ~2 ms PP hand-off; stages do not overlap at 1 stream. Prefill step with a
512-token chunk (85 ms): MoE 25 ms (kernel floor: 16 rows/expert), TP4 all-reduce 22 ms (PCIe), attention 18 ms, GEMM ~8 ms.
