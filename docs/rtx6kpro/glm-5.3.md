# GLM-5.3 (743B MoE, NVFP4) on 8× RTX PRO 6000 — TP4×PP2 / DCP1 / MTP3

**Status: production-qualified on one node** (8× RTX PRO 6000 Blackwell Workstation 96 GB, PCIe 5.0, no NVLink, two NUMA nodes,
300 W power cap). Serving the full GLM-5.3 (not Flash) with a 750k-token context, MTP3 speculative decoding and an NVFP4 KV cache.
Draft for `local-inference-lab/rtx6kpro/models/glm-5.3.md`; recipe, patches, scripts and raw measurements live in
**`https://github.com/kacper-daftcode/GLM5.3-8xPRO6K`** (Apache-2.0). Numbers below are from that repository's `docs/RESULTS.md`.

## Release identity

| Item | Value |
|---|---|
| Image | `<REGISTRY>/glm53-stack:2026.09.19` (rebuilt from `image/Dockerfile`; fingerprint-identical to the production image `b7c70850807d`) |
| Upstream base | `vllm/vllm-openai:nightly-2afa3f7e9` — vLLM 0.23.1rc1.dev925 (2026‑07‑08), torch 2.11.0+cu130, Triton 3.6.0, transformers 5.13.0. The tag has been pruned from Docker Hub; the pinned base is archived (see RUNBOOK §1) |
| Added kernels / libraries | FlashInfer 0.6.18.post1 (+ `flashinfer-jit-cache`, JIT sparse-MLA for `120f`), `nvidia-cutlass-dsl` 4.7.1, `quack-kernels` 0.6.5 |
| SM120 enablement layer | NVFP4 KV cache for sparse-MLA (FlashInfer + vLLM patches + `nvfp4_cache_ext`), decode-context-parallel for the SM120 sparse-MLA backend (not in upstream vLLM) |
| Stack patches | MTP under pipeline parallelism with async scheduling, shared-only long-prefill chunking, online FP8 W8A8 (per-channel) for the BF16 layers, online NVFP4 for the BF16 MTP draft experts — all anchored text patches, asserted at build time |
| Checkpoint | `incoai/GLM-5.3-NVFP4` (ModelOpt NVFP4 routed experts; attention, shared experts, indexer, `lm_head` and the MTP layer in BF16) |
| Reproducibility gate | `image/tools/image_fingerprint.sh` (package versions, sha256 of the 16 patched vLLM files, FlashInfer sparse-MLA sources, extension/JIT artifacts) + `bench/run_all.sh` |

## Start the server

```bash
git clone https://github.com/kacper-daftcode/GLM5.3-8xPRO6K glm53-stack && cd glm53-stack
cp deploy/env.example deploy/env && $EDITOR deploy/env      # MODEL_DIR (checkpoint), CACHE_ROOT, IMAGE
CAND=1 deploy/serve-glm53-prod.sh                            # stop old → start → wait for /health (~2.5 min)
deploy/verify-prod.sh                                        # health, startup log, smoke, greedy parity, metrics
```

`CAND=1` expands to: `--tensor-parallel-size 4 --pipeline-parallel-size 2 --decode-context-parallel-size 1`,
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`, `--async-scheduling`, `--kv-cache-dtype nvfp4`,
`--quantization modelopt_fp4`, `--max-model-len 750000 --max-num-seqs 32 --max-num-batched-tokens 2048`,
`--long-prefill-token-threshold 512`, `VLLM_PP_LAYER_PARTITION=41,37`,
`VLLM_TQ_FP8_LINEARS=fused_qkv_a_proj,q_b_proj,o_proj,shared_experts,indexer.wq_b,lm_head,eh_proj VLLM_TQ_FP8_CHANNEL=1`,
`VLLM_TQ_NVFP4_MOE=layers.78.mlp.experts`, NCCL P2P over PCIe (`NCCL_P2P_LEVEL=SYS`). The API is on port 8000, model name `glm-5.3`.

## Serving defaults

| Setting | Deployment default |
|---|---|
| Parallelism | TP4 × PP2 (stage 0 = 41 layers on NUMA 0, stage 1 = 37 layers + MTP head on NUMA 1), DCP1 |
| Speculation | MTP3 (`num_speculative_tokens` 3); MTP5 measured equal at 1 stream, −4…−7% at 4–32 streams |
| Target precision | ModelOpt NVFP4 routed experts (FlashInfer CUTLASS MoE); BF16 layers of the checkpoint quantized **online** to FP8 W8A8 per-channel |
| MTP draft | Draft experts quantized online to NVFP4 (ModelOpt layout, static activation scales from the last target layer ×2); `eh_proj` FP8 |
| KV cache | NVFP4 for sparse-MLA (352 B/token/layer): **1 807 193 tokens** at `max-model-len 750000` |
| Scheduler | 2048 batched tokens, 32 sequences, long-prefill chunk 512 **only while the engine is shared** (a lone long prefill runs full chunks) |
| Context / GPU fraction | 750 000 tokens / 0.90 |
| Reasoning / tools | `--reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice` |
| Power | 300 W per GPU (450 W measured: +0.5–2% prefill only) |
| Restart | 2 min 17 s (stop 6 s, start 140 s with a warmed compile cache; first start with a fresh cache +25 s, second start recompiles once) |

## Measured performance

Same node, 300 W (production) / 450 W (testbed; decode identical within noise). Decode: `bench/bench_decode.py`, 256 output tokens per
request, ms/step from the engine's draft counters (independent of acceptance) plus delivered tok/s; the first round after a concurrency change
is discarded. Prefill: cold, unique prompts, time to first token. Baseline = the same image family run as TP8 / DCP2 / MTP5 without the
stack patches (the previous production configuration).

| Metric | Baseline TP8 DCP2 MTP5 | **TP4×PP2 DCP1 MTP3 (this page)** |
|---|---:|---:|
| C1 decode | 36 ms/step, 91–105 tok/s | **21.0–21.2 ms/step, 118–150 tok/s** (acceptance 2.9–3.0 tok/step) |
| C4 | 54–62 ms/step, 199–225 tok/s | **38 ms/step, 300–311 tok/s** |
| C16 | 88–93 ms/step, 530–590 tok/s | **70–71 ms/step, 654–666 tok/s** |
| C32 | – | **89 ms/step, 1 041–1 056 tok/s** |
| Prefill 32k / 131k / 262k | 3.5k / 3.3k / 3.0k tok/s | **7.9k / 7.3k / 6.3k tok/s** |
| Prefill 650k (solo) | – | **3.4k tok/s** (190 s) |
| KV pool @750k | 1 427 943 | **1 807 193** |
| New request TTFT during a 131k prefill | 39.5 s | **0.4–0.5 s**; other users 17–19 tok/s (ITL 170 ms) |

For scale, rtx6kpro reports 1 188 tok/s at C32 for GLM-5.2 v20 (B12X + MTP3); this stack measures 1 050 tok/s at C32 on the full GLM-5.3
with a 2.4× larger KV pool and the fairness policy above. Numbers are not directly comparable (different model, image and scheduler settings).

## Quality

| Check | Result |
|---|---|
| GSM8K, full test set (1319), greedy, chat with reasoning, `max_tokens` 2048 | **97.19%**; the same stack with online quantization off (BF16 attention/shared/head/draft): 97.50% — paired McNemar p = 0.45 (10 vs 6 discordant), i.e. no measurable cost of the online FP8/NVFP4 |
| FP8 ablation (1319) | without `lm_head` in FP8 97.50% (p = 0.50), without `lm_head`+`o_proj` 97.19% (p = 1.0) — kept in FP8 (−0.6 ms and −1.1 ms per step, +47k KV tokens) |
| Draft quantization | NVFP4 vs BF16 draft experts: acceptance 3.44 vs 3.45 tok/step on GSM8K, 2.89 vs 2.90 on benchmark prompts, 2.86 vs 2.85 on production probe traffic — lossless by construction (rejection sampling), no measurable acceptance cost |
| Teacher-forced CE (500–700-token texts) | pl 0.250–0.256 / code 0.280–0.290 / mixed 0.245–0.246 nats/token vs 0.260 / 0.290 / 0.253 for the baseline (run-to-run ±0.01) |
| Long context | chat-format needle 131k / 262k / 500k / 650k: 50/51 during a 195-minute mixed soak, 5/5 at 650k in isolation and under 12–25 streams |
| Soak (195 min, 12 chat + 2 tool + 1 long-context stream + bursts to 32 running) | ~5 800 requests, 0 HTTP errors, 0 empty responses, tool called 100%, 0 Xid |

## Known limitations

- **Raw `/v1/completions` prompts ≥ ~600k tokens** can degenerate (the model sometimes "continues the instructions" instead of answering); this happens on the
  baseline TP8 configuration too and is a model/format property — long-context checks use the chat format.
- **Fairness policy is a trade-off**: while a long prefill runs, other users get 17–19 tok/s (chunk 512) — a 700k prefill under load keeps them there for
  minutes. A dynamic policy (smaller chunk when many users decode) is measured in the repository's `docs/FAIRNESS.md`.
- **Static draft activation scales** (from the checkpoint's last target layer ×2) — validated on the workloads above only.
- **Text patches on a pruned nightly**: the stack pins a vLLM nightly from 2026‑07‑08 whose upstream tag no longer exists; the base image is archived, and a
  port of the generic patches to current vLLM is in progress (PP+MTP acceptance under pipeline parallelism, shared-only long-prefill threshold).
- `prompt_logprobs` for prompts shorter than ~450 tokens returns garbage on this vLLM build (pre-existing upstream bug; the CE probe skips short texts).

## Required source changes

| Responsibility | Where |
|---|---|
| NVFP4 KV cache for sparse-MLA (FlashInfer kernels, vLLM cache plumbing, `nvfp4_cache_ext.cu`) | `image/patches/team/` (SM120 enablement layer) |
| Decode-context-parallel for the SM120 sparse-MLA backend | `image/patches/team/dcp_sm120.patch` |
| MTP speculative decoding under PP with async scheduling (7 fixes: draft embeddings/head under PP, spec-state broadcast, output alignment) | `image/patches/ours/patch_pp_mtp.py` → upstream PR (in preparation) |
| Long-prefill threshold applied only while shared (+ optional dynamic chunk by request count) | `image/patches/ours/patch_sched_fair.py` → upstream PR (in preparation) |
| Online FP8 W8A8 / NVFP4 for layers excluded from the ModelOpt checkpoint | `image/patches/ours/patch_fp8_excluded.py`, `patch_b1s2.py` → upstream RFC |

The image contains no model weights and no compile caches; the checkpoint is mounted at `/model`.
