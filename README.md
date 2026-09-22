# glm53-stack — GLM-5.3 (743B MoE, NVFP4) on 8× RTX PRO 6000 Blackwell with vLLM

Production serving stack for [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) quantized to ModelOpt NVFP4
(`incoai/GLM-5.3-NVFP4`: routed experts NVFP4, attention / shared experts / indexer / lm_head / MTP layer BF16) on a single
PCIe-only node with 8× NVIDIA RTX PRO 6000 Blackwell Workstation (96 GB, SM120, no NVLink, 2 NUMA nodes, 300 W power cap).

Measured on that hardware (details and methodology in [docs/RESULTS.md](docs/RESULTS.md)):

| metric | stock-ish baseline (TP8 DCP2 MTP5, same image family) | **this stack (TP4×PP2 DCP1 MTP3)** |
|---|---|---|
| single-stream decode (coding task) | 91–105 tok/s | **118–150 tok/s** (step 21 ms; MTP acceptance decides) |
| 4 / 16 / 32 concurrent streams | 222 / 530–590 / – tok/s | **300 / 660 / 1 050 tok/s** |
| cold prefill 32k / 131k / 262k / 650k | 3.5k / 3.3k / 3.0k / – tok/s | **7.9k / 7.3k / 6.3k / 3.4k tok/s** |
| KV cache pool @ `max-model-len 750k` | 1.43M tokens | **1.81M tokens** (NVFP4 KV, 352 B/token/layer) |
| new request TTFT while a 131k prefill runs | 39.5 s (starved) | **0.4–0.5 s**; other users keep 17–19 tok/s |
| restart | ~4.5 min | **2 min 20 s** (stop ≤6 s, start 140 s with warm compile cache) |
| quality | GSM8K (1319) 97.50% (same stack, online quant off) | **GSM8K (1319) 97.19%** (±0.45 pp SE; paired McNemar p = 0.45 — no measurable cost), teacher-forced CE ±0.01, needle 32k–650k OK, tool-call OK |

## What is in the image

`image/Dockerfile` layers, bottom to top (full rationale in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

1. **Upstream** `vllm/vllm-openai:nightly-2afa3f7e9` (vLLM 0.23.1rc1.dev925, 2026‑07‑08, torch 2.11 / CUDA 13.0). *This tag has been
   pruned from Docker Hub*; the pinned base is the team image `vllm-nightly-fi614:nvfp4` (see `image/patches/team/BASE-BUILD-STEPS.txt`).
2. **Team layer** (SM120 enablement, not in upstream): NVFP4 KV cache for sparse-MLA (FlashInfer + vLLM patches + `nvfp4_cache_ext.cu`),
   decode-context-parallel (DCP) for the SM120 sparse-MLA backend, baked JIT kernels.
3. **This repo**: FlashInfer 0.6.14 → 0.6.18.post1; MTP speculative decoding under pipeline parallelism with async scheduling
   (`patch_pp_mtp.py`); long-prefill chunking only when the GPU is shared (`patch_sched_fair.py`); online FP8 W8A8 (per-channel) for the
   BF16 layers the checkpoint leaves unquantized and online NVFP4 for the BF16 MTP draft experts (`patch_fp8_excluded.py`, `patch_b1s2.py`);
   B12X/CUTLASS MoE hybrid (present, disabled — negative result with MTP).

All patches are anchored text patches applied to the installed packages at build time; every anchor is asserted, so a base change
fails the build instead of silently drifting.

## Quick start

```bash
docker build -t glm53-stack:$(date +%Y.%m.%d) image/          # needs the pinned base image locally (see docs/RUNBOOK.md)
cp deploy/env.example deploy/env && $EDITOR deploy/env         # model path, GPU selection, port
CAND=1 deploy/serve-glm53-prod.sh                              # TP4×PP2 DCP1 MTP3, partition 41/37, online FP8+NVFP4, prefill threshold 512
deploy/verify-prod.sh                                          # health, startup log checks, smoke, parity (needle/math/tool), metrics
```

Rollback to the plain TP8/DCP2 configuration: `NCCL_MODE=p2p_sys MTP=5 DCP=2 IMAGE=<previous image> deploy/serve-glm53-prod.sh`.

## Repository layout

| path | content |
|---|---|
| `image/` | `Dockerfile`, `patches/team/` (SM120 KV-NVFP4 + DCP), `patches/ours/` (PP+MTP, fairness, online FP8/NVFP4, B12X hybrid), `tools/` (anchor checks, image fingerprint) |
| `deploy/` | `serve-glm53-prod.sh` (all knobs, `CAND=1` preset, `DRY=1`), `common.sh` (GPU selection by UUID), `verify-prod.sh`, `experimental/` |
| `bench/` | `run_cand.sh`/`serve_full.sh` (testbed launcher), `bench_decode.py`, `bench_prefill.py`, `bench_fairness.py`, `parity.py`, `lp_probe.py`, `needle_len.py`, `smoke.py`, `soak.py`+`soak.sh`, profiling helpers |
| `tools/` | micro-benchmarks and unit tests used during development (MoE, skinny GEMM, FP8 linear, NVFP4 quantizer round-trip, NCCL all-reduce) |
| `docs/` | `RUNBOOK.md`, `ARCHITECTURE.md`, `RESULTS.md`, `FAIRNESS.md` |
| `results/` | raw benchmark outputs (JSON/logs) behind every number quoted here |

## Known limitations

- Base image is a pruned nightly: rebuilding from public sources requires porting the patches to a current vLLM (see `docs/upstream/`).
- The MTP draft's NVFP4 activation scales are static proxies (last target layer × 2); acceptance on real traffic 2.8–3.0 (MTP3).
- Fairness is a policy knob (`--long-prefill-token-threshold`): 512 gives others 17–19 tok/s during a long prefill; smaller chunks trade prefill speed for responsiveness — see `docs/FAIRNESS.md`.
- Raw `/v1/completions` prompts above ~600k tokens degrade (model/format property, same on the baseline); chat format is fine up to the tested 650k.
- Custom all-reduce over PCIe (team patch) is disabled: wrong results for >2 GPUs without NVLink.

## License

Apache-2.0 (see `LICENSE`, `NOTICE`). vLLM and FlashInfer are Apache-2.0; the model checkpoint is not part of this repository.
