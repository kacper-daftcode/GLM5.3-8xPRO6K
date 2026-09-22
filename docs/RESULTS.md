# RESULTS — measurements (raw data in `results/`)

Hardware: 8× RTX PRO 6000 Blackwell Workstation 96 GB (SM120), PCIe 5.0, no NVLink, 2× EPYC, 2 NUMA nodes. Testbed B at 450 W unless noted (300 W for the soak),
production A at 300 W. Model `incoai/GLM-5.3-NVFP4`, `max-model-len 750000`, `max-num-seqs 32`, `max-num-batched-tokens 2048`, KV `nvfp4`, GMU 0.90.
Scripts: `bench/`. Decode numbers are ms/step from `/metrics` draft counters (independent of MTP acceptance) plus tok/s; ignore the first round after a
concurrency change (warm-up).

## 1. Configurations compared

| id | image | parallelism | spec | quant online | prefill threshold |
|---|---|---|---|---|---|
| P0 | `nvfp4` (team) | TP8 DCP2 | MTP5 | – | none (chunk 2048) |
| P1 | `nvfp4-fi618` | TP8 DCP2 | MTP5 | – | 512 shared-only |
| C1 | `nvfp4-fi618` | TP4×PP2 DCP1, 42/36 | MTP5 | FP8 per-tensor (6 layer types) | 512 |
| C2 | `nvfp4-fi618` | TP4×PP2 DCP1, 41/37 | MTP5 | FP8 per-channel (+eh_proj), draft experts NVFP4 | 512 |
| **C3 (prod)** | `nvfp4-fi618` = `glm53-stack:2026.09.19-rc1` | TP4×PP2 DCP1, 41/37 | **MTP3** | as C2 | 512 |

## 2. Decode

| conc | P0 ms/step (tok/s) | C1 | C2 | **C3** |
|---|---|---|---|---|
| 1 | 36 (105) | 26.3–26.8 (141) | 25.0–25.5 (140–148) | **21.0–21.2 (142–150)** |
| 4 | 54–62 (199–225) | 49–60 (227–263) | 45–50 (275–298) | **38 (300–311)** |
| 16 | 88–93 (530–590) | 84–88 (592–635) | 79–83 (623–665) | **70–71 (654–666)** |
| 32 | – | – | 108 (946–989) | **89 (1 041–1 056)** |

MTP acceptance (tokens/step incl. the sampled one): MTP5 3.3–3.7 on benchmark prompts, 3.0–3.2 on real/mixed traffic; MTP3 2.9–3.0 / 2.8.
Coding tasks on production (single stream, 2500 output tokens, MTP5): Python 140 tok/s (acc 3.55), TypeScript 118 (3.02), Go 133 (3.36); TTFT 76–88 ms.

## 3. Prefill (cold, unique prompts, tok/s)

| context | P0 | C1 | C2/C3 |
|---|---|---|---|
| 32k | 3.5k | 7.8k | **7.9k** |
| 131k | 3.3k | 7.2k | **7.3k** |
| 262k | 3.0k | 6.2k | **6.3k** |
| 650k (solo) | – | – | **3.4k** (190 s) |

## 4. Fairness (`bench_fairness.py`: 131k prefill + one decoding + one new request)

| config | shared prefill tok/s | new request TTFT | other users during prefill |
|---|---|---|---|
| P0 (no threshold) | 3 071 | **39.5 s** | 4.8 tok/s (ITL 600 ms) |
| P1 (512 shared-only) | 2 761 | 0.56 s | 14–18 tok/s |
| C1 | 4 018 | 0.43 s | 17–19 tok/s (ITL 170 ms) |
| C2 | 4 051 | 0.48 s | 17–19 tok/s |

## 5. KV pool (`GPU KV cache size`, max-model-len 750k)

P0 1 427 943 · P1 1 427 943 · C1 1 653 148 · C2 1 790 617 · **C3 1 807 193 (A) / 1 813 657 (B)**. Partition sweep at C2: 43/35 → 1 502 879, 42/36 → 1 651 868, 41/37 → 1 790 617, 40/38 → 1 761 306.

## 6. Quality

- Teacher-forced CE (`lp_probe.py`, texts 507–683 tokens, nats/token), baseline = P0: pl 0.260 / code 0.290 / mix 0.253. C1: 0.246 / 0.279 / 0.265;
  C2: 0.250–0.255 / 0.287–0.289 / 0.243–0.250; C3: 0.250–0.256 / 0.280–0.290 / 0.245–0.246. Run-to-run spread ≈ ±0.01; top-1 agreement with P0 97.4–99.0%.
- Greedy battery (`parity.py`): needle 32k/131k, sum of primes < 100 (1060), tool call — OK on every configuration. Texts diverge after 1–700 characters
  between configurations (expected for FP4 + different kernels/batching); with MTP they can also diverge between two runs of the same configuration
  (non-deterministic reductions flip near-ties) — raw `/v1/completions` needle prompts ≥131k are therefore checked in chat format.
- Long context: chat-format needle 131k/262k/500k/650k: 50/51 during the soak (the miss was a reasoning cut by `max_tokens`), 5/5 at 650k in isolation
  and under 12–25 concurrent streams. Raw-format prompts at 650k degenerate on C2 **and** on P1 alike (model/format property).
- Draft quantization (C2) is lossless by construction (rejection sampling); acceptance unchanged on benchmark prompts.

## 6a. GSM8K — full test set (1319 questions, greedy, chat with reasoning, `max_tokens` 2048, `bench/gsm8k_eval.py --conc 8`; B at 450 W,
`glm53-stack:2026.09.19-rc1`, TP4×PP2 41/37 MTP3, threshold 512; 2026‑09‑19, raw data `results/gsm8k_rc1_mtp3_*_full.json`, `results/d10/`)

Ablation of the online quantization (`FP8_LINEARS` knob of `bench/run_cand.sh`; paired comparison with `bench/gsm8k_compare.py`, exact McNemar):

| `FP8_LINEARS` (online FP8 per-channel W8A8) | draft experts | accuracy | truncated @2048 | KV pool | decode @1 ms/step | vs C3 (paired) |
|---|---|---|---|---|---|---|
| **C3 (prod)**: `fused_qkv_a_proj,q_b_proj,o_proj,shared_experts,indexer.wq_b,lm_head,eh_proj` | NVFP4 | **97.19%** (1282) | 12 | 1 813 657 | **21.0–21.2** | – |
| C3 without `lm_head` | NVFP4 | 97.50% (1286) | 16 | 1 813 657 | 21.7–21.9 | +0.31 pp; 12 vs 8 discordant, p = 0.50 |
| C3 without `lm_head` and `o_proj` | NVFP4 | 97.19% (1282) | 14 | 1 766 554 | 22.8–23.0 | ±0.00 pp; 7 vs 7, p = 1.00 |
| none (BF16 attention/shared/indexer/lm_head; BF16 draft) | BF16 | 97.50% (1286) | 14 | 1 481 952 | 25.9 | +0.31 pp; 10 vs 6, p = 0.45 |

- SE at n = 1319 is ±0.45 pp; every difference is ≤0.31 pp with p ≥ 0.45. Predictions agree on 1294–1300 of 1319 questions between any two
  configurations; 25 questions are wrong in all four, 53 in at least one; ~9 of the misses per configuration are reasoning cut by `max_tokens` 2048.
- Run-to-run noise of the *same* configuration is of the same size: the first 500 questions of the full C3 run scored 490/500 vs 486/500 in the
  earlier 500-question run (4 discordant, all one way); the BF16 reference scored 489/500 vs 491/500. The 1.0 pp gap reported from the
  500-question runs (97.2% vs 98.2%) was therefore noise, not a quantization effect.
- Conclusion (plan D10): **no measurable accuracy cost of FP8 W8A8 on `lm_head` or `o_proj`** at this sample size (95% CI of the C3 vs BF16 delta
  ≈ ±0.6 pp). `lm_head` in FP8 is worth 0.6–0.7 ms/step (3%) at 1 stream, `o_proj` another ~1.1 ms/step and 47k KV tokens; production stays on C3.
- For scale: SGLang's GLM-5.3-Flash cookbook reports 97.35% (1319) for the Flash model on GB300.
- Side observation: the second start with a compile cache created by a single run recompiles (`Compiling model again … Kernel index 1 not found`,
  16–23 s) and re-saves the AOT artifacts; the third start loads them in ~4 s (see RUNBOOK §2).

## 6b. Draft quantization — NVFP4 vs BF16 draft experts (plan D5, 2026‑09‑19)

Paired on B (`rc1`, C3 vs C3 with `NVFP4_MOE=`; same prompts; acceptance from `/metrics` deltas around each workload; raw `results/d5/`):

| workload | NVFP4 draft (C3) | BF16 draft | Δ |
|---|---|---|---|
| GSM8K‑500 greedy chat, conc 8 — acceptance (tok/step) | **3.44** (109 778 / 45 003 drafts) | 3.45 (112 145 / 45 704) | +0.015 |
| GSM8K‑500 — accuracy / throughput | 97.6% / 479 tok/s | 97.6% / 459 tok/s | NVFP4 +4.5% tok/s |
| `bench_decode` @1 — acceptance / ms per step | 2.89 / 21.1–21.2 | 2.90 / 21.8 | +0.02 / NVFP4 −0.6 ms |
| KV pool (41/37) | **1 813 657** | 1 673 628 | NVFP4 +8.4% (BF16 draft makes the last stage the memory bottleneck) |

Production A/B (A, BF16 draft `CAND=1 NVFP4_MOE=` 08:58–09:48 UTC in a Saturday maintenance window vs the preceding 10 h of C3; engine-log
1-minute windows, `results/d5/prod_ab_acceptance.txt`): on the gateway's 1/min probe request (identical every minute, so a paired sample) BF16 2.85
(54 min) vs NVFP4 2.86 (611 min); per-minute means 2.95 ± 0.08 vs 2.97 ± 0.02. Minutes with real requests were too few to compare (4 vs 4).
Conclusion: the online NVFP4 draft costs **no measurable acceptance** (≤0.02 tok/step on a reasoning workload, on benchmark prompts and on the
production probe) while saving 0.6 ms/step and 140k KV tokens — it stays in production; activation-scale calibration (D5 follow-up) is not needed.

## 7. Soak (B, 300 W, C2, 195 min in 3 segments with `docker stop -t 60` restarts between them)

12 chat streams (essays/code/QA/multi-turn, T 0–1.0) + 2 tool-call streams + 1 long-context stream (131k–650k) + a burst of 28 short requests every 15 min
(→ 32 running = `max-num-seqs`): ~2 700 chat, ~2 800 tool, 51 long, 280 burst requests — **0 HTTP errors, 0 empty responses, tool called 100%, needle 50/51,
0 Xid/NVRM lines, container RSS 30.4 → 31.3 GiB per segment (reset on restart), acceptance 3.16–3.18 (MTP5)**. Restarts: stop 1–2 s, start 130 s.
Latencies under that load: chat p50 40 s / p95 135 s (up to 1500 tokens), tool p50 5 s / p95 14 s, long 131k 60 s, 262k 130 s, 500k 290 s, 650k 410 s.
During a 650k prefill with 25 streams, other streams dropped to ~4 tok/s for ~12 min (policy, see FAIRNESS.md).

## 8. Production (A, 300 W)

Cut-over 2026‑09‑18 20:25 UTC: downtime 2 min 18 s (stop 6 s, start 140 s, compile 4 s with the cache copied from B). KV 1 786 329 (C2) → 1 807 193 (C3 at 22:53 UTC).
`bench_decode.py` on A: C2 24.8–25.5 ms @1, 47–50 @4; C3 20.9–21.0 @1, 38–42 @4. After 2.4 h of real traffic on C2: 313 requests, 0 errors, 0 preemptions,
acceptance 3.02 (previous production TP8 DCP2 MTP5 long-run: 3.30 — different traffic window; to be A/B-tested against a BF16 draft).

C3 after 8.5 h (2026‑09‑19 07:20 UTC): 566 requests, 0 errors, 0 preemptions, `waiting` 0, dmesg clean, cumulative MTP3 acceptance 2.93. Caveat for any
acceptance comparison: **~505 of those requests are a 1/min health probe** arriving through the gateway (all traffic comes from the gateway address, so
requests cannot be attributed to end users; `deploy/acceptance_by_hour.py` on the engine log: real traffic only in the 22h UTC hour — acceptance 3.02 —
and the 05h hour — 2.76; probe-only hours 2.8–2.9). A/B tests of the draft (D5) therefore compare 1-minute windows: probe-only (≤60 drafted tokens) vs busy.

Maintenance window 2026‑09‑19 (Saturday, traffic ≈ probe only): 08:52 UTC switch to the BF16 draft (fresh compile cache: stop 3 s, start 220 s,
compile 41.8 s; KV 1 671 132; verify OK) — 09:49 UTC back to C3: **stop 6 s + start 140 s = 2 min 17 s downtime, AOT artifacts loaded directly
(`torch.compile took 4.21 s`, init engine 41 s instead of 80 s)** — the compile-cache lifecycle from RUNBOOK §2 confirmed on A. KV 1 807 193, verify 8/8, 0 errors.

**Incident 2026‑09‑19 12:59 UTC** (unrelated to any configuration change; the engine had been idle apart from the 1/min probe for 3 h): workers stopped answering
(`shm_broadcast` timeouts from 12:59), `EngineDeadError` at 13:03:05; at teardown the driver logged ~11 500 `NVRM: GPU1 … Possible bad register read 0xbadf3200` /
`NV_ERR_GPU_IN_FULLCHIP_RESET` lines, GPU 0 (bus 01:00.0) went to "GPU requires reset", `nvidia-smi --gpu-reset` returned "Not Supported", and the vLLM process
tree stayed zombie holding 93–95 GB on every PRO 6000. Host reboot was the only way out; the graceful reboot hung after "Sending SIGTERM to remaining processes"
until a manual power cycle on 2026‑09‑20 15:53 UTC → **production outage ≈ 27 h**. Driver 595.71.05 (the testbed runs 610.43.02 without such incidents — D7).

**2026‑09‑20 16:00 UTC — rc2 + D3 in production**: `CAND=1 IMAGE=glm53-stack:2026.09.19-rc2 LONG_PREFILL_DYNAMIC=4:128` (now the `CAND=1` default). Cold start
after reboot 260 s (weights 137 s from a cold page cache; `torch.compile` 4.2 s from the AOT cache), KV 1 807 193, verify: single 149 tok/s, 272 tok/s @4,
parity 8/8, acceptance 2.94, 0 errors. First production use of the dynamic chunk (`docs/FAIRNESS.md`).

**2026‑09‑22 13:06 UTC — D2 RAM tier in production**: `CAND=1 KV_OFFLOAD_GB=668` (now part of the `CAND=1` default; §12d for the gate). Deployed during daytime
traffic (~13 req/min) at a `Running: 0` moment: stop 13:02:15 → READY 13:06:17 = **4 min 02 s downtime** (weights 18 s from the page cache, one-off `torch.compile`
recompile 39 s after the config change, pinning **720 GiB** ≈ 60 s). `Shmem` 720 GB, `MemAvailable` 1425 → 696 GB, KV pool unchanged 1 813 657, `Allocating 53/48
CPU tensors` per stage. Verify: parity 8/8 (needle 32k/131k, tool call), 0 errors, dmesg clean; 57 GB stored in the first 5 min of traffic, 0 allocation failures,
0 preemptions. The smoke numbers of that run (single stream 11.8 tok/s, 66.7 tok/s @4) are **not** a regression: a user's ~600k-token prefill was in flight for the
whole smoke, and in that state this production has always given all decoders together a median of 14.3 tok/s (p25 9.4) at a shared prefill rate of ~1.8k tok/s
(182 such 10 s windows in the 40 h log before the change; after the change 10.8 tok/s / 1.45k tok/s on 13 windows of one episode — inside the old distribution).
With three concurrent short requests right after: ~70 tok/s per stream (≈210 aggregate, 14–15 ms/step), the usual figure. What to watch: `vllm:kv_offload_load_bytes_total`
and `external_prefix_cache_hits_total` once the GPU pool has cycled (first RAM hits), `kv_offload_allocation_failure`, `MemAvailable` (should stay ≈ 690 GB).

## 9. Reproducibility gate (2026‑09‑19)

`glm53-stack:2026.09.19-rc1` rebuilt from `image/Dockerfile` on host A: `image/tools/image_fingerprint.sh` identical to the production image
(`b7c70850807d`: package versions, sha256 of all 16 patched vLLM files, FlashInfer sparse-MLA sources, extension/JIT artifacts). On B: KV 1 813 657,
decode 21.0–21.2 ms @1 / 69.7 ms @16 (MTP3), parity OK, CE 0.250 / 0.290 / 0.246.

## 10. Current upstream vLLM (`vllm/vllm-openai:nightly` 2026‑09‑19, 0.29.1rc1, V2 runner) on the same node

Unpatched upstream, `incoai/GLM-5.3-NVFP4`, `--kv-cache-dtype fp8_ds_mla` (upstream has the SM120 sparse-MLA backend but no NVFP4 KV and no DCP for it),
`max-model-len 131072`, MTP3, async scheduling; two workarounds were required to start at all (`docs/upstream/ISSUE-pp-mtp-acceptance.md`: draft head
quantization, FlashInfer autotune stall). `bench/serve_upstream.sh`, raw data `results/p41/`.

| configuration | ms/step @1 | tok/s @1 | acceptance @1 | GSM8K‑500 (conc 8): accuracy / truncated / mean tokens / acceptance |
|---|---:|---:|---:|---|
| upstream TP8 | 25.1 | 124–128 | 3.13–3.21 | 97.6% / 5 / 298 / 3.49 |
| upstream TP4×PP2 async | 26.8–27.1 | 107–110 | 2.91–2.93 | **96.2% / 15 / 382 / 3.40** — longer, more often non-terminating outputs |
| **this stack (C3)**, TP4×PP2 DCP1 async, NVFP4 KV | **21.0–21.2** | **142–150** | 2.9–3.1 | 97.6% / 6 / 310 / 3.44 (D5 run, same prompts) |

Upstream TP8 MTP3 at 25.1 ms/step is a large improvement over the July nightly (TP8 DCP2 MTP5: 36 ms), but the PP2 path is broken for spec decode
and the KV pool is limited to FP8 (898k tokens at 131k max-len on PP2 vs 1.81M at 750k here).

## 11. Closed negative results (do not retry without new evidence)

B12X MoE with MTP (−6–10% @1–4; only wins for ≤4 tokens/step); Marlin for NVFP4 MoE (−3.5% decode, −10% prefill) and FP8 W8A16 (−5%); DFlash2 draft
(acceptance 2.7–4.0 ≈ MTP5, port = days); NCCL proto/algo/channel tuning (all-reduce is at the PCIe limit); Triton skinny GEMM (cuBLAS already at bandwidth);
custom all-reduce over PCIe; `"quantization":"fp8"` in `--speculative-config` (ignored for MTP); 450 W (+0.5–2% prefill); partition 43/35 and 40/38.

## 12. KV offload to host RAM (plan D2, native vLLM `OffloadingConnector`, 2026‑09‑21/22) — **gate PASSED and deployed on A 2026‑09‑22 (§12d, §8); the 2026‑09‑21 "hangs" were a watchdog artefact (§12c)**

> **Read §12c first.** Everything below that says "hang" (the three 2026‑09‑21 runs, E1–E33) was re-analysed on 2026‑09‑22 against the raw logs and traces:
> no software hang ever occurred. The engine's status logger is silent during a single long chunked prefill, the 60 s watchdog fired inside every 550k
> prefill while the GPUs were computing at the 300 W cap, and `docker kill` in the middle of live multi-GPU work left GPUs wedged, which was mistaken for
> confirmation. The two Xid 79 events both happened at 8×450 W, 20 min before the office breakers tripped under the same load. The tables are kept as a
> record of what was run; their "hang" verdicts are void. The gate itself (200k and 650k reload from RAM with the production configuration) passed on 2026‑09‑22.

Decision: native in-tree connector rather than LMCache (no extra dependency on a patched nightly, layout-agnostic raw block copies, preemption support;
LMCache would add restart-surviving cache, pin/quota APIs and non-prefix reuse, none of which is needed on a single node with 0.75–1.4 TB RAM). Code review
of the image (`vllm 0.23.1rc1.dev925`): the connector views each layer's KV as `(num_blocks, page_size_bytes)` bytes with stride `page_size_bytes`; the
team's NVFP4 cache is exactly `(num_blocks, 64, 352)` bytes with `real_page_size_bytes = 64·352`, so the DMA copies are layout-correct; the cross-layer
uniform layout is not triggered (`indexes_kv_by_block_stride=False`), the `HND` request is overridden by the MLA backend's own layout, async scheduling,
MTP (trailing block of draft groups excluded) and PP (connector output passed through `sample_tokens` on non-last ranks; scheduler uses rank 0's
`kv_cache_config`, the larger stage) are all handled. Knob: `KV_OFFLOAD_GB` in `bench/run_cand.sh` and `deploy/serve-glm53-prod.sh` (raw data `results/d2/`,
container logs `llm-tests/logs/B-full-glm*`).

What worked (B, rc2 TP4×PP2 DCP1 MTP3):

| item | result |
|---|---|
| host RAM footprint | torch pins **one tensor per layer and rounds each up to a power of two**: request 400 GiB → **~700 GB pinned**, swap (8 GB) exhausted, OOM-killer hit a user `systemd`; request **334 GiB → 360 GiB pinned** (stage‑0 workers 41×1 GiB + 12×0.5 GiB = 47.1 GiB, stage‑1 38×1 + 10×0.5 = 43.1 GiB). Sweet spots for this config: 334 GiB (2.8M tokens, 1.54× GPU pool) and 668 GiB (5.6M tokens, ~720 GiB) |
| bytes stored per token | **120.0 KiB** (= Σ over the 8 workers of 352 B×MLA layers + 132 B×indexer layers; KV is replicated across the 4 TP ranks with DCP1) → 200k-token prompt = 22.85 GiB in RAM |
| startup | +2–3 min for pinning 360 GiB, KV pool unchanged (1,813,657) |
| stores during prefill | 200k prompt: TTFT 29.2 s = 6.85k tok/s at 450 W (no visible store overhead vs 7.3k cold), needle OK, parity 8/8, GPU prefix-cache hit 0.41 s |
| determinism control | the same 200k prompt as a GPU prefix-cache hit gives a *different* greedy text than the cold run (recomputed tail uses different kernels) — byte identity is not a valid gate on this stack; `bench/kv_offload_reload.py` uses needle + `vllm:kv_offload_load_bytes` instead |

The three 2026‑09‑21 runs with the 334 GiB pool, as originally reported (struck-through claims corrected on 2026‑09‑22 from `results/d2/reload_200k*.log` and the container logs):

| run | power | what happened (corrected) |
|---|---|---|
| 1, 12:15 UTC, 334 GiB | 450 W | parity, cold 200k, GPU hit, churn 1 (549k) OK; during churn 2 (`pciehp Slot(17): Link Down`) **Xid 79 "GPU has fallen off the bus"** on PRO 6000 `0000:01:00.0` (index 0, stage 0) → `CUDA error: unspecified launch failure`, `uvm global fatal error 0x60`, host reboot required. No stall preceded it |
| 2, 12:52 UTC, 334 GiB | 450 W | ~~the first (cold 200k) request stalled ~50 s in~~ **cold 200k completed in 29.2 s** (needle OK), GPU hit 0.42 s, **churn 1 completed normally in 112.2 s (4.9k tok/s)**, churn 2 was progressing (status lines 12:57:13 and 12:57:23) when **Xid 79** hit a *different* card, `0000:81:00.0` (index 4, stage 1, slot 65) at 12:57:29 — 20 min before the office breakers tripped under the same 8×450 W load |
| 3, 15:03 UTC, 334 GiB | 300 W | ~~identical stall ~55 s into the cold request~~ **cold 200k 36.8 s, GPU hit, churn 1 completed normally in 146.1 s (3.8k tok/s) at 15:10:17**; churn 2 had been running for 24 s (status lines 15:10:25/15:10:35, KV usage rising) when the container was **killed at 15:10:41** because the operator read the 15:08:05 status line as the request having stopped. The `py-spy` snapshot (stage 1 inside the MoE router GEMM, stage 0 in `dequeue`) is simply a live prefill step. After the SIGKILL **GPUs 0–3 stayed at 100 % / ~130 W with no process** — the kill-during-live-work wedge, not a hung engine; host power-cycled |
| control A, 13:11 UTC, no offload | 450 W | cold, GPU hit and churn 1 OK; at 13:17 the whole chassis (host + BMC) lost power — **office breakers overloaded by 8×450 W**, unrelated to the software; B's limit lowered to 300 W afterwards |
| control B, 15:20 UTC, no offload | 300 W | **full sequence completed**: cold 36.3 s (5.5k tok/s at 300 W), churn 4×549k in 144.4/149.7/151.6/152.6 s (3.6–3.8k tok/s), warm (evicted, no RAM tier) 39.95 s = full recompute, GPU hit 0.37 s; **0 Xid, 0 errors** |

### 12a. Root-cause experiments (2026‑09‑21 evening, B at 300 W, `bench/d2_rootcause/exp.sh`; each run = start → 200k needle cold → GPU hit → 550k prefill under a hang watchdog, py‑spy of all workers, PCIe status, kill, BMC power cycle)

> **Void (2026‑09‑22).** Every "hang" below is the 60 s log-silence watchdog firing ~80 s into a 550k prefill that takes 144–153 s; at each trigger all
> eight PRO GPUs were drawing 285–315 W (computing at the cap), `dmesg` stayed clean, and the "GPUs at 100 % after kill" is the SIGKILL wedge (§12c).
> The PASS rows (E8, E15) are valid but irrelevant. Kept for the record.

| # | variable changed (everything else = production config + `OffloadingConnector`, 32 GiB pool) | result (as logged; all "hang" = watchdog trigger) |
|---|---|---|
| E1 | pool 32 GiB instead of 334 | **hang** ~60 s (during decode of the 200k prompt, ~7 GB stored) |
| E2 | GPU→CPU stores via Triton SM kernel instead of `cuMemcpyBatchAsync` (patched `gpu_worker.py`) | cold OK, **hang** ~10 s into the 550k prefill |
| E3 | `NCCL_P2P_DISABLE=1` (shm transport) | cold OK, **hang** in the 550k prefill |
| E7 | `SimpleCPUOffloadConnector` (different scheduler + worker, own pinning, low-priority streams) | cold OK, **hang** in the 550k prefill |
| E8 | **no vLLM**: 8 containers × (matmuls + 512×22 KB `cudaMemcpyAsync` D2H per iteration), 5 min | **PASS** — 145–156 GiB per GPU copied, 0 Xid |
| E9 | TP8 PP1 (no pipeline parallel) | cold OK (45.7 GiB stored), **hang** in the 550k prefill |
| E10 | MTP off | cold OK, **hang** |
| E11 | `--enforce-eager` (no CUDA graphs) | cold OK, **hang** |
| E12 | `--kv-cache-dtype fp8_ds_mla` (upstream layout, no NVFP4 KV kernels) | cold OK (40.7 GiB stored), **hang** |
| E14 | `--gpu-memory-utilization 0.75` (25 % VRAM free) | cold OK, **hang** |
| E15 | **Qwen3‑1.7B** (dense, `FLASH_ATTN`), same image/topology/connector, 41 × 24k prompts | **PASS** — 107 GB stored in 15 s (7 GB/s) |
| E15b | Qwen3‑1.7B, 500 × 24k prompts | **PASS** — **1.28 TB stored in 2.5 min (8.5 GB/s)**, 0 Xid, PCIe clean |
| E16 | GLM + `CUDA_LAUNCH_BLOCKING=1` | **hang**; blocked calls = `fp8_fp4_mqa_logits` (DeepGEMM, `sparse_attn_indexer.py:466`) and `top_k_per_row_prefill` (`:475`) on 2–3 ranks, the rest in `ncclAllReduce` waiting for them |
| E17 | connector active but **zero stores** (`store_threshold: 2`; `kv_offload_store_bytes` stayed 0) | **hang** in the 550k prefill |
| E18 | `--long-prefill-token-threshold` off (fairness patch inactive) | cold OK, **hang** |

PCIe correctable/uncorrectable error bits and replay counters were clean at every hang. Two consecutive long prompts are needed in most runs (the
first completes; the hang comes 10–60 s into the second); with MTP3 + default connector the hang sometimes came already in the first prompt's decode.

### 12b. Debugging session (2026‑09‑21 evening, B at 300 W; tooling in `bench/d2_rootcause/`, evidence in `results/d2/experiments/` and on B)

> **Void (2026‑09‑22), same reason as §12a.** The snapshots (py-spy, core dumps, cuda-gdb) show live prefill steps, not a hung engine; the nsys
> "profiler signature" is the start-up autotune (all 5 796 `delayStreamKernel` launches at +195…+207 s of the session, READY at +242 s), and the
> post-READY 7.5–7.9 s blocked calls at +377/+392 s are the watchdog's own `cuda-gdb` attaches (§12c). Kept for the record.

| # | experiment | result / finding (as logged) |
|---|---|---|
| E19–E21 | connector (`store_threshold: 2`, no data movement) + per-step trace of the indexer metadata (`indexer_traced.py`), `CUDA_LAUNCH_BLOCKING=1` | **hang** ~311k–344k tokens into the 550k prompt; `cu_seqlen_ks/ke`, seq lens, block tables and slot mapping all consistent at the hanging step (3 000 steps, no anomaly); GPU memory flat (torch reserved 89.7 GB + 1.9 GB non-torch on stage 0; stage‑1 GPUs at 97 GB at hang — tight but E14 showed headroom does not help) |
| E22 | **baseline, no connector**, only the tracer's `torch.cuda.synchronize()` per step | **hang** at 366k tokens — any extra host-side sync/latency per step triggers it; the connector is one such perturbation, not the cause |
| E24 | `--no-async-scheduling` (MTP off) + connector | hang |
| E25 | `CUDA_DEVICE_MAX_CONNECTIONS=1` + connector | hang |
| E26–E27 | cuda-gdb attach at hang (`rc2-dbg`, `--cap-add=SYS_PTRACE`) | host thread blocked inside `cuLaunchKernel`; attach cannot enumerate resident kernels on this platform |
| E28 | `NCCL_DEBUG=INFO,SUBSYS=COLL,P2P` per process | stage‑0 ranks end the step with the PP `Send`, stage‑1 ranks stop inside `AllReduce`; `opCount` not incremented in this NCCL build |
| E29–E30 | user-triggered **GPU core dumps** (`CUDA_ENABLE_USER_TRIGGERED_COREDUMP`) | 3 ranks of a TP group spin in `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` (`ISETP … 0x270f` poll loop), the 4th has **no resident kernel** and its CPU is blocked in a launch — its stream head is a non-kernel wait; the other stage's GPUs are idle |
| E31–E32 | Nsight Systems (`rc2-dbg2`, `--cuda-flush-interval=2000`, `nsys stop --session`) — 41 M events | at session stop the GPUs were **computing again** (the watchdog had caught a >60 s stall); the timeline shows repeated **7.5–12.4 s stalls**: one rank of a TP group runs ~14 600 `device_kernel` (CUTLASS grouped-GEMM candidates), ~4 400 `finalizeMoeRouting`, ~4 600 `doActivation`, expert prefix sums and **966 × `delayStreamKernel` (4.9 s)** — the TRT‑LLM/FlashInfer **profiler signature** — while the other three wait in the pynccl all-reduce; concurrently a 7.5 s `AllReduce grid(3)` on a torch.distributed stream on the profiling ranks. The late rank changes from stall to stall |
| E33 | `--kernel-config '{"enable_flashinfer_autotune":false}'` + connector, 60 s watchdog | one watchdog trigger (unverified whether a real hang) |
| **E34** | same, under nsys, 180 s watchdog, `--churn 4` | **PASS 2/2**: cold 200k → GPU hit → 4 × 550k (2.2 M tokens, **274 GB stored to RAM**) → warm → GPU hit; 0 Xid; no stall > 180 s. First full run of GLM with an active connector |

~~Working hypothesis (2026‑09‑21): runtime FlashInfer/TRT‑LLM MoE tactic profiling desynchronises the TP group; autotune off fixes it.~~ **Withdrawn 2026‑09‑22 — see §12c.**
E34 passed because its watchdog was 180 s (longer than a 550k prefill), not because autotune was off; E33 "hung" with autotune off for the same reason all the others did.

### 12c. Re-analysis (2026‑09‑22): there was no hang

Evidence, all from the raw artefacts of 2026‑09‑21 (`results/d2/experiments/exp_*/{exp.log,reload.log,container.log,nvidia-smi_hang.txt}`, `results/d2/reload_200k*.log`,
`results/d2/experiments/exp_e32_nsys_flush/nsys_timeline.txt`):

| # | observation | consequence |
|---|---|---|
| 1 | **The status logger is silent during a single long chunked prefill.** In every run (with or without connector, autotune on or off, under nsys or not) a 550k request produces 1–2 `Engine 000:` lines ~6 s and ~16 s after it starts (KV usage 2 % → 5 %) and then nothing until the request ends 144–153 s later, when one line reports 54 900 tok/s "prompt throughput" for the whole prompt at once (control B, E34 run 2 with 150–170 s gaps, gate runs of 2026‑09‑22) | log silence is not a progress signal; any watchdog shorter than the longest prefill (here 600 s is safe) fires on healthy runs |
| 2 | **Every `exp.sh` "hang" (E1–E33, 25 runs) is that watchdog.** Same timeline each time: cold 200k done in 36.5 s → GPU hit → churn 1 starts → status lines at +6/+16 s → "HANG detected" 60–65 s after the last line, i.e. ~80 s into a 150 s prefill. `HANG_SECS` was 15 s on top of a 45 s window | 25/25 false positives; the "13/13 hang with the default config" statistic is void |
| 3 | **At each trigger the GPUs were computing, not spinning.** `nvidia-smi_hang.txt`: all 8 PRO GPUs at 285–315 W (the 300 W cap) with 100 % utilisation in 24 of 25 runs (E26 with cuda-gdb attached: 224–260 W). A NCCL spin-wait — the real hang signature — is 100 % utilisation at ~130 W (seen only *after* SIGKILL) | the hardware said "live prefill" every time |
| 4 | **The 334 GiB runs did not stall either.** Run 2: cold 29.2 s, GPU hit, churn 1 in 112.2 s, churn 2 progressing (status lines 12:57:13/23) → Xid 79 at 12:57:29. Run 3: cold 36.8 s, churn 1 in 146.1 s (15:10:17), churn 2 progressing (15:10:25/35) → killed at 15:10:41 by the operator, who read the 15:08:05 line as "the request stopped" | none of the three original "hangs" was a hang; run 3 would have completed like control B |
| 5 | **The SIGKILL wedge explains the "confirmation".** After `docker kill` in the middle of live 8‑GPU work (TP all-reduce + PP send/recv over P2P + in-flight `cuMemcpyBatchAsync`) several GPUs stay at 100 % utilisation / ~130 W with no process until a power cycle (~20 cycles that day). A graceful `docker stop -t 30/60` after a finished run leaves every GPU at 0 % / 1 MiB (E34, all 2026‑09‑22 runs) | "GPUs stuck after kill" is a consequence of the kill, not evidence of a prior hang; **never SIGKILL a busy multi-GPU vLLM on this driver** |
| 6 | **Xid 79 correlates with 8×450 W, not with software.** Both link losses (slots 17 and 65, 12:26 and 12:57) happened at 450 W during normally progressing prefills, 20–50 min before the office breakers tripped under the same load (13:17). At 300 W: 0 Xid in ~25 SIGKILLs and ~1.5 h of full-load prefill on 2026‑09‑21, 0 Xid in 2026‑09‑22's runs | Xid 79 belongs to the power-delivery incident; the only driver-level robustness item left is the SIGKILL wedge (#5) |
| 7 | **No runtime autotuning exists.** `flashinfer/autotuner/autotuner.py:1630` (0.6.18.post1): outside an `autotune(True)` context a cache miss returns the fallback tactic −1 immediately, no profiling. In E29 (autotune on) all 291 `[Autotuner]` log lines are in the warm-up window 20:28:18–20:28:41 (`Autotuning process starts … ends` = 11.6 s per rank, PP0 and PP1 with different config caches), none after READY | the E32 "7.5–12.4 s profiling stalls" are the start-up autotune; the per-rank 11.6 s autotune with the peers waiting in the first collective *is* the "one rank runs 14 600 GEMM candidates while TP0 waits 12 s" picture |
| 8 | **nsys E32 on the absolute timeline** (`nsys_timeline.py`): 5 796 `delayStreamKernel` launches, all at +195.1…+207.5 s (READY at +242 s); during the churn prefill (+287…+361 s) every device is 100 % busy and no CUDA API call blocks >2 s; the blocked calls at +376.9 s and +391.7 s (7.5–7.9 s) coincide with the watchdog's `cuda-gdb -p` attaches (21:23:41–21:24:11). One 8.0 s block of stage‑0 ranks 1–3 at +361.5 s coincides with the 32 GiB CPU pool filling up (~280k stored tokens ≈ 75 s into the churn); the 2026‑09‑22 32 GiB run with a 1 Hz power trace was scheduled to check the eviction path (§12d) | the core dumps (E29/E30: three ranks in the all-reduce, one late) are snapshots of ordinary rank skew inside a live step, not a deadlock |

Root cause of the wrong conclusion: a progress signal (log lines) that does not exist during the very phase under test, plus a destructive reaction (SIGKILL) whose side effect
looked like confirmation. Fixes shipped in `bench/d2_rootcause/exp.sh` (2026‑09‑22): hang = **hardware criterion** (≥ 60 s with every PRO GPU below 200 W while ≥ 1 shows ≥ 50 %
utilisation and no status line for ≥ 45 s) or 600 s of log silence; 1 Hz GPU util/power trace (`gpu_trace.csv`) with a stall classifier (`gpu_trace_stalls.py`: prefill / desync
within a TP group / spin / idle windows); `docker stop -t 60` before any kill.

### 12d. D2 gate (2026‑09‑22, B at 300 W, production configuration rc2 TP4×PP2 DCP1 MTP3, autotune **on**, `KV_OFFLOAD_GB=334` → 360 GiB pinned, `bench/kv_offload_reload.py`)

| run | phase | result |
|---|---|---|
| `exp_d2gate_334_r1` | start | READY 140 s (warm compile cache), `pinned(shmem)=360 GB MemAvailable=350 GB` |
| | cold 200k | TTFT **36.12 s** (5 530 tok/s; control without offload 36.33 s / 5 499 tok/s → **no store overhead**), needle OK, 22.85 GiB stored = 120.0 KiB/token |
| | GPU hit | 0.40 s |
| | churn 4 × 550k | **143.5 / 147.9 / 150.1 / 151.0 s** (3 640–3 826 tok/s) vs control 144.4 / 149.7 / 151.6 / 152.6 s → identical; 274.3 GiB stored in total; `ext_hits/q = 0/2 397 900` (unique prompts, as expected) |
| | **warm from RAM** (prompt evicted from the 1.81M GPU pool by 2.2M churn tokens) | **TTFT 0.46 s** instead of 36.1 s cold / 39.95 s recompute (control B); `loaded = 22.85 GiB` (119.9 KiB/token), `ext_hits = 199 680` tokens (= 3 120 full blocks), needle OK |
| | GPU hit after warm | 0.35 s |
| | health | 1 Hz trace: 618 s of prefill, **0 desync windows, 0 spin windows**; 0 Xid; graceful stop → all GPUs 0 % / 1 MiB |

| `exp_d2gate_334_650k` (`--len 650000 --churn 3`) | cold 650k | TTFT **186.58 s** (3 481 tok/s), needle OK, 74.31 GiB stored |
| | GPU hit | 1.07 s |
| | churn 3 × 550k | 149.9 / 151.9 / 152.6 s (3 600–3 664 tok/s); 262.9 GiB stored in total |
| | **warm** (2.3M tokens > 1.81M pool: LRU evicted the first 484k tokens of the prompt, the last 165k were still on the GPU) | **TTFT 1.33 s**: `prefix_cache_hits +165 504` (GPU) + `external hits 483 904` (RAM), `loaded = 55.36 GiB` (= 484k × 120 KiB), needle OK — partial GPU hit + RAM reload compose correctly |
| | GPU hit after warm | 1.03 s |
| | health | 0 desync / 0 spin windows, 0 Xid, clean stop |
| decode with the 334 GiB tier active (`bench_decode.py --conc 1,16 --tokens 256 --rounds 3`, 300 W) | @1 | **21.2 ms/step**, 142 tok/s, acceptance 3.01 |
| | @16 | **70.9–71.1 ms/step** (r1–r2), 658–665 tok/s, acceptance 2.9 |
| decode without the tier (same day, same power; run right after) | @1 / @16 | **21.1–21.2 ms/step** (135–147 tok/s, acceptance 2.9–3.1) / **70.7–74.5 ms/step** (624–665 tok/s) → the tier is free in decode |
| fairness (`bench_fairness.py --others 4 --short-tokens 1000`: 131k prefill + 4 decoders + new request), tier on vs off | off | shared prefill 2 780 tok/s (TTFT 47.3 s), others 14.6 tok/s during the prefill, **new request TTFT 0.43 s**; `smoke.py --conc 4` 293 tok/s, single 140.7 tok/s |
| | on (334 GiB) | shared prefill 2 802 tok/s (TTFT 46.8 s), others 14.5 tok/s, **new request TTFT 0.47 s**; `smoke.py --conc 4` 291 tok/s, single 143.0 tok/s; 16.2 GB stored during the run, 0 preemptions → no cost under concurrency either |

**D2 conclusion (2026‑09‑22).** The native CPU KV tier works with this stack as designed: stores are free (prefill 200k/550k/650k and decode @1/@16 identical to the
no-tier baseline, also under concurrency), a 200k prompt evicted from the GPU comes back in 0.46 s instead of 36–40 s, a 650k prompt in 1.33 s instead of 187 s
(partial GPU hits compose with RAM reloads), the eviction path of a full tier costs nothing measurable, and 2.3 h of tier-active prefill at 300 W produced 0 Xid
and 0 stall windows. **Deployed on A the same day, 13:06 UTC, with `KV_OFFLOAD_GB=668` (720 GiB pinned; A has 1.5 TB) — §8.**
| `exp_pool32_autotune_on` — **exactly the E1–E32 configuration** (autotune on, connector, 32 GiB tier), 200k + 4 × 550k; the tier fills after ~280k stored tokens and is then overwritten ~9× (297 GiB through 32 GiB) = the eviction path a full tier on A would exercise | | reproducer **completed** (yesterday's 25 "hangs" in this configuration were the watchdog); cold 37.5 s, churn **147.9 / 151.5 / 152.4 / 152.6 s** (no cost of continuous eviction), warm = full recompute 39.95 s as expected for a tier smaller than the churn (D2 verdict FAIL by design: `load_bytes` 0), GPU hit 0.38 s; 671 s of prefill, **0 desync / 0 spin windows** (the 8 s block seen in E32 at pool-full time did not reproduce at 1 Hz resolution), 0 Xid, clean stop |

Text identity is not a criterion on this stack (the GPU-hit control of the same prompt already differs from the cold text; needle + `kv_offload_load_bytes` are the gate).
Reference: the 2026‑09‑21 no-offload decode at 450 W was 21.0–21.2 ms @1 and 70.6–73.3 ms @16 (`results/d2/baseline_decode.log`), i.e. the tier costs nothing in decode either.
