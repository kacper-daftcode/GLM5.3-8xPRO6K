# Upstream issue drafts — MTP speculative decoding with ModelOpt NVFP4 checkpoints on current `main`

Target: `vllm-project/vllm` issues, then PR‑1 (port of `image/patches/ours/patch_pp_mtp.py` fixes 3–4 to the current runner).
Tested 2026‑09‑19 on `vllm/vllm-openai:nightly` = vLLM `0.29.1rc1.dev397+ga8d1aa9c9`, torch 2.13.0+cu130, V2 model runner, 8× RTX PRO 6000
(`bench/serve_upstream.sh`, logs `results/p41/`).

## Issue A (blocking, found first) — MTP draft head of a ModelOpt NVFP4 checkpoint cannot be loaded

**Title:** `[Bug] MTP speculative decoding with a ModelOpt NVFP4 checkpoint fails at load: "NVFP4 weight_scale for layer 'parallel_lm_head' was never loaded (still NaN)"`

- Command: `vllm serve /model --quantization modelopt_fp4 --tensor-parallel-size 8 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
  (same with TP4×PP2, with and without `--async-scheduling`, `--kv-cache-dtype fp8_ds_mla` or `auto`). Model: `incoai/GLM-5.3-NVFP4`
  (DeepSeek-V3-style MTP layer `model.layers.78`; the checkpoint has **no** `model.layers.78.shared_head.head.weight` — the MTP head shares `lm_head`,
  which is BF16 and listed in `exclude_modules`).
- Traceback: `gpu/model_runner.py:412 load_model → speculator.load_model → mtp/speculator.py:20 load_draft_model → eagle/utils.py:110 load_eagle_model
  → get_model → base_loader.py:91 process_weights_after_loading → modelopt.py:2530 → modelopt.py:1968 process: RuntimeError: NVFP4 weight_scale for
  layer 'parallel_lm_head' was never loaded (still NaN)` on every rank that hosts the draft (all TP8 ranks; PP stage 1 ranks with PP2).
- Cause: the draft's `SharedHead.head` (`ParallelLMHead`, prefix `model.layers.78.head`) is built with the NVFP4 quant method because that prefix
  is not in the checkpoint's `exclude_modules` (the checkpoint has no such tensor at all), so nothing loads into it; the NaN sanity check in
  `process_weights_after_loading` fires **before** `load_eagle_model` ties the head to the target's `lm_head` (`eagle/utils.py:127–140`, "MTP layers
  route logits through layer.shared_head.head").
- The July nightly (2afa3f7e9, V1 runner) loaded the same checkpoint fine (no NaN check; the head was tied after loading).
- Workaround (verified): add `model.layers.78.shared_head.head` (and `model.layers.78.head`) to `quantization_config.ignore` in an overlay
  **`config.json`** — this vLLM reads the compressed-tensors-style `quantization_config` from `config.json`, not the legacy `hf_quant_config.json`
  (editing the latter has no effect). With that the draft loads and is tied to the target `lm_head`; acceptance is normal (below).
  Proposed fix: build the draft head with `quant_config=None` when the draft has no own lm_head (`has_own_lm_head=False`), or skip the NaN
  check for heads that will be replaced by the target's; alternatively treat "no tensor loaded and `lm_head` excluded" as excluded.

## Issue C (found while testing) — FlashInfer autotune warmup stalls with TP8 on SM120

**Title:** `[Bug] FlashInfer autotune kernel warmup never completes on 8× RTX PRO 6000 (SM120): rank 0 at 100% GPU, other ranks idle`

- After weight loading, `kernel_warmup.py:358 Running FlashInfer autotune with 2048 tokens and token buckets (1 … 2048)` is logged by all ranks
  at the same second; then nothing for >12 minutes: GPU 0 at 100% utilisation (spinning), GPUs 1–7 at 0%, worker CPUs ~113% (busy-wait),
  no new entries in `flashinfer_autotune_cache/`, `shm_broadcast` "No available shared memory broadcast block found in 60 seconds" every minute.
  Same with the NVFP4 MoE backend `FLASHINFER_CUTLASS` auto-selected. Our patched July nightly (no autotune warmup) does not have this stage.
- Workaround (verified): `--kernel-config '{"enable_flashinfer_autotune":false}'` — startup then completes in ~2 min (TP8) / ~2 min (TP4×PP2).
- To do before filing: repeat once with `VLLM_LOGGING_LEVEL=DEBUG` and a `py-spy dump` of rank 0 and rank 1 to name the collective/op it waits on.

## Issue B — acceptance under pipeline parallelism with async scheduling (the original 4.1 question)

Measured on `vllm/vllm-openai:nightly` 2026‑09‑19 (`4cbfd34aac14`, vLLM 0.29.1rc1.dev397) with the two workarounds above, `incoai/GLM-5.3-NVFP4`,
`--kv-cache-dtype fp8_ds_mla`, `max-model-len 131072`, MTP3, `--async-scheduling`; `bench/bench_decode.py --conc 1 --tokens 256 --rounds 3`
(rounds 1–2 after warm-up; `results/p41/`):

| configuration | ms/step | tok/s | acceptance (tok/step) | greedy facts (`parity.py`: qa/math/tool/needle-32k) |
|---|---:|---:|---:|---|
| TP8 (no PP) | 25.1 | 124–128 | **3.13 / 3.21** | all OK |
| TP4 × PP2, async | 26.8–27.1 | 107–110 | **2.91 / 2.93** | all OK; texts diverge from TP8 as they do between any two MTP runs |

PP2 loses ≈ 0.25 tok/step (−8%) of acceptance on identical prompts at 1 stream; with our patched stack the same PP2 configuration measures 3.05–3.07
(rc2 gate, `results/p41/rc2_decode.log`).

**GSM8K‑500 paired (greedy, chat with reasoning, `max_tokens` 2048, conc 8, `max-num-seqs` 16; `results/p41/gsm8k_d_*.json`, `bench/gsm8k_compare.py`):**

| configuration | accuracy | truncated @2048 | completion tokens (mean / median / p90) | acceptance (Δ `/metrics`) | tok/s |
|---|---:|---:|---|---:|---:|
| upstream `main`, TP8 | **97.6%** | 5 | 298 / 261 / 414 | 3.49 (106 289 / 42 740 drafts) | 482 |
| upstream `main`, TP4×PP2 async | **96.2%** | **15** | **382 / 276 / 613** (+28% tokens, +48% p90) | 3.40 (134 675 / 56 108) | 379 |
| our patched July nightly, TP4×PP2 async (`results/d5/gsm8k_d5_nvfp4draft.json`, same prompts) | 97.6% | 6 | 310 / 260 / 436 | 3.44 | 479 |

Paired: upstream PP2 vs TP8 — 10 vs 3 discordant answers (McNemar p = 0.09), 483/500 identical predictions; our PP2 vs upstream TP8 — 3 vs 3, 492/500.
**So on current `main` the PP2 + MTP + async path does not merely lose acceptance: it emits different (longer, more often non-terminating) outputs
for greedy prompts** — consistent with stale/mis-aligned draft state on the first stage being accepted as sampled tokens. With the two fixes carried
in `patch_pp_mtp.py` (spec-state broadcast to rank 0, `output_token_ids` alignment on all ranks) PP2 output statistics match TP8.

## Title

`[Bug] MTP/EAGLE speculative decoding with pipeline_parallel_size > 1 and async scheduling: draft state not propagated to the first stage → low acceptance / wrong outputs`

## Summary

With `--pipeline-parallel-size 2 --speculative-config '{"method":"mtp",...}' --async-scheduling`, the drafter runs on the last PP rank but the
first rank samples/accepts on the next step. On the nightly of 2026‑07‑08 (2afa3f7e9) four things were wrong; #46994 and #53575 fixed the
first two on `main` (draft `embed_tokens`/`lm_head` loading under PP, guards for `drafter is None` on non-last ranks) and #53575 still
carries the note "acceptance degradation under PP", which matches the remaining two:

1. **Spec-decode state is not broadcast from the last rank to rank 0.** After the drafter runs, rank 0 needs `prev_sampled_token_ids`,
   `valid_sampled_token_count` and `_draft_token_ids` for the next step's input preparation; without them the next step verifies stale
   or empty drafts. Fix in our patch: broadcast `[next_token, valid_count, draft tokens]` from the last rank after the drafter, receive on
   PP0 and populate the runner state.
2. **`output_token_ids` length differs between ranks.** Non-last ranks append a different number of output tokens per request than the last
   rank (they do not see the accepted count), so `num_output_tokens` and the KV/position bookkeeping drift. Fix: align `output_token_ids` to
   `num_output_tokens` on every rank.

Symptoms: acceptance length collapses (< 2 with MTP3, vs 2.9–3.0 with TP only), or outputs diverge between PP=1 and PP=2 for the same greedy
prompt; sync scheduling (`--no-async-scheduling`) hides part of the problem but is not supported for this model/config combination.

## Reproduction on current main (to fill)

- Image: `vllm/vllm-openai:nightly` digest `sha256:4cbfd34aac14…` (2026‑09‑19).
- Model: `incoai/GLM-5.3-NVFP4` (or a small MTP model, e.g. DeepSeek-V3-style with `num_nextn_predict_layers`), 8× RTX PRO 6000.
- Command TP8 (reference): `vllm serve /model --tensor-parallel-size 8 --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --kv-cache-dtype fp8_ds_mla …`
- Command TP4×PP2: same with `--tensor-parallel-size 4 --pipeline-parallel-size 2 --async-scheduling`.
- Measure: `bench/bench_decode.py --conc 1 --tokens 256 --rounds 3` → acceptance from `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_drafts_total`;
  greedy parity of a fixed prompt between PP1 and PP2 (`bench/parity.py`).
- Expected: acceptance equal to TP8 (±0.1), identical greedy output. Observed on main: **TBD**.

## Proposed fix (PR‑1)

Port of the two remaining fixes to the V2 runner (`vllm/v1/worker/gpu/…`): (a) after `propose()` on the last rank, broadcast the spec state
to the first rank (one small collective per step, only when `pp_size > 1`), and (b) make `output_token_ids` handling identical on all ranks.
Test: `tests/v1/spec_decode/` PP2×TP1 with a small MTP/EAGLE model — acceptance and outputs identical to PP1.

## Evidence from the patched nightly (2026‑07 base + the 7 fixes)

TP4×PP2 DCP1 MTP3 on GLM‑5.3 NVFP4: acceptance 2.9–3.0 tok/step on benchmark prompts (same as TP8 MTP3 ≈ 2.9), greedy parity with TP8 on
needle/math/tool prompts, 195-minute soak and production traffic without errors (`docs/RESULTS.md`).
