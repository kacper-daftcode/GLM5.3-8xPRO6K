# Changelog

## 2026‑09‑20 — host A driver 595.71.05 → **610.43.02** (open kernel module, NVIDIA repo, held), reboot 17:48–17:52 UTC; production auto-restarted,
verify OK (KV 1 813 657 — +6.5k tokens vs 595 —, 146 tok/s single, parity 8/8, acceptance 2.92). Public repository prepared (`docs/internal/make-public.sh`).

## 2026.09.19-rc2 — `glm53-stack:2026.09.19-rc2` (`68c8e892a30a`), **in production since 2026‑09‑20 16:00 UTC** (`CAND=1` = rc2 + `LONG_PREFILL_DYNAMIC=4:128`)
- Deployed after the 2026‑09‑19 incident: GPU 0 of host A hung at 12:59 UTC (engine dead 13:03; teardown left the GPU in "requires reset", reset unsupported,
  vLLM processes zombie with VRAM held on all GPUs); the graceful reboot hung for 26 h until a power cycle on 2026‑09‑20 15:53 UTC. Production outage
  2026‑09‑19 13:03 → 2026‑09‑20 16:00 UTC. RUNBOOK §5 has the recovery recipe (sysrq reboot, keep the container for auto-restart).
- `patch_sched_fair.py`: optional dynamic long-prefill chunk (`VLLM_LONG_PREFILL_THRESHOLD_DYNAMIC=N:T`; knob `LONG_PREFILL_DYNAMIC` in the deploy and
  testbed scripts). Measured (D3): with 8–16 decoding users `4:128` gives them +85% / +75% tok/s for −22% / −26% shared prefill; ≤4 requests unchanged.
  Default (unset) behaviour identical to rc1; gate on B passed (KV 1 813 657, 21.1 ms @1, 69.3 ms @16, parity).
- D6: `deploy/alert-prod.sh` (cron every minute on the production host): health, waiting, preemptions, Xid, errors, MTP acceptance window.
- `bench/serve_upstream.sh`: run the checkpoint on an unpatched upstream vLLM image (issue reproduction / port work). `bench/bench_fairness.py --others K`.
- Upstream status on `vllm/vllm-openai:nightly` 2026‑09‑19 (vLLM 0.29.1rc1, V2 runner), `docs/upstream/ISSUE-pp-mtp-acceptance.md`: (A) GLM‑5.3 NVFP4 + MTP
  **does not load** — the draft's shared head is built as NVFP4 and hits the new NaN sanity check before it is tied to `lm_head` (workaround: `config.json`
  `quantization_config.ignore`); (C) FlashInfer autotune warmup stalls on SM120 TP8 (workaround `--kernel-config '{"enable_flashinfer_autotune":false}'`);
  (B) with both workarounds, TP4×PP2 + MTP3 + async emits longer, more often non-terminating greedy outputs than TP8 on GSM8K‑500 (96.2% / 15 truncated /
  382 tokens vs 97.6% / 5 / 298; acceptance 3.40 vs 3.49) — our patched stack on PP2 matches TP8 (97.6% / 6 / 310). Basis for PR‑1.
- Publication scope C: `docs/rtx6kpro/glm-5.3.md` (page draft), `docs/upstream/PR-2-long-prefill-shared-only.md`, publishing procedure in
  `docs/internal/PUBLISH-CHECKLIST.md` (orphan `public` branch without internal docs).

## Unreleased (2026‑09‑19, no image change)
- GSM8K on the full test set (1319): C3 97.19%, without `lm_head` FP8 97.50%, without `lm_head`+`o_proj` 97.19%, no online quantization 97.50% —
  all within noise (paired McNemar p ≥ 0.45); production configuration unchanged (docs/RESULTS.md §6a).
- `bench/run_cand.sh`: knobs `FP8` / `FP8_CHANNEL` / `NVFP4_MOE` / `LONG_PREFILL` (same semantics as the production script), automatic compile-cache
  suffix per quantization set. `bench/gsm8k_compare.py`: paired comparison of `gsm8k_eval.py` outputs. `deploy/acceptance_by_hour.py`: MTP acceptance
  per hour and requests per client from the engine log.
- `bench/serve_full.sh`: profiler traces in `PROF_DIR` instead of `bench/prof`; `soak.sh`/`run_b1s2.sh` use script-relative paths.
  Testbed B now runs from a checkout of this repository (updated by `git push testbed main`).
- RUNBOOK: compile-cache lifecycle (first start compiles, second start recompiles once — `Kernel index 1 not found` — third start loads AOT in 4 s).
- No host-specific paths in the repository: `deploy/*.sh` read `deploy/env`, `bench/*.sh` read `bench/env` (git-ignored; templates `*.example`) for
  `MODEL_DIR`/`MODEL`, `CACHE_ROOT`, `RESULTS_DIR`, `PROF_DIR`, `PATCHED_DIR`; the checkpoint path is now required (`:?`) instead of defaulted.
  `docs/history/` and `docs/internal/` are `export-ignore` (internal handoffs stay out of `git archive`). Comment-only path fix in the documentation
  copy `image/patches/team/nvfp4_expand.cuh` (the file inside the base image is unchanged; the image fingerprint is unaffected).
- `deploy/serve-glm53-prod.sh`: fix — `CAND=1 NVFP4_MOE=` / `FP8_LINEARS=` were silently replaced by the preset defaults (`${VAR:-}`); empty now means off.
- D5 closed: NVFP4 vs BF16 draft experts — paired GSM8K‑500 on B (3.44 vs 3.45 tok/step), bench prompts (2.89 vs 2.90) and a production A/B on the
  gateway probe (2.86 vs 2.85): no acceptance cost; the NVFP4 draft (+0.6 ms/step, +140k KV tokens) stays (docs/RESULTS.md §6b, `results/d5/`).
- Production restart with a warmed compile cache measured: 2 min 17 s downtime (AOT artifacts loaded, compile 4.2 s).

## 2026.09.19-rc1 — `glm53-stack:2026.09.19-rc1` (`cfff0525779a`)
- First build from `image/Dockerfile` in this repository; fingerprint-identical to the production image `vllm-nightly-fi614:nvfp4-fi618` (`b7c70850807d`).
- Production configuration switched from MTP5 to MTP3 (`CAND=1` preset): equal single-stream, +4–7% at 4–32 streams, KV pool +1%.
- `bench/parity.py`: long-context needle tests use the chat format (raw completions ≥131k flip non-deterministically); math test `max_tokens` 700 → 1200.
- `bench/soak.py`: long-context needle via chat with `max_tokens` 800.

## 2026.09.18 — `vllm-nightly-fi614:nvfp4-fi618` (`b7c70850807d`), deployed to production 20:25 UTC
- `patch_b1s2.py`: MTP draft experts quantized online to NVFP4 (ModelOpt checkpoint layout), per-channel FP8 for excluded linears, draft `eh_proj` in FP8.
- Pipeline partition 41/37 (KV pool 1.79M tokens).
- 195-minute mixed-traffic soak with restarts: no errors.

## 2026.09.17 — `vllm-nightly-fi614:nvfp4-fi618` (first version)
- FlashInfer 0.6.14 → 0.6.18.post1 (AOT sparse_mla_sm120 removed, JIT baked for 120f, quack-kernels 0.6.5).
- `patch_pp_mtp.py`: MTP speculative decoding under pipeline parallelism with async scheduling (7 fixes).
- `patch_sched_fair.py`: long-prefill threshold applied only when the engine is shared.
- `patch_fp8_excluded.py`: online FP8 W8A8 for BF16 layers excluded from the checkpoint quantization.
- `patch_b12x_hybrid.py`: B12X/CUTLASS MoE hybrid (disabled by default; negative result with MTP).

## 2026.09.10 — `vllm-nightly-fi614:nvfp4-car` / `nvfp4` (team images)
- Upstream nightly 2afa3f7e9 + NVFP4 KV cache for sparse-MLA + DCP for SM120 (+ custom all-reduce over PCIe, disabled).
