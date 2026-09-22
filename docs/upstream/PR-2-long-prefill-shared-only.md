# Upstream PR draft — apply `--long-prefill-token-threshold` only while the engine is shared

Target: `vllm-project/vllm`, `vllm/v1/core/sched/scheduler.py` (+ `SchedulerConfig`, CLI arg, scheduler unit test).
Status: draft; port of `image/patches/ours/patch_sched_fair.py` (nightly 2afa3f7e9, 2026‑07) to `main`. Open as a small, self-contained PR.

## Title

`[Scheduler] Option to apply long_prefill_token_threshold only when more than one request is in the system`

## Problem

`--long-prefill-token-threshold N` caps the number of prompt tokens a long request may take per step. It protects decoding users from
multi-second steps caused by a 100k–700k prefill: with 2048 batched tokens a step containing a full prefill chunk takes ~270 ms on
8× RTX PRO 6000 (GLM‑5.3 NVFP4, TP4×PP2), so decoders drop to ~5 tok/s and a newly arriving request waits for its first token until the
prefill finishes (39.5 s measured for a 131k prompt).

The threshold, however, is applied unconditionally. A **lone** long prefill — the common case for a single long-context user at night, or
for batch summarization — is slowed down by the same factor it is meant to protect others from: at N = 512 a 131k prefill takes
~2× longer than with full chunks (7.3k → ~4k tok/s), for nobody's benefit.

## Proposal

A boolean scheduler option (name to be bikeshedded): `--long-prefill-threshold-shared-only` / `SchedulerConfig.long_prefill_threshold_shared_only`
(default `False` to keep current behaviour). When set, the effective threshold for a step is computed once per `schedule()`:

```python
threshold = self.scheduler_config.long_prefill_token_threshold
if (self.scheduler_config.long_prefill_threshold_shared_only
        and threshold > 0
        and len(self.running) + len(self.waiting) <= 1):
    threshold = 0          # alone in the system: full chunks
```

and used in both places where `long_prefill_token_threshold` caps `num_new_tokens` (the RUNNING loop and the WAITING loop). As soon as a
second request arrives the next step is capped again, so a newcomer's TTFT is bounded by one step (`max_num_batched_tokens` tokens of prefill,
≈ 270 ms above) plus its own first chunk.

Optional follow-up (measured, not part of this PR): a dynamic cap that shrinks the chunk further when many requests are decoding
(`N:T`, e.g. 512 when ≤ 4 requests, 256 above) — see the numbers below; it needs a design discussion about where such a policy belongs.

## Measurements (8× RTX PRO 6000, GLM‑5.3 NVFP4, 2048 batched tokens, 131k-token prefill; `bench/bench_fairness.py`)

Rows marked TP8 were measured on TP8 DCP2 MTP5 (2026‑09‑17), the 512 rows on TP4×PP2 DCP1 MTP3 (the production configuration).

| chunk while shared | shared prefill tok/s | new request TTFT | 1 other user tok/s (ITL) | lone 131k prefill |
|---|---|---|---|---|
| none (2048), TP8 | 3 071 | **39.5 s** | 4.8 (600 ms) | full speed |
| 1024, TP8 | 2 944 | 0.9 s | 9–11 (333 ms) | |
| 512, unconditional (upstream today) | 4 051 | 0.4–0.5 s | 17–19 (170 ms) | **4–6k tok/s** (bounded by the ~85 ms step at chunk 512; 4.05k measured with one decoder present) |
| **512, shared-only (this PR)** | **4 051** | **0.4–0.5 s** | **17–19 (170 ms)** | **7.3k tok/s** (full 2048 chunks) |
| 256, TP8 | 1 959 | 0.36 s | 21–25 (131 ms) | |

With 8 / 16 decoding users the same threshold gives 14.6 / 14.0 tok/s per user (chunk 512); 256 gives 20.6 / 19.4 and 128 gives 27.9 / 24.7
tok/s per user at the cost of prefill throughput while shared (`docs/FAIRNESS.md`, `results/d3/`).

## Tests

`tests/v1/core/test_scheduler.py`: with `long_prefill_token_threshold=4`, `long_prefill_threshold_shared_only=True`:
1. one waiting request with a 16-token prompt → scheduled with all 16 tokens (no cap);
2. add a second request → the long request's next chunk is capped at 4 tokens, the second request gets its tokens in the same step;
3. `shared_only=False` → capped at 4 in both cases (current behaviour).

## Notes for reviewers

- No change when the option is off; no new state — the decision is a pure function of `running`/`waiting` sizes at the start of the step.
- Related: `--max-num-partial-prefills` / `--max-long-partial-prefills` are still rejected by the V1 arg guard; this option does not depend on them.
- Production use since 2026‑09‑18 on the configuration above (195-minute mixed soak + production traffic, 0 errors).
