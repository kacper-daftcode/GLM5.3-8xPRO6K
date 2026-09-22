# FAIRNESS — long prefills vs. everyone else

The engine runs one batch per step. A long prefill (100k–700k tokens) is split into chunks; a step that contains a prefill chunk is expensive
(2048 tokens → ~270 ms, 512 → ~85 ms), while decoding users advance one token per step (with async scheduling a decoding request enters every
other batch, so its inter-token latency is 2 × step). Without a limit, one 131k prefill made other users crawl at 4.8 tok/s and a newly arriving
request waited 39.5 s for its first token; a 700k prefill starved everyone for minutes.

`--long-prefill-token-threshold N` caps the prefill chunk at N tokens per step. `patch_sched_fair.py` applies it **only while the engine is shared**
(`running + waiting > 1`), so a lone long prefill keeps full speed (7.3k tok/s @131k on TP4×PP2).

## Measured trade-off (131k prefill + 1 decoding user + 1 new request; TP8 DCP2 numbers from 2026‑09‑17, TP4×PP2 for the current config)

| chunk when shared | shared prefill tok/s | new request TTFT | other users tok/s (ITL) | 131k prefill wall time |
|---|---|---|---|---|
| none (2048) | 3 071 (TP8) | 39.5 s | 4.8 (600 ms) | 43 s |
| 1024 | 2 944 (TP8) | 0.9 s | 9–11 (333 ms) | 45 s |
| **512 (production)** | 2 761 (TP8) / **4 051 (TP4×PP2)** | **0.4–0.5 s** | **17–19 (170 ms)** | 33 s (TP4×PP2) |
| 256 | 1 959 (TP8), ≈ −40% | 0.36 s | 21–25 (131 ms) | ~55 s |
| 128 | lower still | – | ~30 | ~2–3× solo |

Why chunks below 512 cost so much: a step with a 512-token chunk takes ~85 ms, of which ~60–70 ms is nearly independent of the chunk size —
the MoE kernel floor (512 tokens × top-8 = 4096 rows / 256 experts = 16 rows per expert, tiles 1/8 full: 25 ms), the TP4 all-reduce over PCIe
(22 ms, at the bus limit), attention + indexer (18 ms). Halving the chunk therefore does not halve the step: prefill throughput falls faster
than the decoders' speed rises. Even with the MoE at zero cost the step would be ~60 ms → ~24 tok/s for others; "≥30 tok/s for others"
requires a ≤55 ms step at chunk 512, which these kernels cannot deliver — it is a policy choice, not an engineering gap.

Worst case observed (soak): a 650k prefill under 25 concurrent streams took 12 min (410 s vs 190 s alone) while the other streams got ~4 tok/s each.
The previous production would have blocked all of them completely for the duration of the prefill and queued new requests behind it.

## Dynamic chunk by request count (D3, measured 2026‑09‑19)

`patch_sched_fair.py` gained an optional second stage: `VLLM_LONG_PREFILL_THRESHOLD_DYNAMIC="N:T"` — with more than N requests in the system
(running + waiting) the chunk shrinks from the base threshold to T. Measured on B (TP4×PP2 DCP1 MTP3, 131k prefill, `bench_fairness.py --others K
--short-tokens 1000`, decoders long enough to cover the whole prefill window; raw data `results/d3/`):

| decoding users | chunk while shared | prefill tok/s (shared) | tok/s per decoding user | all decoders tok/s | new request TTFT |
|---|---|---|---|---|---|
| 8 | **512** (static, production) | **2 763** | 14.4 | 115 | 0.37 s |
| 8 | 256 (`4:256`) | 1 913 | 19.4 | 155 | 0.37 s |
| 8 | **128 (`4:128`)** | 2 145 | **26.7** | **214** | 0.28 s |
| 16 | 512 | 2 603 | 13.4 | 214 | 0.55 s |
| 16 | 256 | 1 805 | 18.3 | 293 | 0.32 s |
| 16 | **128** | 1 920 | **23.4** | **374** | 0.26 s |

- With ≤ 4 requests nothing changes (512, full-speed lone prefill). Above that, **128 gives decoders +85% (8 users) / +75% (16 users) for
  −22% / −26% prefill throughput while shared**; 256 is dominated by 128 on both axes here (the scheduler places a 256 chunk only every
  other step at this load — not understood yet, noted as a follow-up). Earlier runs with 600-token decoders overstated the prefill numbers for
  small chunks: the decoders finished early and the prefill continued alone at full chunks (`results/d3/early_*`).
- **In production since 2026‑09‑20 16:00 UTC**: `LONG_PREFILL_DYNAMIC=4:128` (`deploy/serve-glm53-prod.sh`, image `2026.09.19-rc2`) — interactive users are the
  ones who notice; long-context users lose a quarter of the shared prefill speed only while ≥ 5 requests are active. Rollback: `CAND=1 LONG_PREFILL_DYNAMIC=`.

## Options

1. Keep 512 (current): best total throughput, acceptable responsiveness (TTFT 0.5 s, 17–19 tok/s).
2. 256: others +25%, long-context users −40% prefill while shared (1 decoder); at 8–16 decoders dominated by 128.
3. **Dynamic (`N:T`, implemented, measured above)** — 512 up to 4 requests, 128 above.
4. Kernel-level: B12X "dynamic" MoE for prefill chunks measured −6…−11% of MoE time (≈ −2–3% step) — does not change the picture.
5. Priority scheduling (`--scheduling-policy priority`) with client-side priorities for short interactive requests — untested here.

Decide from production telemetry: `vllm:num_requests_waiting`, `vllm:request_queue_time_seconds`, complaints about TTFT vs. about long-context latency.
