# RUNBOOK — operating glm53-stack

Hosts in the reference deployment: **A** = production (8× RTX PRO 6000 + 2× RTX 5090 used by other services, 300 W cap),
**B** = testbed (same 8× PRO 6000, **300 W since 2026‑09‑21** — 8×450 W under full load tripped the office breakers twice that day; the limit is applied at boot by B's `nvidia-powerlimit.service` → `/usr/local/sbin/nvidia-set-power-limit.sh` (`TARGET=300`, the RTX 5090 is clamped to its 400 W minimum); A has the 8 PRO cards at 300 W and the team's two 5090s at their 600 W default). Both run the NVIDIA open kernel module **610.43.02** from NVIDIA's CUDA apt repository
(`nvidia-open` metapackage, packages on `apt-mark hold`; A since 2026‑09‑20 after a GPU fault on 595.71.05, see §5), `nvidia-container-toolkit` 1.19.1. Host-specific locations (checkpoint directory, cache root, results directory) are **not** in the
scripts: `deploy/*.sh` read `deploy/env` and `bench/*.sh` read `bench/env` (both git-ignored; templates `deploy/env.example`, `bench/env.example`).
Below, `$CACHE_ROOT` is the host cache root from that file.

## 1. Images and versions

| tag | image ID | role |
|---|---|---|
| `vllm-nightly-fi614:nvfp4` | `ccb91de90880` | team base (upstream nightly 2afa3f7e9 + KV-NVFP4 + DCP SM120). Archived on host A: `vllm-nightly-fi614_nvfp4_ccb91de90880.tar.zst` (+`.sha256`) |
| `vllm-nightly-fi614:nvfp4-fi618` | `b7c70850807d` | production image (2026‑09‑18). Archived on host A: `vllm-nightly-fi614_nvfp4-fi618_b7c70850807d.tar.zst` |
| `glm53-stack:2026.09.19-rc1` | `cfff0525779a` | rebuilt from `image/Dockerfile`; fingerprint-identical to `b7c70850807d` (`image/tools/image_fingerprint.sh`) |
| `glm53-stack:2026.09.19-rc2` | `68c8e892a30a` | **production since 2026‑09‑20 16:00 UTC** with `LONG_PREFILL_DYNAMIC=4:128` (rc1 + D3-capable scheduler; default behaviour identical to rc1). Gate on B: KV 1 813 657, 21.1 ms @1 / 69.3 ms @16, parity OK; verify on A: KV 1 807 193, 149 tok/s single, parity 8/8 |

Rebuild: `docker build -t glm53-stack:<date> image/` (~15 min, internet: PyPI + flashinfer.ai). Fingerprint check:
`docker run --rm --entrypoint bash -v $PWD/image/tools:/t:ro <image> /t/image_fingerprint.sh > fp.txt` and diff against the reference.
Restore an archived image: `zstd -dc <file>.tar.zst | docker load`.

Distribute between hosts: `docker save <image> | ssh <host> docker load` (40 GB, ~2.5 min on 1 GbE — layers shared with the base are skipped),
or a local registry (`registry:2`).

## 2. Production start / stop / rollback (host A)

Only inside a maintenance window (traffic ≈ 0 after 22:00 local; check `Running: 0` in the engine log for ≥10 min).

```bash
cd deploy
CAND=1 ./serve-glm53-prod.sh          # stop old (-t 60) → wait VRAM free → start → wait /health (~4 min total with the RAM tier: 2.5 min + pinning 720 GiB)
./verify-prod.sh <tag>                # health, startup log (TQ_B1/TQ_FP8 lines, KV pool), smoke, parity, metrics
```

Knobs (all overridable, `DRY=1` prints the docker command): `IMAGE TP PP DCP MTP NCCL_MODE PP_PARTITION FP8_LINEARS FP8_CHANNEL NVFP4_MOE
LONG_PREFILL LONG_PREFILL_DYNAMIC KV_OFFLOAD_GB KV_OFFLOAD_OPTS CACHE MAX_SEQS BATCHED MODEL_DIR CACHE_ROOT`. `CAND=1` preset (= production since 2026‑09‑20, RAM tier since 2026‑09‑22 13:06 UTC) = `IMAGE=glm53-stack:2026.09.19-rc2
TP=4 PP=2 DCP=1 MTP=3 NCCL_MODE=p2p_sys PP_PARTITION=41,37 FP8_LINEARS=fused_qkv_a_proj,q_b_proj,o_proj,shared_experts,indexer.wq_b,lm_head,eh_proj FP8_CHANNEL=1
NVFP4_MOE=layers.78.mlp.experts LONG_PREFILL=512 LONG_PREFILL_DYNAMIC=4:128 KV_OFFLOAD_GB=668 CACHE=$CACHE_ROOT/vllm-fi618-cand-mtp3`. An explicitly empty `FP8_LINEARS=` /
`NVFP4_MOE=` / `LONG_PREFILL_DYNAMIC=` switches that feature off (the preset uses `${VAR-default}`, so empty is honoured); `KV_OFFLOAD_GB=0` switches the RAM tier off
(same image, same cache — the smallest rollback). The image guard for PP+MTP accepts `*fi618*` and `glm53-stack:*`. The 2026‑09‑22 deployment: stop 13:02:15 UTC (at a
`Running: 0` moment, ~13 req/min of daytime traffic) → READY 13:06:17 = **4 min 02 s downtime**; weights 18 s (page cache), `torch.compile` 39 s (one-off recompile after the
config change), pinning 720 GiB ≈ 60 s; `Shmem` 720 GB, `MemAvailable` 1425 → 696 GB; KV pool unchanged (1,813,657); 37 GB stored in the first 90 s of traffic.

**Every distinct combination of image / FP8 set / partition / MTP gets its own `CACHE` directory** (torch.compile + Triton cache); sharing
causes recompiles and can mix artifacts. Copying a cache directory from the testbed makes the first start fast (compile 4 s instead of 26 s).

Compile-cache lifecycle (observed on A and B, torch 2.11 AOT compile): the **first** start with a fresh cache compiles (~26 s) and saves the AOT
artifacts; the **second** start fails to load the backbone artifacts with `Compiling model again due to a load failure … Kernel index 1 not found in
id_to_kernel` (the bundled Triton kernel side table of the first save does not cover the indexer kernel `_fused_indexer_q_rope_quant_kernel`),
recompiles (~23 s) and re-saves; from the **third** start on, `Directly load AOT compilation` and `torch.compile took ~4 s`. Consequences: expect
one extra ~25 s start per new cache directory (harmless), and copy a cache to another host only after it has served **two** starts (the production
cache `$CACHE_ROOT/vllm-fi618-cand-mtp3` was copied after one start on B, hence the recompile at the 2026‑09‑18 22:52 UTC deployment; its
re-saved artifacts loaded directly at the 2026‑09‑19 09:49 UTC restart: `torch.compile took 4.21 s`, start 140 s, downtime 2 min 17 s).

Rollbacks, from smallest to largest: `CAND=1 KV_OFFLOAD_GB=0` (no RAM tier = production of 2026‑09‑20…22, same image, same cache) → `CAND=1 KV_OFFLOAD_GB=0 LONG_PREFILL_DYNAMIC=`
(static chunk 512) → `CAND=1 KV_OFFLOAD_GB=0 IMAGE=vllm-nightly-fi614:nvfp4-fi618 LONG_PREFILL_DYNAMIC=` (exact production of 2026‑09‑19, same cache) → `NCCL_MODE=p2p_sys MTP=5 DCP=2 ./serve-glm53-prod.sh` (original TP8 DCP2 MTP5, image `nvfp4`,
cache `$CACHE_ROOT/vllm`). Intermediate variants: `CAND=1 MTP=5 CACHE=$CACHE_ROOT/vllm-fi618-cand`, `CAND=1 NVFP4_MOE= CACHE=$CACHE_ROOT/vllm-fi618-cand-mtp3-noq`
(BF16 draft; KV pool 1.67M instead of 1.81M because the last pipeline stage becomes the memory bottleneck), `CAND=1 FP8_CHANNEL=0`, `CAND=1 PP_PARTITION=42,36` —
each with its own `CACHE`.

Expected startup log lines: `TQ_B1: online NVFP4 for model.layers.78.mlp.experts … rel. weight error 0.0949`, `TQ_FP8: lm_head -> FP8`,
`TQ_FP8: eh_proj -> FP8`, `Selected CutlassFP8ScaledMMLinearKernel`, `GPU KV cache size: 1,80x,xxx tokens`.

**KV offload to host RAM (D2) — in production on A since 2026‑09‑22 13:06 UTC with `KV_OFFLOAD_GB=668` (720 GiB pinned, 5.6M tokens = 3.1× the GPU pool), part of the `CAND=1` preset.** Gate passed on B the same day with the production configuration (RESULTS §12d: 200k reload TTFT 36 s → 0.46 s, 650k 187 s → 1.33 s, no prefill/decode/store overhead); the 2026‑09‑21 "hang" verdict was a watchdog artefact (RESULTS §12c). `KV_OFFLOAD_GB=<GiB>` adds
`--kv-transfer-config` with vLLM's native `OffloadingConnector` (CPU tier, LRU; `KV_OFFLOAD_OPTS='"eviction_policy":"arc","block_size":256'` for extra
fields). Completed prompt blocks are copied to pinned RAM as they are produced (120 KiB/token for this config), and a later request whose prefix was
evicted from the GPU reloads them instead of recomputing. Sizing rules: (1) the request is summed over all 8 workers and must exceed the GPU pool
(~200 GiB) to add any hit rate; (2) torch pins one tensor per layer rounded up to a power of two, so the real footprint is 1.1–2× the request — use the
measured points **334 GiB → 360 GiB pinned (2.8M tokens; B)** or **668 GiB → 720 GiB pinned (5.6M tokens; A)** for rc2 TP4×PP2 41/37; 400 GiB pinned
~700 GB on B and exhausted swap, and anything between 668 and 1336 rounds up to ~1440 GiB — do not use; (3) the script refuses to start when
`MemAvailable < 1.25×request + 100 GB` and prints `Shmem`/`MemAvailable` after `/health`. The pinned memory is never released while the container runs
(A: 1507 GB total → 696 GB available with the tier, the 433 GB of weights stay in the page cache); after a host reboot the `unless-stopped` container pins it again. Expected log lines: `Creating v1 connector with name: OffloadingConnector`, `KV offloading: EAGLE/MTP draft attention groups [0] detected`,
`Allocating 53 CPU tensors` (stage 0) / `48` (stage 1). Metrics: `vllm:kv_offload_store_bytes`, `vllm:kv_offload_load_bytes`,
`vllm:kv_offload_allocation_failure`. Startup takes ~2–3 min longer (pinning). Gate (**passed on B 2026‑09‑22**, RESULTS §12d): `bench/kv_offload_reload.py`
PASS at 200k (warm TTFT 36 s → 0.46 s) and 650k (187 s → 1.33 s), prefill and decode identical to the no-offload baseline. Before A: `MemAvailable` on A
≥ 1.25×request + 100 GB, a maintenance window (the container restarts), `verify-prod.sh`, then watch `vllm:kv_offload_load_bytes` / `external_prefix_cache_hits` on real traffic.

## 3. Health and monitoring

- `curl -s localhost:8000/health` → 200; `/v1/models` lists `glm-5.3` and the alias `glm-5.2`.
- `/metrics`: `vllm:num_requests_running|waiting`, `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_drafts_total`
  (acceptance = 1 + ratio; expect ≥2.8 with MTP3, ≥3.1 with MTP5), `vllm:num_preemptions_total` (expect 0), `vllm:request_success_total`.
- Engine log every 10 s: `Engine 000: Avg prompt throughput … Running: N reqs, Waiting: M reqs, GPU KV cache usage: X%`. Note: prompt
  throughput does **not** count tokens of a long chunked prefill — watch KV usage rising instead.
- `dmesg -T | grep -iE "xid|nvrm"` — must stay empty. `docker logs glm53-nvfp4-prod 2>&1 | grep -cE "ERROR|Traceback"` — 0.
- A hung engine looks like: worker CPUs idle, `WorkerAsyncOutputCopy` in `synchronize` → unmatched NCCL collective; diagnose with
  `py-spy dump --pid <host pid>` from the host and `VLLM_TQ_PP_DEBUG=1`.
- **Automated check (D6)**: `deploy/alert-prod.sh` from cron every minute (`* * * * * <repo>/deploy/alert-prod.sh`): container running, `/health`,
  `waiting > WAIT_MAX` in two consecutive checks, new preemptions, new Xid/NVRM lines, ERROR/Traceback in the last 2 minutes, MTP acceptance over
  the last `ACC_WINDOW` minutes below `ACC_MIN` (from `/metrics` deltas, ≥200 drafts). Findings go to `ALERT_LOG` and to `ALERT_CMD "<message>"`
  if set (webhook script); state in `/var/tmp/glm53-alert.state`. `VERBOSE=1` prints an OK line for manual runs.
- Per-hour acceptance and request counts from the engine log: `docker logs glm53-nvfp4-prod 2>&1 | deploy/acceptance_by_hour.py`.

## 4. Testbed (host B)

B runs the scripts from a checkout of this repository (the old script directory is a symlink to its `bench/`). Publish changes from the
development host with `git push testbed main` (remote configured with `receive.denyCurrentBranch=updateInstead`; the B working tree must have
no modified tracked files — never edit scripts on B). Host settings live in `bench/env` on B (`MODEL`, `CACHE_ROOT`, `RESULTS_DIR`, `PROF_DIR`,
`PATCHED_DIR`). Results produced on B (`$RESULTS_DIR`, `results/<tag>/` from `run_all.sh`) are copied back with `scp` and committed on the
development host.

```bash
cd bench
./run_cand.sh                                # candidate config, image nvfp4-fi618 (PART=41,37 MTP=3 by default)
IMAGE=glm53-stack:2026.09.19-rc1 FP8=fused_qkv_a_proj,q_b_proj,o_proj,shared_experts,indexer.wq_b,eh_proj ./run_cand.sh   # ablation (no lm_head)
FP8= NVFP4_MOE= ./run_cand.sh                # reference without online quantization (own compile cache is derived automatically)
KV_OFFLOAD_GB=334 IMAGE=glm53-stack:2026.09.19-rc2 ./run_cand.sh   # D2: native CPU KV tier (334 GiB request = 360 GiB pinned on B, see §2)
python3 kv_offload_reload.py --len 200000 --churn 4 --churn-len 550000 --out /tmp/reload.json   # D2 gate: cold → evict from GPU → reload from RAM
./run_b1s2.sh                                # same, but with patched python files bind-mounted (iterate without rebuilding the image)
python3 smoke.py glm-5.3 --conc 4
python3 bench_decode.py --conc 1,4,16,32 --tokens 256 --rounds 3      # ms/step independent of MTP acceptance (reads /metrics)
python3 bench_prefill.py --model glm-5.3 --lens 32768,131072,262144 --seed 900
python3 bench_fairness.py --seed 900
python3 parity.py --out /tmp/p.json --compare results/b1s2_parity.json    # greedy battery + needle 32k/131k
python3 lp_probe.py --out /tmp/lp.json --compare results/base_lp.json     # teacher-forced CE (texts ≥ ~500 tokens only)
python3 needle_len.py --lens 500000,650000 --fmt chat                    # long-context needle in isolation
TAG=soak ./soak.sh 75 75 45                                              # mixed-traffic soak, restart between segments, parity after each
python3 soak_status.py results/soak_seg*.jsonl
python3 gsm8k_eval.py --n 1319 --conc 8 --out ../results/gsm8k_<tag>.json   # quality gate (~15 min); n=500 has SE ±0.7 pp — use 1319 for decisions
python3 gsm8k_compare.py ../results/gsm8k_<ref>.json ../results/gsm8k_<tag>.json   # paired comparison (discordant pairs, exact McNemar p)
./run_all.sh <tag>                                                       # everything above → ../results/<tag>/summary.md
```

Benchmark pitfalls: the first round after changing concurrency is warm-up (ignore r0); `bench_half.py`-style repeated prompts hit the prefix
cache and are meaningless with MTP; `prompt_logprobs` for prompts shorter than ~450 tokens returns garbage on this vLLM (pre-existing bug).

## 5. Failure modes seen so far

| symptom | cause | action |
|---|---|---|
| container `Exited (0)` shortly after start | worker error swallowed | `docker logs … \| grep -E "ERROR\|Error"` |
| `CUDA driver initialization failed` at start | a GPU in "Reboot required" after GSP fault | `nvidia-smi`; reboot window |
| crash-loop with `Concurrent Partial Prefill is not supported` | `--max-num-partial-prefills` passed (unsupported); only `--long-prefill-token-threshold` works | drop the flag |
| garbage output after enabling custom AR | PCIe custom all-reduce with >2 GPUs | keep `VLLM_CUSTOM_AR_PCIE_MAX_SIZE=0` |
| KV pool much smaller than expected | wrong `VLLM_PP_LAYER_PARTITION` (stage-0 memory is the bottleneck) | 41/37 for this config; re-check after any FP8/NVFP4 set change |
| draft acceptance collapses (<2) | draft embeddings/lm_head not loaded under PP, or sync scheduling with PP | `patch_pp_mtp.py` handles both; PP+MTP requires `--async-scheduling` |
| `torch.AcceleratorError: CUDA error: unspecified launch failure` on one worker → `EngineDeadError`, container exits; kernel log: `pcieport … pciehp: Slot(N): Link Down` / `Card not present`, `NVRM: Xid (PCI:…): 79, GPU has fallen off the bus`, then thousands of `uvm encountered global fatal error 0x60, requiring os reboot to recover`; `nvidia-smi -L` lists one GPU less (indices shift!) | PCIe link loss of one GPU under load (seen twice 2026‑09‑21 on B, slots 17 and 65, both at **8×450 W** during normally progressing 550k prefills, 20–50 min before the office breakers tripped under the same load; 0 Xid at 300 W since, incl. ~25 SIGKILLs and hours of full-load prefill). Power delivery, not software (RESULTS §12c) | host reboot required (GPU reset impossible). B has a working BMC (`/dev/ipmi0`, `ipmitool`) → remote power cycle if the reboot hangs. After the reboot check `nvidia-smi -L` (9 devices on B) and the GPU index list in `bench/serve_full.sh` (`GPUS`); the team's `llm-stack.service` starts automatically. Keep B at 300 W (`nvidia-powerlimit.service`) |
| a long single request "stops": no `Engine 000:` status line for minutes, `Running: 1`, all 8 PRO GPUs at 100 % **and at the power cap (~300 W)** | **not a hang** — the status logger prints 1–2 lines in the first ~16 s of a long chunked prefill and then nothing until the request finishes (550k tokens ≈ 150 s at 300 W, 650k ≈ 190 s); same image and scheduler settings on A, so expect the same there. Misread as a hang on 2026‑09‑21 (RESULTS §12c) | wait for the expected prefill time (≈ tokens / 3.6k tok/s at 300 W). Judge by hardware: power at the cap = computing; 100 % utilisation at ~130 W on all cards for > 1 min with a request in flight = real NCCL spin (never observed so far). `bench/d2_rootcause/exp.sh` implements exactly this criterion (`gpu_trace.csv`, `gpu_trace_stalls.py`) |
| after `docker kill` / SIGKILL of a **busy** vLLM (prefill in flight) some GPUs stay at 100 % utilisation / ~130 W with no process; a new container cannot use them | teardown of contexts with in-flight P2P collectives / batched D2H copies wedges the GPUs on driver 610.43.02 (RESULTS §12c #5; ~20 power cycles on B on 2026‑09‑21 were this) | host power cycle (BMC on B; A has none). **Prevention: always `docker stop -t 60` (SIGTERM, graceful) — a finished or idle engine stops cleanly every time; never `docker kill`/`docker rm -f` a container that is serving.** `serve_full.sh`/`run_cand.sh` do `docker rm -f` of the previous container: stop it first if it may be busy |
| host swap full, `MemAvailable` tens of GB, OOM-killer active, worker RSS shrinking | `KV_OFFLOAD_GB` set to a value whose per-layer pinned tensors round up to the next power of two (e.g. 400 → ~700 GB) | stop the container, restart with a verified sweet spot (§2); never run other pinned allocations on the host while it is that full |
| workers hang (`shm_broadcast … No available shared memory broadcast block` every 60 s), then `EngineDeadError`; at teardown thousands of `NVRM: GPUx … Possible bad register read 0xbadf3200` / `NV_ERR_GPU_IN_FULLCHIP_RESET`; `nvidia-smi -q` shows `GPU requires reset`; vLLM processes become zombies holding VRAM on all GPUs | GPU fault (driver 595.71.05; seen 2026‑09‑19 on GPU 0 of A after 3 h of idle) | `nvidia-smi --gpu-reset` is **Not Supported** on these boards → host reboot. A graceful `systemctl reboot` **hung at "Sending SIGTERM to remaining processes"** for 26 h (zombies with GPU state) → use `echo b > /proc/sysrq-trigger` (or remote power control) once filesystems are synced. Do **not** `docker rm` the production container before the reboot: with `--restart unless-stopped` it comes back by itself; a removed one needs a manual start (137 s cold weight load). Driver upgraded to 610.43.02 on 2026‑09‑20 (D7): `cuda-keyring` → NVIDIA repo `ubuntu2604`, the exact 18-package set of the testbed pinned to `610.43.02-1ubuntu1` + `apt-mark hold`; Ubuntu's `*-595-server*` packages had to be purged first (`dpkg --purge --force-depends`, file conflict on `nvidia-cuda-mps-control`); DKMS module checked for the running kernel before the reboot; production auto-restarted after the reboot (cold start 4.5 min) |
