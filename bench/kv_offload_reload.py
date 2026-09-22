#!/usr/bin/env python3
"""D2 gate: KV reload from the CPU tier (vLLM OffloadingConnector) must be transparent.

Phases (all greedy, chat format like needle_len.py):
  1. cold   : needle prompt P (L tokens) -> answer A1, TTFT1 (full prefill), connector metrics
  2. churn  : N unique prompts of CH tokens each, max_tokens=1 (raw) -> evicts P from the GPU prefix cache
             (sum must exceed the GPU KV pool, 1.81M tokens)
  3. warm   : P again -> A2, TTFT2; expects A2 == A1, needle OK, kv_offload_load_bytes grew by ~L*bytes/token
  4. gpuhit : P again immediately -> TTFT3 (GPU prefix-cache hit) for reference
Usage: kv_offload_reload.py --len 200000 --churn 4 --churn-len 550000 [--seed 5] [--out file.json]
Requires a server started with KV_OFFLOAD_GB>0 (bench/run_cand.sh). Note: this stack is not bitwise deterministic even for a GPU
prefix-cache hit (gpuhit0 control), so the verdict relies on the needle answer + kv_offload_load_bytes, not on byte identity.
"""
import argparse, json, random, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="glm-5.3")
ap.add_argument("--len", type=int, default=200000); ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--churn", type=int, default=4); ap.add_argument("--churn-len", type=int, default=550000)
ap.add_argument("--max-tokens", type=int, default=400); ap.add_argument("--out", default=None)
ap.add_argument("--skip-cold", action="store_true", help="P is already cached (e.g. from a previous run): start with churn")
a = ap.parse_args()
WORDS = ["alfa", "beta", "gamma", "delta", "kappa", "sigma", "omega", "node", "graph", "tensor", "kernel", "cache", "token", "block", "layer", "expert", "router", "index", "query", "value"]


def post(path, body, timeout=3600):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metrics():
    with urllib.request.urlopen(a.url + "/metrics", timeout=30) as r:
        txt = r.read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for key in ("vllm:kv_offload_load_bytes", "vllm:kv_offload_store_bytes", "vllm:kv_offload_load_time", "vllm:kv_offload_store_time",
                    "vllm:prefix_cache_queries", "vllm:prefix_cache_hits", "vllm:external_prefix_cache_queries", "vllm:external_prefix_cache_hits",
                    "vllm:num_preemptions", "vllm:kv_offload_allocation_failure"):
            if line.startswith(key + "_total{") or line.startswith(key + "{") or line.startswith(key + " ") or line.startswith(key + "_total "):
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def filler(n_words, seed):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 9999)) for _ in range(n_words))


cal = post("/tokenize", {"model": a.model, "prompt": filler(2000, 1)})
tpw = cal["count"] / 2000


def needle_prompt(n_tokens, seed):
    words = filler(int(n_tokens / tpw), seed).split(" ")
    facts = [f"Kod dostepu do serwerowni to {seed * 7 + 11}-ZETA.", f"Ulubiony kolor Marka Kowalskiego to turkusowy numer {seed + 3}.",
             f"Pociag do Gdyni odjezdza o godzinie {6 + seed % 12}:{(seed * 13) % 60:02d}."]
    for frac, f in zip((0.1, 0.5, 0.9), facts):
        words.insert(int(len(words) * frac), " " + f + " ")
    q = "\n\nNa podstawie powyzszego tekstu odpowiedz krotko: (1) jaki jest kod dostepu do serwerowni? (2) jaki jest ulubiony kolor Marka Kowalskiego? (3) o ktorej odjezdza pociag do Gdyni? Odpowiedz w jednej linii."
    return " ".join(words) + q, [f"{seed * 7 + 11}-ZETA", "turkusowy", f"{6 + seed % 12}:{(seed * 13) % 60:02d}"]


def chat_stream(prompt, max_tokens):
    """Returns (ttft_s, total_s, text, prompt_tokens, completion_tokens)."""
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; parts = []; usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
    return ttft, time.perf_counter() - t0, "".join(parts), usage.get("prompt_tokens"), usage.get("completion_tokens")


def fmt_m(m):
    gb = lambda k: f"{m.get(k, 0) / 2**30:.2f}G"
    return (f"store={gb('vllm:kv_offload_store_bytes')} load={gb('vllm:kv_offload_load_bytes')} "
            f"pc_hits/q={m.get('vllm:prefix_cache_hits', 0):.0f}/{m.get('vllm:prefix_cache_queries', 0):.0f} "
            f"ext_hits/q={m.get('vllm:external_prefix_cache_hits', 0):.0f}/{m.get('vllm:external_prefix_cache_queries', 0):.0f} "
            f"preempt={m.get('vllm:num_preemptions', 0):.0f} alloc_fail={m.get('vllm:kv_offload_allocation_failure', 0):.0f}")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


seed = a.seed * 100 + a.len % 997
P, needles = needle_prompt(a.len, seed)
res = {"len": a.len, "seed": seed, "phases": {}}
m0 = metrics(); log(f"start L={a.len} churn={a.churn}x{a.churn_len} | {fmt_m(m0)}")

if not a.skip_cold:
    ttft1, tot1, txt1, pt1, ct1 = chat_stream(P, a.max_tokens)
    ok1 = all(x.lower() in txt1.lower() for x in needles)
    m1 = metrics()
    log(f"cold   : prompt={pt1} TTFT={ttft1:6.2f}s total={tot1:6.1f}s ({(pt1 or 0)/ttft1:6.0f} tok/s prefill) needle={'OK' if ok1 else 'NIE'} | {fmt_m(m1)}")
    res["phases"]["cold"] = {"ttft": ttft1, "total": tot1, "prompt_tokens": pt1, "completion_tokens": ct1, "needle": ok1, "text": txt1, "metrics": m1}
    time.sleep(3)  # let async stores drain
    m1b = metrics(); log(f"         after 3 s: {fmt_m(m1b)}  (store delta {(m1b.get('vllm:kv_offload_store_bytes',0)-m0.get('vllm:kv_offload_store_bytes',0))/2**30:.2f} GiB"
                         f" = {((m1b.get('vllm:kv_offload_store_bytes',0)-m0.get('vllm:kv_offload_store_bytes',0))/max(pt1 or 1,1))/1024:.1f} KiB/token)")
    # determinism control: same prompt again = GPU prefix-cache hit, same KV bits -> any difference is decode nondeterminism, not offload
    ttft0, tot0, txt0, pt0, ct0 = chat_stream(P, a.max_tokens)
    det = txt0 == txt1
    log(f"gpuhit0: prompt={pt0} TTFT={ttft0:6.2f}s total={tot0:6.1f}s identical_to_cold={det}  (intra-run determinism control)")
    res["phases"]["gpuhit0"] = {"ttft": ttft0, "total": tot0, "identical": det, "text": txt0}
else:
    txt1 = None; ttft1 = None; det = None


def common_prefix(x, y):
    k = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    return k

churn_tokens = 0
for i in range(a.churn):
    cp = filler(int(a.churn_len / tpw), 90000 + seed * 10 + i)
    t0 = time.perf_counter()
    r = post("/v1/completions", {"model": a.model, "prompt": cp, "max_tokens": 1, "temperature": 0.0})
    dt = time.perf_counter() - t0; pt = r.get("usage", {}).get("prompt_tokens", 0); churn_tokens += pt
    log(f"churn {i+1}/{a.churn}: prompt={pt} {dt:6.1f}s ({pt/dt:6.0f} tok/s) cum={churn_tokens}")
mc = metrics(); log(f"after churn ({churn_tokens} tokens): {fmt_m(mc)}")
res["phases"]["churn"] = {"tokens": churn_tokens, "metrics": mc}

ttft2, tot2, txt2, pt2, ct2 = chat_stream(P, a.max_tokens)
ok2 = all(x.lower() in txt2.lower() for x in needles)
m2 = metrics()
load_delta = m2.get("vllm:kv_offload_load_bytes", 0) - mc.get("vllm:kv_offload_load_bytes", 0)
same = (txt1 is None) or (txt2 == txt1)
cp = common_prefix(txt1, txt2) if txt1 is not None else None
log(f"warm   : prompt={pt2} TTFT={ttft2:6.2f}s total={tot2:6.1f}s needle={'OK' if ok2 else 'NIE'} identical_to_cold={same if txt1 is not None else 'n/a'}"
    f"{'' if cp is None else f' common_prefix={cp}/{len(txt1)}'} loaded={load_delta/2**30:.2f} GiB ({load_delta/max(pt2 or 1,1)/1024:.1f} KiB/token) | {fmt_m(m2)}")
res["phases"]["warm"] = {"ttft": ttft2, "total": tot2, "prompt_tokens": pt2, "completion_tokens": ct2, "needle": ok2, "identical": same, "common_prefix": cp, "load_bytes": load_delta, "text": txt2, "metrics": m2}
if txt1 is not None and not same:
    k = cp
    log(f"  DIVERGENCE at char {k}: cold={txt1[max(0,k-60):k+60]!r}\n                        warm={txt2[max(0,k-60):k+60]!r}")

ttft3, tot3, txt3, pt3, ct3 = chat_stream(P, a.max_tokens)
m3 = metrics()
log(f"gpuhit : prompt={pt3} TTFT={ttft3:6.2f}s total={tot3:6.1f}s identical_to_warm={txt3 == txt2} | {fmt_m(m3)}")
res["phases"]["gpuhit"] = {"ttft": ttft3, "total": tot3, "prompt_tokens": pt3, "identical": txt3 == txt2, "metrics": m3}

# PASS = needle answered from RAM-loaded KV, KV really came from the CPU tier, and the text matches cold unless the
# determinism control itself differs (then byte identity is not a valid criterion and we require needle + load only).
reasons = []
if not ok2:
    reasons.append("needle not answered after reload")
if load_delta <= 0:
    reasons.append("kv_offload_load_bytes did not grow (nothing came back from RAM: pool smaller than the churn, or prompt recomputed)")
if not same and det is not False:
    reasons.append("text differs from cold although the GPU-hit control was identical")
verdict = not reasons
log(f"VERDICT: {'PASS' if verdict else 'FAIL - ' + '; '.join(reasons)}{'' if det is not False else ' (decode nondeterministic within run: identity not required)'}"
    f"  (cold TTFT {ttft1 if ttft1 is None else round(ttft1,2)} s -> warm-from-RAM {ttft2:.2f} s -> GPU hit {ttft3:.2f} s)")
res["verdict"] = "PASS" if verdict else "FAIL"; res["fail_reasons"] = reasons
if a.out:
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
