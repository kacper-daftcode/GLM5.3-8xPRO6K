#!/usr/bin/env python3
"""Fairness: co dzieje sie z krotkimi requestami podczas dlugiego prefillu.
Faza 1: dlugi prompt (--long tok, unikalny) z max_tokens=1 w tle. Po --delay s: krotki chat (stream) -> TTFT, ITL, tok/s.
Rownolegle --others krotkich chatow wystartowanych PRZED dlugim (juz w decode) -> ich tok/s w trakcie prefillu (D3: prog dynamiczny wg liczby requestow).
Uzycie: bench_fairness.py [--long 131072] [--short-tokens 200] [--others 1] [--model glm-5.3] [--seed N]
"""
import argparse, json, random, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="glm-5.3")
ap.add_argument("--long", type=int, default=131072); ap.add_argument("--short-tokens", type=int, default=200)
ap.add_argument("--delay", type=float, default=3.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--others", type=int, default=1, help="number of concurrent decoding requests started before the long prefill")
a = ap.parse_args()
WORDS = ["alfa", "beta", "gamma", "delta", "kappa", "sigma", "omega", "node", "graph", "tensor", "kernel", "cache", "token", "block", "layer", "expert", "router", "index", "query", "value"]


def post(path, body, timeout=3600):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(content, max_tokens, tag, t_start_ref):
    """Zwraca dict: ttft, itl list, tokens, wall, timeline [(t, n_tokens)]."""
    t0 = time.perf_counter()
    r = post("/v1/chat/completions", {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}})
    times = []; ntok = None
    for line in r:
        if not line.startswith(b"data:") or b"[DONE]" in line:
            continue
        j = json.loads(line[5:])
        if j.get("usage") and j["usage"].get("completion_tokens"):
            ntok = j["usage"]["completion_tokens"]
        if not j.get("choices"):
            continue
        d = j["choices"][0].get("delta", {})
        if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
            times.append(time.perf_counter())
    # kazdy chunk = zwykle 1 token (przy MTP moze byc kilka tokenow w chunku -> ITL liczymy per chunk, tok/s z usage)
    itl = [b - x for x, b in zip(times, times[1:])]
    return {"tag": tag, "ttft": (times[0] - t0) if times else None, "chunks": len(times), "tokens": ntok, "wall": time.perf_counter() - t0,
            "itl_p50": sorted(itl)[len(itl) // 2] if itl else None, "itl_p95": sorted(itl)[int(len(itl) * 0.95)] if itl else None,
            "t_first": (times[0] - t_start_ref) if times else None, "t_last": (times[-1] - t_start_ref) if times else None, "times": [t - t_start_ref for t in times]}


rnd = random.Random(4242 + a.seed)
cal = json.loads(post("/tokenize", {"model": a.model, "prompt": " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 9999)) for _ in range(2000))}).read())["count"]
tpw = cal / 2000
long_prompt = " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 9999)) for _ in range(int(a.long / tpw)))

res = {}
T0 = time.perf_counter()
# (0) requesty "stare": decode juz trwa, gdy wchodzi dlugi prefill (--others rownoleglych; rozne prompty, zeby nie trafic w prefix cache)
TOPICS = ["historii Krakowa", "historii Gdańska", "historii Wrocławia", "historii Poznania", "historii Lublina", "historii Toruniu", "historii Szczecina", "historii Łodzi",
          "historii Katowic", "historii Rzeszowa", "historii Olsztyna", "historii Kielc", "historii Opola", "historii Bydgoszczy", "historii Zielonej Góry", "historii Białegostoku"]
old_threads = []
for i in range(a.others):
    tag = "old" if i == 0 else f"old{i}"
    th = threading.Thread(target=lambda tag=tag, i=i: res.__setitem__(tag, stream_chat(f"Napisz obszerny esej (1500 słów) o {TOPICS[i % len(TOPICS)]} (wariant {i}).", a.short_tokens + 400, tag, T0)))
    th.start(); old_threads.append(th)
time.sleep(2.0)
# (1) dlugi prefill
long_res = {}
def run_long():
    t = time.perf_counter(); r = post("/v1/completions", {"model": a.model, "prompt": long_prompt, "max_tokens": 1, "temperature": 0.0}); j = json.loads(r.read())
    long_res.update(ttft=time.perf_counter() - t, tokens=j["usage"]["prompt_tokens"], t_start=t - T0, t_end=time.perf_counter() - T0)
th_long = threading.Thread(target=run_long); th_long.start()
time.sleep(a.delay)
# (2) nowy krotki request w trakcie prefillu
res["new"] = stream_chat("Wyjaśnij w kilku zdaniach, czym jest fotosynteza.", a.short_tokens, "new", T0)
th_long.join()
for th in old_threads:
    th.join()

print(f"dlugi prefill: {long_res['tokens']} tok, TTFT {long_res['ttft']:.1f} s => {long_res['tokens']/long_res['ttft']:.0f} tok/s (okno [{long_res['t_start']:.1f}, {long_res['t_end']:.1f}] s); others={a.others}")
if a.others > 1:  # zbiorczo dla wszystkich "starych" dekoderow: srednia tok/s per strumien w trakcie prefillu
    rates = []
    for i in range(a.others):
        r = res["old" if i == 0 else f"old{i}"]
        inside = [t for t in r["times"] if long_res["t_start"] <= t <= long_res["t_end"]]
        tpc = (r["tokens"] / r["chunks"]) if r.get("tokens") and r["chunks"] else float("nan")
        rates.append(((len(inside) - 1) / (inside[-1] - inside[0]) if len(inside) > 2 else 0.0) * tpc)
    print(f"old x{a.others} W TRAKCIE prefillu: srednio {sum(rates)/len(rates):.1f} tok/s per strumien (min {min(rates):.1f}, max {max(rates):.1f}), lacznie {sum(rates):.0f} tok/s")
for tag in ("old", "new"):
    r = res[tag]
    # tok/s (chunki) w oknie prefillu vs poza nim
    inside = [t for t in r["times"] if long_res["t_start"] <= t <= long_res["t_end"]]
    outside = [t for t in r["times"] if t > long_res["t_end"]]
    rate_in = (len(inside) - 1) / (inside[-1] - inside[0]) if len(inside) > 2 else 0.0
    rate_out = (len(outside) - 1) / (outside[-1] - outside[0]) if len(outside) > 2 else float("nan")
    tpc = (r["tokens"] / r["chunks"]) if r.get("tokens") and r["chunks"] else float("nan")  # tokenow na chunk (~akceptacja MTP)
    print(f"{tag:>4}: TTFT {r['ttft']:.2f} s (start {r['t_first'] - r['ttft']:.1f} s), {r['tokens']} tok / {r['chunks']} chunkow ({tpc:.2f} tok/chunk), ITL p50 {r['itl_p50']*1000:.0f} ms p95 {r['itl_p95']*1000:.0f} ms | "
          f"W TRAKCIE prefillu: {rate_in:.1f} chunk/s ~ {rate_in*tpc:.1f} tok/s ({len(inside)} chunkow) | po prefillu: {rate_out:.1f} chunk/s ~ {rate_out*tpc:.1f} tok/s")
