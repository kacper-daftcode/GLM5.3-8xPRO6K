#!/usr/bin/env python3
"""Needle-in-haystack dla zadanych dlugosci kontekstu (izolacja problemow z dlugim kontekstem).
Uzycie: needle_len.py --lens 500000,650000 [--fmt raw|chat|both] [--seed 5] [--model glm-5.3]
"""
import argparse, json, random, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="glm-5.3")
ap.add_argument("--lens", default="524288,650000"); ap.add_argument("--fmt", default="both"); ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--max-tokens", type=int, default=96)
a = ap.parse_args()
WORDS = ["alfa", "beta", "gamma", "delta", "kappa", "sigma", "omega", "node", "graph", "tensor", "kernel", "cache", "token", "block", "layer", "expert", "router", "index", "query", "value"]


def post(path, body, timeout=3600):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


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


for L in [int(x) for x in a.lens.split(",")]:
    for fmt in (["raw", "chat"] if a.fmt == "both" else [a.fmt]):
        seed = a.seed * 100 + L % 997
        p, nd = needle_prompt(L, seed)
        t0 = time.perf_counter()
        try:
            if fmt == "raw":
                r = post("/v1/completions", {"model": a.model, "prompt": p, "max_tokens": a.max_tokens, "temperature": 0.0})
                txt = r["choices"][0]["text"]
            else:
                r = post("/v1/chat/completions", {"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": max(a.max_tokens, 400), "temperature": 0.0})
                m = r["choices"][0]["message"]
                txt = (m.get("reasoning_content") or m.get("reasoning") or "") + "\n" + (m.get("content") or "")
            dt = time.perf_counter() - t0
            u = r.get("usage", {})
            ok = all(x.lower() in txt.lower() for x in nd)
            print(f"[{time.strftime('%H:%M:%S')}] {fmt:>4} L={L:>7} prompt={u.get('prompt_tokens')} {dt:6.1f}s ({(u.get('prompt_tokens') or 0)/dt:5.0f} tok/s) needle={'OK ' if ok else 'NIE'} expect={nd} -> {txt[:140]!r}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{time.strftime('%H:%M:%S')}] {fmt:>4} L={L:>7} ERROR {e!r}", flush=True)
