#!/usr/bin/env python3
"""Prefill throughput vs dlugosc kontekstu: unikalny (niecache'owalny) prompt o ~N tokenach, max_tokens=1, mierzy TTFT.
Uzycie: bench_prefill.py [--url ...] [--lens 8192,32768,131072] [--model half]
"""
import json, time, argparse, random, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--lens", default="8192,32768,131072")
ap.add_argument("--model", default="half"); ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--seed", type=int, default=0, help="offset seeda (unikalne prompty, omija prefix cache z poprzednich przebiegow)")
a = ap.parse_args()
def post(path, body, timeout=3600):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
WORDS = ["alfa","beta","gamma","delta","kappa","sigma","omega","node","graph","tensor","kernel","cache","token","block","layer","expert","router","index","query","value"]
def prompt_of(n_words, seed):
    rnd = random.Random(seed); return " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 9999)) for _ in range(n_words))
# kalibracja tokenow/slowo
cal = post("/tokenize", {"model": a.model, "prompt": prompt_of(2000, 1)})["count"]; tpw = cal / 2000
print(f"kalibracja: {tpw:.2f} tok/slowo")
for L in [int(x) for x in a.lens.split(",")]:
    for r in range(a.repeat):
        p = prompt_of(int(L / tpw), 1000 + L + r + a.seed)
        t0 = time.perf_counter()
        out = post("/v1/completions", {"model": a.model, "prompt": p, "max_tokens": 1, "temperature": 0.0})
        dt = time.perf_counter() - t0
        n = out["usage"]["prompt_tokens"]
        print(f"ctx={n:>7} tok: TTFT {dt:7.1f} s => {n/dt:8.0f} tok/s prefill")
