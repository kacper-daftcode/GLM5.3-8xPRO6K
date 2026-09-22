#!/usr/bin/env python3
"""Profil jednego dlugiego prefillu: unikalny prompt ~L tokenow, /start_profile ... /stop_profile.
Uzycie: prof_prefill.py [L=32768] [--decode-bg]  (--decode-bg: rownolegle jeden request dekodujacy, jak w bench_fairness)
"""
import json, random, sys, threading, time, urllib.request

L = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 32768
DECODE_BG = "--decode-bg" in sys.argv


def post(path, body=None, timeout=900):
    req = urllib.request.Request("http://127.0.0.1:8000" + path, data=json.dumps(body).encode() if body is not None else b"",
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout).read()


WORDS = ["alfa", "beta", "gamma", "delta", "kappa", "sigma", "omega", "node", "graph", "tensor", "kernel", "cache", "token", "block", "layer", "expert", "router", "index", "query", "value"]
rnd = random.Random(int(time.time()))
cal = json.loads(post("/tokenize", {"model": "glm-5.3", "prompt": " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 9999)) for _ in range(2000))}))["count"] / 2000
p = " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 9999)) for _ in range(int(L / cal)))
post("/v1/completions", {"model": "glm-5.3", "prompt": "rozgrzewka " + p[:2000], "max_tokens": 1})
if DECODE_BG:
    th = threading.Thread(target=lambda: post("/v1/chat/completions", {"model": "glm-5.3", "messages": [{"role": "user", "content": "Napisz obszerny esej o historii Krakowa."}],
                                                                       "max_tokens": 3000, "temperature": 0.0, "ignore_eos": True}))
    th.start(); time.sleep(2)
post("/start_profile"); t0 = time.perf_counter()
r = json.loads(post("/v1/completions", {"model": "glm-5.3", "prompt": p, "max_tokens": 1, "temperature": 0}))
dt = time.perf_counter() - t0; post("/stop_profile")
n = r["usage"]["prompt_tokens"]
print(f"prefill {n} tok w {dt:.2f} s => {n / dt:.0f} tok/s" + (" (z dekodowaniem w tle)" if DECODE_BG else ""))
