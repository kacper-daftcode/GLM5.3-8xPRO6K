#!/usr/bin/env python3
"""Decode A/B odporny na MTP: N rownoleglych, ROZNYCH naturalnych promptow (chat, temperature 0, max_tokens stale,
ignore_eos), a z /metrics liczy liczbe krokow (drafts) => ms/krok NIEZALEZNE od akceptacji MTP, plus tok/s i akceptacje.
Uzycie: bench_decode.py [--conc 1,4,16] [--tokens 256] [--rounds 2] [--model glm-5.3]
"""
import argparse, json, re, time, urllib.request, concurrent.futures as cf

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="glm-5.3")
ap.add_argument("--conc", default="1,4,16"); ap.add_argument("--tokens", type=int, default=256); ap.add_argument("--rounds", type=int, default=2)
a = ap.parse_args()

TOPICS = ["historia Gdańska", "jak działa silnik odrzutowy", "przepis na bigos", "zasady gry w szachy", "fotosynteza", "budowa komputera kwantowego",
          "życie Marii Skłodowskiej-Curie", "jak powstaje tęcza", "ekonomia behawioralna", "architektura gotycka", "trening maratoński",
          "uprawa pomidorów", "protokół TCP/IP", "muzyka barokowa", "wulkany Islandii", "sztuczna inteligencja w medycynie",
          "historia druku", "prawo Ohma", "kuchnia japońska", "migracje ptaków", "teoria względności", "energetyka jądrowa",
          "system solarny", "język esperanto", "rozwój kolei w XIX wieku", "nawigacja gwiezdna", "produkcja sera", "ekologia lasów tropikalnych",
          "historia matematyki", "budowa mostów", "psychologia snu", "rzemiosło szklarskie"]


def post(path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(a.url + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def metrics():
    m = post("/metrics")
    def g(name):
        v = re.findall(rf"^vllm:{name}\{{[^}}]*\}} ([0-9.e+]+)$", m, flags=re.M)
        return sum(float(x) for x in v)
    return {"drafts": g("spec_decode_num_drafts_total"), "accepted": g("spec_decode_num_accepted_tokens_total"), "gen": g("generation_tokens_total")}


def gen(i):
    t0 = time.perf_counter()
    r = json.loads(post("/v1/chat/completions", {"model": a.model, "messages": [{"role": "user", "content": f"Napisz obszerny esej (co najmniej 1500 słów) na temat: {TOPICS[i % len(TOPICS)]}."}],
                                                "max_tokens": a.tokens, "temperature": 0.0, "ignore_eos": True}))
    return r["usage"]["completion_tokens"], time.perf_counter() - t0


gen(0)
for c in [int(x) for x in a.conc.split(",")]:
    for rnd in range(a.rounds):
        m0 = metrics(); t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(c) as ex:
            res = list(ex.map(gen, range(rnd * c, rnd * c + c)))
        wall = time.perf_counter() - t0; m1 = metrics()
        tot = sum(r[0] for r in res)
        drafts = m1["drafts"] - m0["drafts"]; acc = m1["accepted"] - m0["accepted"]
        # kroki silnika ~= drafts/c (kazdy krok = 1 draft na sekwencje); akceptacja = 1 + acc/drafts
        # bez MTP (drafts=0): 1 token = 1 krok
        steps = drafts / c if drafts else tot / c; acc_len = 1 + acc / drafts if drafts else 1.0
        print(f"conc={c:>3} gen={a.tokens} r{rnd}: {tot/wall:7.1f} tok/s lacznie | {wall/steps*1000:6.1f} ms/krok | akceptacja {acc_len:.2f} tok/krok | {wall:.1f} s", flush=True)
