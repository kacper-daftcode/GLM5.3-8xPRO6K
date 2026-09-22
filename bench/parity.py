#!/usr/bin/env python3
"""Greedy parity test: staly zestaw promptow (temperature 0, sekwencyjnie) -> JSON z pelnymi wyjsciami.
Z --compare REF.json porownuje z referencja (identycznosc, pozycja pierwszej roznicy, needle OK/NIE).
Uzycie: parity.py --out base.json [--model glm-5.3] [--long 32768,131072] [--compare base.json]
"""
import argparse, json, random, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="glm-5.3")
ap.add_argument("--out", required=True)
ap.add_argument("--compare", default=None)
ap.add_argument("--long", default="32768,131072", help="dlugosci kontekstu dla testu needle (tokeny)")
a = ap.parse_args()


def post(path, body, timeout=3600):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.perf_counter() - t0


def chat(name, msgs, max_tokens, needle=None, **extra):
    body = {"model": a.model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.0, "seed": 1}
    body.update(extra)
    r, dt = post("/v1/chat/completions", body)
    m = r["choices"][0]["message"]
    txt = (m.get("reasoning_content") or m.get("reasoning") or "") + "\n<<<CONTENT>>>\n" + (m.get("content") or "")
    if m.get("tool_calls"):
        txt += "\n<<<TOOLS>>>\n" + json.dumps(m["tool_calls"], sort_keys=True)
    u = r.get("usage", {})
    return dict(name=name, kind="chat", text=txt, prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                secs=round(dt, 2), needle=needle, finish=r["choices"][0].get("finish_reason"))


def compl(name, prompt, max_tokens, needle=None):
    r, dt = post("/v1/completions", {"model": a.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "seed": 1})
    u = r.get("usage", {})
    return dict(name=name, kind="completion", text=r["choices"][0]["text"], prompt_tokens=u.get("prompt_tokens"),
                completion_tokens=u.get("completion_tokens"), secs=round(dt, 2), needle=needle, finish=r["choices"][0].get("finish_reason"))


WORDS = ["alfa", "beta", "gamma", "delta", "kappa", "sigma", "omega", "node", "graph", "tensor", "kernel", "cache", "token", "block", "layer", "expert", "router", "index", "query", "value"]


def filler(n_words, seed):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 9999)) for _ in range(n_words))


def needle_prompt(n_tokens, seed):
    """Filler ~n_tokens z 3 faktami wstawionymi na 10%/50%/90%; pytanie o fakt srodkowy i pierwszy."""
    cal, _ = post("/tokenize", {"model": a.model, "prompt": filler(2000, 1)})
    tpw = cal["count"] / 2000
    n_words = int(n_tokens / tpw)
    words = filler(n_words, seed).split(" ")
    facts = [f"Kod dostepu do serwerowni to {seed * 7 + 11}-ZETA.", f"Ulubiony kolor Marka Kowalskiego to turkusowy numer {seed + 3}.",
             f"Pociag do Gdyni odjezdza o godzinie {6 + seed % 12}:{(seed * 13) % 60:02d}."]
    for frac, f in zip((0.1, 0.5, 0.9), facts):
        words.insert(int(len(words) * frac), " " + f + " ")
    text = " ".join(words)
    q = "\n\nNa podstawie powyzszego tekstu odpowiedz krotko: (1) jaki jest kod dostepu do serwerowni? (2) jaki jest ulubiony kolor Marka Kowalskiego? (3) o ktorej odjezdza pociag do Gdyni? Odpowiedz w jednej linii."
    return text + q, [f"{seed * 7 + 11}-ZETA", f"turkusowy", f"{6 + seed % 12}:{(seed * 13) % 60:02d}"]


results = []
t_all = time.perf_counter()
results.append(chat("qa_short", [{"role": "user", "content": "Odpowiedz jednym słowem: jaka jest stolica Polski?"}], 64, needle=["Warszawa"]))
results.append(chat("code_py", [{"role": "user", "content": "Napisz w Pythonie funkcję, która parsuje log nginx (combined format) i zwraca top 10 IP wg liczby żądań oraz łączny transfer w MB dla każdego. Tylko kod z docstringiem."}], 900))
results.append(chat("math", [{"role": "user", "content": "Ile wynosi suma wszystkich liczb pierwszych mniejszych od 100? Policz krok po kroku i podaj wynik."}], 1200, needle=["1060"]))  # 700 bylo na granicy (rozumowanie ucinane przed wynikiem)
results.append(chat("pl_text", [{"role": "user", "content": "Opisz w ~200 słowach historię Torunia."}], 450))
results.append(compl("compl_det", "Definicja: Model językowy to", 300))
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Pogoda dla miasta", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}}, "required": ["city"]}}}]
results.append(chat("tool_call", [{"role": "user", "content": "Jaka jest teraz pogoda w Krakowie w stopniach Celsjusza? Użyj narzędzia."}], 600, needle=["get_weather", "Krak"], tools=tools, tool_choice="auto"))
for L in [int(x) for x in a.long.split(",") if x]:
    p, nd = needle_prompt(L, seed=L % 1000 + 7)
    # chat (szablon + reasoning) zamiast surowego /v1/completions: raw przy >=131k bywa niedeterministycznie "kontynuowany"
    # zamiast odpowiedzi (near-tie pierwszego tokenu + niedeterminizm kerneli) - fakty sprawdzamy w reasoning+content
    results.append(chat(f"needle_{L}", [{"role": "user", "content": p}], 600, needle=nd))
    print(f"[{time.strftime('%H:%M:%S')}] needle {L}: {results[-1]['prompt_tokens']} tok, {results[-1]['secs']} s -> {results[-1]['text'][:160]!r}", flush=True)

for r in results:
    r["needle_ok"] = None if not r.get("needle") else all(n.lower() in r["text"].lower() for n in r["needle"])
json.dump(results, open(a.out, "w"), ensure_ascii=False, indent=1)
print(f"zapisano {a.out}; {len(results)} testow w {time.perf_counter() - t_all:.0f} s")
for r in results:
    print(f"  {r['name']:>14}: prompt={r['prompt_tokens']:>7} compl={r['completion_tokens']:>4} {r['secs']:>7.1f}s finish={r['finish']} needle={r['needle_ok']}")

if a.compare:
    ref = {r["name"]: r for r in json.load(open(a.compare))}
    print(f"\n=== porownanie z {a.compare} ===")
    for r in results:
        b = ref.get(r["name"])
        if not b:
            print(f"  {r['name']:>14}: brak w referencji"); continue
        x, y = r["text"], b["text"]
        if x == y:
            print(f"  {r['name']:>14}: IDENTYCZNE ({len(x)} zn.)")
        else:
            i = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            print(f"  {r['name']:>14}: ROZNE od znaku {i}/{len(y)} ({100 * i / max(1, len(y)):.0f}% wspolne); needle ref={b.get('needle_ok')} new={r['needle_ok']}")
            print(f"{'':>16}ref: ...{y[max(0, i - 40):i + 60]!r}")
            print(f"{'':>16}new: ...{x[max(0, i - 40):i + 60]!r}")
