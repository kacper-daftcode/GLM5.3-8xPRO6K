#!/usr/bin/env python3
"""Szybki test funkcjonalny + pomiar szybkosci dla serwera OpenAI-compatible (vLLM).
Uzycie: smoke.py <model-name> [--effort low|high|max|xhigh|medium] [--no-think]
"""
import json, sys, time, urllib.request, argparse, concurrent.futures as cf

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--effort", default=None, help="reasoning_effort (chat_template_kwargs)")
ap.add_argument("--no-think", action="store_true", help="chat_template_kwargs.enable_thinking=false (Qwen)")
ap.add_argument("--conc", type=int, default=4, help="rownoleglosc w tescie przepustowosci")
a = ap.parse_args()

def post(path, body, timeout=1200):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0

def chat(msgs, max_tokens, **extra):
    body = {"model": a.model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.6}
    ctk = {}
    if a.effort: ctk["reasoning_effort"] = a.effort
    if a.no_think: ctk["enable_thinking"] = False
    if ctk: body["chat_template_kwargs"] = ctk
    body.update(extra)
    return post("/v1/chat/completions", body)

def show(tag, resp, dt):
    ch = resp["choices"][0]; m = ch["message"]; u = resp.get("usage", {})
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    content = m.get("content") or ""
    ct = u.get("completion_tokens", 0)
    print(f"\n=== {tag} === {dt:.1f}s, prompt={u.get('prompt_tokens')} completion={ct} => {ct/dt:.1f} tok/s (calosc, z prefill), finish={ch.get('finish_reason')}")
    if reasoning: print(f"  [reasoning {len(reasoning)} zn.] {reasoning[:200].replace(chr(10),' ')}...")
    print(f"  [content {len(content)} zn.] {content[:400].replace(chr(10),' ')}")
    if m.get("tool_calls"): print("  [tool_calls]", json.dumps(m["tool_calls"])[:400])
    return ct, dt

print("== /v1/models ==")
models, _ = post("/v1/models", {}) if False else (json.loads(urllib.request.urlopen(a.url + "/v1/models", timeout=10).read()), 0)
print("  ", [(m["id"], m.get("max_model_len")) for m in models["data"]])

# 1) krotka odpowiedz
r, dt = chat([{"role": "user", "content": "Odpowiedz jednym słowem: jaka jest stolica Polski?"}], 256)
show("krotki QA", r, dt)

# 2) kodowanie + rozumowanie (mierzy decode tok/s przy dluzszej generacji)
r, dt = chat([{"role": "user", "content": "Napisz w Pythonie funkcję, która parsuje log nginx (combined format) i zwraca top 10 IP wg liczby żądań oraz łączny transfer w MB dla każdego. Krótko, bez długich wyjaśnień, tylko kod z docstringiem."}], 3000)
ct2, dt2 = show("kod (single stream)", r, dt)

# 3) tool calling
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Pogoda dla miasta", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}}, "required": ["city"]}}}]
try:
    r, dt = chat([{"role": "user", "content": "Jaka jest teraz pogoda w Krakowie w stopniach Celsjusza? Użyj narzędzia."}], 1500, tools=tools, tool_choice="auto")
    show("tool call", r, dt)
except Exception as e:
    print("\n=== tool call === BLAD:", str(e)[:300])

# 4) przepustowosc: N rownoleglych generacji po ~600 tok
def one(i):
    r, dt = chat([{"role": "user", "content": f"Opisz w ~300 słowach historię miasta numer {i} z listy: Gdańsk, Wrocław, Poznań, Łódź, Lublin, Szczecin, Toruń, Rzeszów. Wybierz miasto o tym numerze (licząc od 1)."}], 700)
    return r.get("usage", {}).get("completion_tokens", 0), dt
t0 = time.time()
with cf.ThreadPoolExecutor(a.conc) as ex:
    res = list(ex.map(one, range(1, a.conc + 1)))
wall = time.time() - t0
tot = sum(c for c, _ in res)
print(f"\n=== przepustowosc conc={a.conc} === {tot} tok w {wall:.1f}s => {tot/wall:.1f} tok/s lacznie, {tot/wall/a.conc:.1f} tok/s na strumien")
print(f"\nSINGLE-STREAM decode (test 2): {ct2} tok / {dt2:.1f}s = {ct2/dt2:.1f} tok/s")
