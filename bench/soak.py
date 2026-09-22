#!/usr/bin/env python3
"""Soak mieszanego ruchu dla GLM-5.3 (kandydat prod). Rownolegle:
  - chat workers: eseje/kod/QA/multi-turn, losowe max_tokens i temperatura (walidacja finish_reason, niepuste wyjscie),
  - tool workers: tool-call (walidacja, ze model wywoluje narzedzie),
  - long worker(s): needle-in-haystack 131k/262k/500k/700k (unikalne seedy -> bez prefix cache; walidacja odpowiedzi),
  - burst: co N minut 24-32 rownoleglych krotkich requestow (max_num_seqs),
  - monitor co 60 s: /metrics (running/waiting/KV%/akceptacja MTP), nvidia-smi (pamiec/moc/temp), dmesg (Xid/NVRM), bledy w docker logs.
Log JSONL (--log) + podsumowanie na koniec (kod wyjscia 1 przy bledach/zawieszeniach/needle NIE).
Uzycie: soak.py --minutes 90 --log results/soak_seg1.jsonl [--chat 12 --tools 2 --long 1 --burst-every 15]
"""
import argparse, json, random, re, subprocess, threading, time, urllib.request, urllib.error, collections

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="glm-5.3")
ap.add_argument("--minutes", type=float, default=60); ap.add_argument("--log", required=True)
ap.add_argument("--chat", type=int, default=12); ap.add_argument("--tools", type=int, default=2); ap.add_argument("--long", type=int, default=1)
ap.add_argument("--long-lens", default="131072,131072,262144,500000,650000", help="tokeny (max-model-len 750k; kalibracja tok/slowo +-5%)")
ap.add_argument("--burst-every", type=float, default=15, help="minuty; 0 = bez burstow"); ap.add_argument("--burst-size", type=int, default=28)
ap.add_argument("--container", default="full-glm"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

T_END = time.time() + a.minutes * 60
LOCK = threading.Lock()
LOG = open(a.log, "a")
STATS = collections.Counter()
LAT = collections.defaultdict(list)
STOP = threading.Event()
RNG = random.Random(a.seed or int(time.time()))


def log(kind, **kw):
    kw.update(kind=kind, t=round(time.time(), 1), ts=time.strftime("%H:%M:%S"))
    with LOCK:
        LOG.write(json.dumps(kw, ensure_ascii=False) + "\n"); LOG.flush()


def post(path, body=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(a.url + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def chat(msgs, max_tokens, temperature, name, timeout=1800, **extra):
    body = {"model": a.model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature}
    body.update(extra)
    t0 = time.perf_counter()
    try:
        r = json.loads(post("/v1/chat/completions", body, timeout=timeout))
        dt = time.perf_counter() - t0
        ch = r["choices"][0]; m = ch["message"]; u = r.get("usage", {})
        txt = (m.get("content") or "") + (m.get("reasoning_content") or m.get("reasoning") or "")
        ok = bool(txt.strip()) or bool(m.get("tool_calls"))
        with LOCK:
            STATS[name + "_ok" if ok else name + "_empty"] += 1
            LAT[name].append(dt)
        return dict(ok=ok, secs=round(dt, 2), prompt=u.get("prompt_tokens"), compl=u.get("completion_tokens"), finish=ch.get("finish_reason"),
                    text=txt, tool_calls=m.get("tool_calls"))
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t0
        with LOCK:
            STATS[name + "_err"] += 1
        log("error", worker=name, secs=round(dt, 1), err=repr(e)[:300])
        return dict(ok=False, secs=round(dt, 2), err=repr(e)[:300])


TOPICS = ["historia Gdańska", "silnik odrzutowy", "przepis na bigos", "zasady szachów", "fotosynteza", "komputery kwantowe", "Maria Skłodowska-Curie",
          "tęcza", "ekonomia behawioralna", "gotyk", "maraton", "uprawa pomidorów", "TCP/IP", "barok", "wulkany Islandii", "AI w medycynie", "druk",
          "prawo Ohma", "kuchnia japońska", "migracje ptaków", "teoria względności", "energetyka jądrowa", "Układ Słoneczny", "esperanto", "kolej XIX w.",
          "nawigacja gwiezdna", "produkcja sera", "lasy tropikalne", "historia matematyki", "mosty", "psychologia snu", "szkło", "Bałtyk", "Kraków", "kawa"]
CODE_TASKS = ["parser logów nginx z top 10 IP", "LRU cache z TTL", "klient REST z retry i backoff", "walidator JSON Schema (podzbiór)", "serwer echo asyncio",
              "algorytm Dijkstry na grafie z listą sąsiedztwa", "konwerter CSV->Parquet z pyarrow", "rate limiter token bucket", "CLI do zmiany nazw plików wg regex",
              "prosty interpreter wyrażeń arytmetycznych"]
LANGS = ["Python", "Go", "Rust", "TypeScript", "bash"]
QA = ["Wymień 5 największych miast Polski wg liczby mieszkańców.", "Wyjaśnij różnicę między TCP i UDP w 3 zdaniach.", "Ile to 17*23? Podaj tylko wynik.",
      "Przetłumacz na angielski: 'Pociąg do Gdyni odjeżdża o siódmej.'", "Podaj wzór na pole koła i oblicz dla r=3.", "Co to jest indeks B-drzewa? Krótko.",
      "Napisz haiku o jesieni.", "Jakie są trzy prawa Newtona? Jedno zdanie każde."]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Pogoda dla miasta", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "search_docs", "description": "Szukaj w dokumentacji", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}},
         {"type": "function", "function": {"name": "create_ticket", "description": "Utworz zgloszenie", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "normal", "high"]}, "body": {"type": "string"}}, "required": ["title", "body"]}}}]
TOOL_PROMPTS = [("Jaka jest pogoda w Krakowie w stopniach Celsjusza? Użyj narzędzia.", "get_weather"), ("Znajdź w dokumentacji informacje o limitach API (max 5 wyników).", "search_docs"),
                ("Utwórz zgłoszenie o wysokim priorytecie: serwer bazy danych nie odpowiada od 10 minut.", "create_ticket"), ("Sprawdź pogodę w Gdańsku i w Warszawie.", "get_weather")]
WORDS = ["alfa", "beta", "gamma", "delta", "kappa", "sigma", "omega", "node", "graph", "tensor", "kernel", "cache", "token", "block", "layer", "expert", "router", "index", "query", "value"]


def chat_worker(i):
    rnd = random.Random(1000 + i + (a.seed or 0))
    hist = None
    while not STOP.is_set() and time.time() < T_END:
        kind = rnd.choices(["essay", "code", "qa", "multi"], weights=[4, 3, 3, 2])[0]
        temp = rnd.choice([0.0, 0.6, 0.8, 1.0])
        if kind == "essay":
            mt = rnd.randint(300, 1500)
            r = chat([{"role": "user", "content": f"Napisz esej (~{mt//2} słów) na temat: {rnd.choice(TOPICS)}. Numer: {rnd.randint(0, 10**6)}."}], mt, temp, "chat")
        elif kind == "code":
            mt = rnd.randint(400, 1500)
            r = chat([{"role": "user", "content": f"Napisz w {rnd.choice(LANGS)}: {rnd.choice(CODE_TASKS)}. Tylko kod z krótkim komentarzem. Wariant {rnd.randint(0, 999)}."}], mt, temp, "chat")
        elif kind == "qa":
            r = chat([{"role": "user", "content": rnd.choice(QA)}], rnd.randint(64, 400), temp, "chat")
        else:
            if not hist:
                hist = [{"role": "user", "content": f"Zaplanuj 3-dniową wycieczkę do miasta: {rnd.choice(TOPICS)}. Krótko, punktami."}]
            r = chat(hist, rnd.randint(150, 600), temp, "chat")
            if r.get("ok") and r.get("text"):
                hist = hist + [{"role": "assistant", "content": r["text"][:2000]}, {"role": "user", "content": rnd.choice(["Skróć to o połowę.", "Dodaj koszty w PLN.", "Zamień dzień 2 na coś z kulturą.", "Podsumuj w jednym zdaniu."])}]
                if len(hist) > 8:
                    hist = None
        if r.get("ok") and r.get("finish") not in ("stop", "length", "tool_calls"):
            log("warn", worker="chat", finish=r.get("finish"))
        time.sleep(rnd.uniform(0.2, 2.0))


def tool_worker(i):
    rnd = random.Random(2000 + i + (a.seed or 0))
    while not STOP.is_set() and time.time() < T_END:
        p, expect = rnd.choice(TOOL_PROMPTS)
        r = chat([{"role": "user", "content": p}], 500, rnd.choice([0.0, 0.5]), "tool", tools=TOOLS, tool_choice="auto")
        if r.get("ok"):
            called = json.dumps(r.get("tool_calls") or "")
            with LOCK:
                STATS["tool_called" if expect in called else "tool_notcalled"] += 1
            if expect not in called:
                log("warn", worker="tool", prompt=p, tool_calls=called[:300], text=(r.get("text") or "")[:200])
        time.sleep(rnd.uniform(1, 4))


def filler(n_words, seed):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 9999)) for _ in range(n_words))


_TPW = None


def needle_prompt(n_tokens, seed):
    global _TPW
    if _TPW is None:
        cal = json.loads(post("/tokenize", {"model": a.model, "prompt": filler(2000, 1)}))
        _TPW = cal["count"] / 2000
    words = filler(int(n_tokens / _TPW), seed).split(" ")
    facts = [f"Kod dostepu do serwerowni to {seed * 7 + 11}-ZETA.", f"Ulubiony kolor Marka Kowalskiego to turkusowy numer {seed + 3}.",
             f"Pociag do Gdyni odjezdza o godzinie {6 + seed % 12}:{(seed * 13) % 60:02d}."]
    for frac, f in zip((0.1, 0.5, 0.9), facts):
        words.insert(int(len(words) * frac), " " + f + " ")
    q = "\n\nNa podstawie powyzszego tekstu odpowiedz krotko: (1) jaki jest kod dostepu do serwerowni? (2) jaki jest ulubiony kolor Marka Kowalskiego? (3) o ktorej odjezdza pociag do Gdyni? Odpowiedz w jednej linii."
    return " ".join(words) + q, [f"{seed * 7 + 11}-ZETA", "turkusowy", f"{6 + seed % 12}:{(seed * 13) % 60:02d}"]


def long_worker(i):
    rnd = random.Random(3000 + i + (a.seed or 0))
    lens = [int(x) for x in a.long_lens.split(",")]
    n = 0
    while not STOP.is_set() and time.time() < T_END:
        L = rnd.choice(lens)
        seed = int(time.time()) % 100000 + i * 7 + n; n += 1
        try:
            p, nd = needle_prompt(L, seed)
        except Exception as e:  # noqa: BLE001
            log("error", worker="long", err=f"needle_prompt: {e!r}"); time.sleep(10); continue
        t0 = time.perf_counter()
        try:
            # chat (szablon + reasoning): surowy /v1/completions pod obciazeniem czesto "kontynuuje dokument" zamiast odpowiedziec;
            # fakty sprawdzamy w reasoning+content (jak parity.py dla testow chat)
            r = json.loads(post("/v1/chat/completions", {"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": 800, "temperature": 0.0}, timeout=3600))
            dt = time.perf_counter() - t0
            m = r["choices"][0]["message"]; u = r.get("usage", {})
            txt = (m.get("reasoning_content") or m.get("reasoning") or "") + "\n" + (m.get("content") or "")
            ok = all(x.lower() in txt.lower() for x in nd)
            with LOCK:
                STATS["long_ok" if ok else "long_needle_fail"] += 1
                LAT["long_%dk" % (L // 1000)].append(dt)
            log("long", tokens=u.get("prompt_tokens"), compl=u.get("completion_tokens"), secs=round(dt, 1), toks_per_s=round((u.get("prompt_tokens") or 0) / dt),
                needle_ok=ok, expect=nd, text=txt[:200] if ok else txt[:1500])
        except Exception as e:  # noqa: BLE001
            with LOCK:
                STATS["long_err"] += 1
            log("error", worker="long", tokens=L, secs=round(time.perf_counter() - t0, 1), err=repr(e)[:300])
        time.sleep(rnd.uniform(5, 30))


def burst_worker():
    rnd = random.Random(4000 + (a.seed or 0))
    if a.burst_every <= 0:
        return
    while not STOP.is_set() and time.time() < T_END:
        if STOP.wait(a.burst_every * 60):
            return
        if time.time() >= T_END:
            return
        t0 = time.perf_counter()
        ths = []
        res = []
        def one(k):
            res.append(chat([{"role": "user", "content": f"{rnd.choice(QA)} (wariant {k})"}], 120, 0.7, "burst", timeout=900))
        for k in range(a.burst_size):
            th = threading.Thread(target=one, args=(k,), daemon=True); th.start(); ths.append(th)
        for th in ths:
            th.join()
        dt = time.perf_counter() - t0
        oks = sum(1 for r in res if r.get("ok"))
        log("burst", size=a.burst_size, ok=oks, secs=round(dt, 1), p95=round(sorted(r["secs"] for r in res)[int(0.95 * (len(res) - 1))], 1))


def metrics():
    m = post("/metrics")
    def g(name):
        v = re.findall(rf"^vllm:{name}\{{[^}}]*\}} ([0-9.e+-]+)$", m, flags=re.M)
        return sum(float(x) for x in v)
    return dict(running=g("num_requests_running"), waiting=g("num_requests_waiting"), kv=round(g("kv_cache_usage_perc") * 100, 1),
                drafts=g("spec_decode_num_drafts_total"), accepted=g("spec_decode_num_accepted_tokens_total"), gen=g("generation_tokens_total"),
                prompt=g("prompt_tokens_total"), succ=g("request_success_total"))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}"


def monitor():
    prev = None
    dmesg0 = sh("dmesg | grep -ciE 'xid|nvrm'").strip()
    err0 = sh(f"docker logs {a.container} 2>&1 | grep -cE 'ERROR|Traceback'").strip()
    while not STOP.is_set() and time.time() < T_END:
        try:
            m = metrics()
        except Exception as e:  # noqa: BLE001
            log("monitor_err", err=repr(e)[:200]); m = None
        gpu = sh("nvidia-smi --query-gpu=index,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits").strip().splitlines()
        used = [int(x.split(",")[1]) for x in gpu if x.split(",")[0].strip() != "6"] if gpu and "ERR" not in gpu[0] else []
        pw = [float(x.split(",")[2]) for x in gpu if x.split(",")[0].strip() != "6"] if used else []
        tmp = [int(x.split(",")[3]) for x in gpu if x.split(",")[0].strip() != "6"] if used else []
        dm = sh("dmesg | grep -ciE 'xid|nvrm'").strip(); er = sh(f"docker logs {a.container} 2>&1 | grep -cE 'ERROR|Traceback'").strip()
        rss = sh(f"docker stats --no-stream --format '{{{{.MemUsage}}}}' {a.container}").strip()
        rec = dict(gpu_used_max=max(used) if used else None, gpu_used_min=min(used) if used else None, pw_avg=round(sum(pw) / len(pw), 0) if pw else None,
                   temp_max=max(tmp) if tmp else None, dmesg_new=int(dm) - int(dmesg0) if dm.isdigit() and dmesg0.isdigit() else dm, log_err_new=int(er) - int(err0) if er.isdigit() and err0.isdigit() else er, rss=rss)
        if m:
            rec.update(running=m["running"], waiting=m["waiting"], kv_pct=m["kv"])
            if prev:
                dd = m["drafts"] - prev["drafts"]; da = m["accepted"] - prev["accepted"]
                rec.update(gen_tps=round((m["gen"] - prev["gen"]) / 60, 1), prompt_tps=round((m["prompt"] - prev["prompt"]) / 60), acc=round(1 + da / dd, 2) if dd else None,
                           req_per_min=round(m["succ"] - prev["succ"]))
            prev = m
        log("monitor", **rec)
        if rec["dmesg_new"] not in (0, "0") and isinstance(rec["dmesg_new"], int) and rec["dmesg_new"] > 0:
            log("ALERT", what="dmesg", lines=sh("dmesg | grep -iE 'xid|nvrm' | tail -3"))
        STOP.wait(60)


threads = [threading.Thread(target=monitor, daemon=True)]
threads += [threading.Thread(target=chat_worker, args=(i,), daemon=True) for i in range(a.chat)]
threads += [threading.Thread(target=tool_worker, args=(i,), daemon=True) for i in range(a.tools)]
threads += [threading.Thread(target=long_worker, args=(i,), daemon=True) for i in range(a.long)]
threads += [threading.Thread(target=burst_worker, daemon=True)]
log("start", minutes=a.minutes, chat=a.chat, tools=a.tools, long=a.long, model=a.model)
for th in threads:
    th.start()
try:
    while time.time() < T_END:
        time.sleep(5)
except KeyboardInterrupt:
    pass
STOP.set()
# daj workerom dokonczyc biezace requesty (dlugie prefille moga trwac minuty)
for th in threads:
    th.join(timeout=900)
alive = sum(1 for th in threads if th.is_alive())


def pct(v, p):
    return round(sorted(v)[int(p * (len(v) - 1))], 1) if v else None


summ = dict(stats=dict(STATS), alive_threads=alive, lat={k: dict(n=len(v), p50=pct(v, .5), p95=pct(v, .95), max=round(max(v), 1)) for k, v in LAT.items()})
log("summary", **summ)
print(json.dumps(summ, ensure_ascii=False, indent=1))
bad = STATS["chat_err"] + STATS["tool_err"] + STATS["long_err"] + STATS["burst_err"] + STATS["long_needle_fail"] + STATS["chat_empty"]
raise SystemExit(1 if (bad or alive) else 0)
