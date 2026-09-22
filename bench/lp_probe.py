#!/usr/bin/env python3
"""Teacher-forced logprob probe (deterministyczny, niezalezny od rozjazdu greedy):
dla stalych tekstow pobiera prompt_logprobs (logprob prawdziwego tokenu na kazdej pozycji + top-1)
i zapisuje JSON. Z --compare REF.json liczy: sredni |dlogprob|, p95, max, zgodnosc top-1, roznice CE (nats/tok).
Uzycie: lp_probe.py --out fi618_lp.json [--compare base_lp.json] [--model glm-5.3]
"""
import argparse, json, math, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="glm-5.3")
ap.add_argument("--out", required=True)
ap.add_argument("--compare", default=None)
a = ap.parse_args()


def post(path, body, timeout=600):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


PL = ("Toruń to jedno z najstarszych miast Polski, położone nad Wisłą. Założony przez Krzyżaków w 1233 roku, szybko stał się "
      "ważnym ośrodkiem handlowym należącym do Hanzy. W 1473 roku urodził się tu Mikołaj Kopernik, twórca teorii heliocentrycznej. "
      "Gotycka starówka Torunia została wpisana na listę światowego dziedzictwa UNESCO w 1997 roku. Miasto słynie z pierników, "
      "których receptura sięga średniowiecza. Ratusz Staromiejski, Krzywa Wieża i ruiny zamku krzyżackiego przyciągają turystów z całego świata. "
      "Uniwersytet Mikołaja Kopernika, założony w 1945 roku, jest największą uczelnią regionu. ") * 3
EN = ("Tensor parallelism splits the weight matrices of a neural network across multiple devices so that each device computes a slice "
      "of every layer. After each sliced matrix multiplication the partial results are combined with an all-reduce, which makes the "
      "interconnect bandwidth and latency critical. On PCIe-only systems without NVLink, the all-reduce of a few tens of kilobytes can "
      "take tens of microseconds, and a decoder step with 78 layers performs roughly two of them per layer. Pipeline parallelism instead "
      "assigns whole layers to devices and passes activations between stages, trading all-reduce traffic for pipeline bubbles. ") * 3
CODE = ('import re\nfrom collections import Counter, defaultdict\n\n\ndef parse_nginx_log(path: str, top: int = 10):\n'
        '    """Parse an nginx combined-format access log and return the top IPs by request count with transfer in MB."""\n'
        '    pattern = re.compile(r"^(\\S+) \\S+ \\S+ \\[[^\\]]+\\] \\"[^\\"]*\\" (\\d{3}) (\\d+|-)")\n'
        '    counts = Counter()\n    transfer = defaultdict(int)\n    with open(path, encoding="utf-8", errors="replace") as fh:\n'
        '        for line in fh:\n            m = pattern.match(line)\n            if not m:\n                continue\n'
        '            ip, _status, size = m.groups()\n            counts[ip] += 1\n            if size != "-":\n'
        '                transfer[ip] += int(size)\n    result = []\n    for ip, n in counts.most_common(top):\n'
        '        result.append({"ip": ip, "requests": n, "mb": round(transfer[ip] / 1_048_576, 2)})\n    return result\n\n\n'
        'if __name__ == "__main__":\n    import json, sys\n    print(json.dumps(parse_nginx_log(sys.argv[1]), indent=2))\n') * 2
MIX = ('{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Kraków", "unit": "C"}}], "reasoning": "Użytkownik pyta o pogodę '
       'w Krakowie w stopniach Celsjusza, więc wywołuję narzędzie get_weather z parametrem unit=C."}\n'
       'Suma liczb pierwszych mniejszych od 100: 2+3+5+7+11+13+17+19+23+29+31+37+41+43+47+53+59+61+67+71+73+79+83+89+97 = 1060.\n'
       'Die Hauptstadt von Polen ist Warschau. La capitale de la Pologne est Varsovie. 波兰的首都是华沙。\n') * 4
TEXTS = {"pl": PL, "en": EN, "code": CODE, "mix": MIX}

out = {}
for name, text in TEXTS.items():
    ids = post("/tokenize", {"model": a.model, "prompt": text})["tokens"]
    r = post("/v1/completions", {"model": a.model, "prompt": text, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1, "logprobs": 1})
    plp = r["choices"][0].get("prompt_logprobs")
    if plp is None:
        raise SystemExit("serwer nie zwrocil prompt_logprobs")
    rows = []
    for i, entry in enumerate(plp):
        if entry is None:  # pierwsza pozycja
            continue
        tid = str(ids[i])
        actual = entry.get(tid)
        top = max(entry.items(), key=lambda kv: kv[1]["logprob"])
        rows.append({"pos": i, "tok": ids[i], "lp": actual["logprob"] if actual else None, "top1": int(top[0])})
    out[name] = {"n": len(rows), "rows": rows}
    lps = [x["lp"] for x in rows if x["lp"] is not None]
    print(f"{name:>5}: {len(rows)} tok, CE={-(sum(lps) / len(lps)):.4f} nat/tok, top1-acc={sum(1 for x in rows if x['top1'] == x['tok']) / len(rows):.3f}")
json.dump(out, open(a.out, "w"))
print("zapisano", a.out)

if a.compare:
    ref = json.load(open(a.compare))
    print(f"\n=== porownanie z {a.compare} ===")
    for name in out:
        A = {x["pos"]: x for x in out[name]["rows"]}
        B = {x["pos"]: x for x in ref[name]["rows"]}
        d = [abs(A[p]["lp"] - B[p]["lp"]) for p in A if p in B and A[p]["lp"] is not None and B[p]["lp"] is not None]
        d.sort()
        agree = sum(1 for p in A if p in B and A[p]["top1"] == B[p]["top1"]) / max(1, len(A))
        ce_a = -sum(A[p]["lp"] for p in A) / len(A); ce_b = -sum(B[p]["lp"] for p in B) / len(B)
        print(f"{name:>5}: mean|dlp|={sum(d) / len(d):.4f} p95={d[int(0.95 * len(d)) - 1]:.4f} max={d[-1]:.4f} | top1 agree={agree:.3f} | CE {ce_b:.4f} -> {ce_a:.4f} (d={ce_a - ce_b:+.4f})")
