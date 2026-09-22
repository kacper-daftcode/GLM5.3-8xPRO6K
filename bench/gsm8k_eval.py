#!/usr/bin/env python3
"""GSM8K accuracy through the OpenAI-compatible API (greedy, chat with reasoning). Quality gate that is comparable across stacks.
Data: openai/grade-school-math test.jsonl (downloaded once to --cache). Answer = last number in `content` (falls back to reasoning).
Usage: gsm8k_eval.py --n 500 --conc 8 --out results/gsm8k_<tag>.json [--url http://127.0.0.1:8000 --model glm-5.3]
"""
import argparse, concurrent.futures as cf, json, os, re, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="glm-5.3")
ap.add_argument("--n", type=int, default=500); ap.add_argument("--conc", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--out", required=True); ap.add_argument("--cache", default=os.path.expanduser("~/.cache/gsm8k_test.jsonl"))
ap.add_argument("--seed", type=int, default=0, help="offset into the test set")
a = ap.parse_args()

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
if not os.path.exists(a.cache):
    os.makedirs(os.path.dirname(a.cache), exist_ok=True)
    urllib.request.urlretrieve(URL, a.cache)
rows = [json.loads(l) for l in open(a.cache) if l.strip()][a.seed:a.seed + a.n]

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def norm(x):
    x = x.replace(",", "").strip().rstrip(".")
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return x


def ask(i, row):
    q = row["question"]
    gold = norm(row["answer"].split("####")[-1])
    body = {"model": a.model, "messages": [{"role": "user", "content": q + "\n\nSolve step by step, then give the final numeric answer on the last line as: Answer: <number>"}],
            "max_tokens": a.max_tokens, "temperature": 0.0, "seed": 1}
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        m = r["choices"][0]["message"]; content = m.get("content") or ""; reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
        src = content if NUM.search(content) else reasoning
        mm = re.search(r"Answer:\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", src)
        pred = norm(mm.group(1)) if mm else (norm(NUM.findall(src)[-1]) if NUM.findall(src) else None)
        return dict(i=i, gold=gold, pred=pred, ok=pred == gold, finish=r["choices"][0].get("finish_reason"), tokens=r["usage"]["completion_tokens"], secs=round(time.perf_counter() - t0, 1))
    except Exception as e:  # noqa: BLE001
        return dict(i=i, gold=gold, pred=None, ok=False, err=repr(e)[:200], secs=round(time.perf_counter() - t0, 1))


t0 = time.perf_counter()
with cf.ThreadPoolExecutor(a.conc) as ex:
    res = list(ex.map(lambda p: ask(*p), enumerate(rows)))
wall = time.perf_counter() - t0
acc = sum(r["ok"] for r in res) / len(res)
errs = sum(1 for r in res if r.get("err"))
trunc = sum(1 for r in res if r.get("finish") == "length")
toks = sum(r.get("tokens", 0) for r in res)
summ = dict(model=a.model, n=len(res), accuracy=round(acc, 4), errors=errs, truncated=trunc, completion_tokens=toks, wall_s=round(wall, 1), tok_s=round(toks / wall, 1), conc=a.conc, max_tokens=a.max_tokens, seed=a.seed)
json.dump(dict(summary=summ, results=res), open(a.out, "w"), indent=1)
print(json.dumps(summ))
