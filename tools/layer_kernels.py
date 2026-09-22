"""Sekwencja kerneli GPU jednej warstwy decode ze sladu torch.profiler (miedzy dwoma kolejnymi kernelami sparse_mla_decode).
Uzycie: layer_kernels.py trace.json.gz [--layer-index 20] [--all-steps]  -> nazwy, czasy, luki; sumuje po kategoriach."""
import gzip, json, sys, re, collections

path = sys.argv[1]
li = int(sys.argv[sys.argv.index("--layer-index") + 1]) if "--layer-index" in sys.argv else 20
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"] if isinstance(data, dict) else data
kern = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset", "Kernel")], key=lambda e: e["ts"])
anchors = [i for i, e in enumerate(kern) if "sparse_mla_decode" in e["name"]]
print(f"kerneli: {len(kern)}, warstw (anchor sparse_mla_decode): {len(anchors)}")
a, b = anchors[li], anchors[li + 1]
seg = kern[a:b]
t0 = seg[0]["ts"]
tot = seg[-1]["ts"] + seg[-1]["dur"] - t0
print(f"warstwa #{li}: {len(seg)} kerneli, {tot:.0f} us od startu attention do nastepnego attention")
prev_end = t0
for e in seg:
    gap = e["ts"] - prev_end
    n = e["name"]
    n = re.sub(r"\(.*", "", n)[:110]
    print(f"  +{e['ts']-t0:7.0f} us  dur {e['dur']:6.1f}  gap {gap:5.1f}  {n}")
    prev_end = max(prev_end, e["ts"] + e["dur"])
# statystyka po wszystkich warstwach: srednia liczba kerneli i suma czasu wg nazwy
agg = collections.defaultdict(lambda: [0, 0.0])
nl = 0
for i in range(len(anchors) - 1):
    s = kern[anchors[i]:anchors[i + 1]]
    if (s[-1]["ts"] + s[-1]["dur"] - s[0]["ts"]) > 5000:  # pomijamy granice krokow
        continue
    nl += 1
    for e in s:
        k = re.sub(r"\(.*", "", e["name"])[:100]
        agg[k][0] += 1; agg[k][1] += e["dur"]
print(f"\n== srednio na warstwe (po {nl} warstwach) ==")
for k, (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f"  {d/nl:7.1f} us  x{c/nl:4.1f}  avg {d/c:6.1f}  {k}")
