"""Sprawdza (bez modyfikacji) anchory patch_flashinfer.py wobec zainstalowanego FlashInfer.

Uzycie w kontenerze: python3 /a2/check_anchors.py /patch/patch_flashinfer.py
"""
import pathlib
import sys
import types

src_path = sys.argv[1]
src = pathlib.Path(src_path).read_text()

results = []


def patch(path, old, new, count=1):
    try:
        s = path.read_text()
    except FileNotFoundError:
        results.append(("MISSING-FILE", str(path), old[:70]))
        return
    if new in s:
        results.append(("ALREADY", path.name, old[:70]))
        return
    n = s.count(old)
    if n == count:
        results.append(("OK", path.name, old[:70]))
    elif n == 0:
        results.append(("NO-ANCHOR", path.name, old[:70]))
    else:
        results.append((f"COUNT={n}", path.name, old[:70]))


def append_once(path, text):
    results.append(("APPEND", path.name, text[:70]))


# wykonaj skrypt z podmienionymi funkcjami; zatrzymaj na sekcji "syntax check"
body = src.split("# syntax check")[0]
body = body.replace("def patch(", "def _orig_patch(").replace("def append_once(", "def _orig_append_once(")
ns = {"__name__": "__anchor_check__"}
# nadpisujemy po definicji: wstrzykujemy nasze funkcje przez exec w dwu krokach
import re

# usun linie zapisujace plik naglowka (krok 0), zeby nic nie modyfikowac
body = body.replace('(INC / "common/nvfp4_expand.cuh").write_text(expand)', "pass")
exec(body.split("# ---------------------------------------------------------------- 0. naglowek")[0], ns)
ns["patch"] = patch
ns["append_once"] = append_once
exec("# ---------------------------------------------------------------- 0. naglowek" + body.split("# ---------------------------------------------------------------- 0. naglowek")[1], ns)

bad = 0
for st, f, a in results:
    flag = "" if st in ("OK", "ALREADY", "APPEND") else "  <<<<"
    if flag:
        bad += 1
    print(f"{st:12s} {f:40s} {a!r}{flag}")
print(f"\nTOTAL={len(results)} BAD={bad}")
