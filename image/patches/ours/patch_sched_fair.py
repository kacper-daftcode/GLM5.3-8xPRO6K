"""Patch vLLM v1 scheduler (obraz nvfp4-fi618): --long-prefill-token-threshold stosowany TYLKO, gdy w systemie jest
wiecej niz jeden request (running + waiting > 1). Samotny dlugi prefill idzie pelnymi chunkami (max-num-batched-tokens),
a gdy pojawia sie inni, chunk jest przycinany do progu => nowi wchodza natychmiast, decode innych ma krotsze kroki.
Sterowanie: VLLM_LONG_PREFILL_THRESHOLD_SHARED_ONLY=1 (domyslnie) / 0 = zachowanie upstream (prog zawsze).
D3 (opcjonalnie): VLLM_LONG_PREFILL_THRESHOLD_DYNAMIC="N:T" - gdy w systemie jest wiecej niz N requestow (running + waiting),
chunk dlugiego prefillu maleje z progu bazowego do T (np. "4:256": 512 przy <=4 requestach, 256 powyzej). Domyslnie wylaczone.
Uwaga: flagi --max-num-partial-prefills / --max-long-partial-prefills nadal sa blokowane przez guard w arg_utils
(V1 i tak ich nie implementuje); sam --long-prefill-token-threshold przechodzi bez patcha.
Idempotentny, twarde anchory (dziala na pliku bazowym i na juz spatchowanym - krok 1b dokleja D3 do kroku 1).
"""
import pathlib

S = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")


def patch(path, old, new, count=1):
    s = path.read_text()
    if new in s:
        return
    assert old in s, f"ANCHOR NOT FOUND in {path}:\n{old[:200]}"
    assert s.count(old) == count, f"anchor x{s.count(old)} != {count} in {path}"
    path.write_text(s.replace(old, new))
    print(f"patched: {path.name}: {old[:60]!r}...")


# 0. flaga modulowa (po pierwszym imporcie w pliku)
patch(
    S,
    "from vllm.logger import init_logger\n",
    """from vllm.logger import init_logger

import os as _tq_os

# TQ_FAIR: prog long_prefill_token_threshold tylko przy >1 requestach w systemie (1 = wlaczone, 0 = jak upstream)
_TQ_LP_SHARED_ONLY = _tq_os.environ.get("VLLM_LONG_PREFILL_THRESHOLD_SHARED_ONLY", "1") != "0"
""",
)

# 0b. D3: parametry dynamicznego progu (N:T), domyslnie wylaczone
patch(
    S,
    """_TQ_LP_SHARED_ONLY = _tq_os.environ.get("VLLM_LONG_PREFILL_THRESHOLD_SHARED_ONLY", "1") != "0"
""",
    """_TQ_LP_SHARED_ONLY = _tq_os.environ.get("VLLM_LONG_PREFILL_THRESHOLD_SHARED_ONLY", "1") != "0"
# TQ_FAIR D3: "N:T" -> powyzej N requestow w systemie chunk dlugiego prefillu = min(prog, T); puste = wylaczone
_TQ_LP_DYN = _tq_os.environ.get("VLLM_LONG_PREFILL_THRESHOLD_DYNAMIC", "").strip()
_TQ_LP_DYN_N, _TQ_LP_DYN_T = (int(v) for v in _TQ_LP_DYN.split(":")) if _TQ_LP_DYN else (0, 0)
""",
)

# 1. efektywny prog liczony raz na krok
patch(
    S,
    """        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0
""",
    """        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # TQ_FAIR: samotny request -> bez progu (pelne chunki); >1 requestow -> prog (fairness)
        _lp_threshold = self.scheduler_config.long_prefill_token_threshold
        if (
            _lp_threshold > 0
            and _TQ_LP_SHARED_ONLY
            and (len(self.running) + len(self.waiting)) <= 1
        ):
            _lp_threshold = 0
""",
)

# 1b. D3: dynamiczny prog (doklejany za krokiem 1; anchor = kod z kroku 1, wiec dziala tez na juz spatchowanym pliku)
patch(
    S,
    """            and (len(self.running) + len(self.waiting)) <= 1
        ):
            _lp_threshold = 0
""",
    """            and (len(self.running) + len(self.waiting)) <= 1
        ):
            _lp_threshold = 0
        # TQ_FAIR D3: powyzej N requestow w systemie chunk maleje do T (VLLM_LONG_PREFILL_THRESHOLD_DYNAMIC="N:T")
        if (
            _lp_threshold > 0
            and _TQ_LP_DYN_N > 0
            and (len(self.running) + len(self.waiting)) > _TQ_LP_DYN_N
        ):
            _lp_threshold = min(_lp_threshold, _TQ_LP_DYN_T)
""",
)

# 2. petla RUNNING
patch(
    S,
    """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
""",
    """            if 0 < _lp_threshold < num_new_tokens:  # TQ_FAIR
                num_new_tokens = _lp_threshold
""",
)

# 3. petla WAITING
patch(
    S,
    """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
""",
    """                    threshold = _lp_threshold  # TQ_FAIR
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
""",
)

import ast

ast.parse(S.read_text())
print("SYNTAX-OK")
print("PATCH-SCHED-FAIR-DONE")
