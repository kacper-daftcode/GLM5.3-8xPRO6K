#!/usr/bin/env python3
"""Wymusza vLLM custom all-reduce (IPC + P2P, one-shot/two-shot) na >2 GPU PCIe bez NVLink.

vLLM ma dwie bramki:
  1) __init__: `if world_size > 2 and not fully_connected: return`  (custom AR w ogole nie startuje)
  2) should_custom_ar(): dla !fully_connected zwraca False           (custom AR nigdy nie jest wybierany)
Stary obraz `custom-ar` patchowal tylko (1), wiec efektu nie bylo. Tu patchujemy obie i dodajemy
prog rozmiaru dla PCIe: VLLM_CUSTOM_AR_PCIE_MAX_SIZE (bajty, domyslnie 2 MiB; 0 = zachowanie oryginalne).
"""
import re, sys

F = "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/custom_all_reduce.py"
src = open(F).read()
orig = src

# (1) bramka inicjalizacji -> zamiast return: ogranicz max_size progiem PCIe i jedz dalej
old1 = '''        if world_size > 2 and not fully_connected:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return
'''
new1 = '''        if world_size > 2 and not fully_connected:
            import os as _os
            _pcie_max = int(_os.environ.get("VLLM_CUSTOM_AR_PCIE_MAX_SIZE", str(2 * 1024 * 1024)))
            if _pcie_max <= 0:
                logger.warning(
                    "Custom allreduce is disabled because it's not supported on"
                    " more than two PCIe-only GPUs. To silence this warning, "
                    "specify disable_custom_all_reduce=True explicitly."
                )
                return
            max_size = min(max_size, _pcie_max)
            logger.warning(
                "[PCIE-CAR] Custom allreduce FORCED on %d PCIe-only GPUs "
                "(max_size=%d bytes, VLLM_CUSTOM_AR_PCIE_MAX_SIZE)",
                world_size, max_size,
            )
'''
assert old1 in src, "bramka (1) nie znaleziona - inna wersja vLLM?"
src = src.replace(old1, new1)

# (2) should_custom_ar -> uzywaj progu max_size takze dla PCIe
old2 = '''        if self.world_size == 2 or self.fully_connected:
            return inp_size < self.max_size
        return False
'''
new2 = '''        return inp_size < self.max_size
'''
assert old2 in src, "bramka (2) nie znaleziona - inna wersja vLLM?"
src = src.replace(old2, new2)

assert src != orig
open(F, "w").write(src)
compile(src, F, "exec")
print("PCIE-CAR patch OK")
