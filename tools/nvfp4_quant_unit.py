"""Unit-test kwantyzatora online NVFP4 (patch_b1s2.py) vs checkpoint ModelOpt i kernel vLLM scaled_fp4_quant.
Uruchamiac w obrazie fi618 z nalozonym patchem (modelopt.py) i checkpointem pod /model, na dowolnym GPU (np. 5090):
  docker run --rm --gpus '"device=6"' --entrypoint python3 -v <patched modelopt.py>:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py \
     -v /path/to/GLM-5.3-NVFP4:/model -v $PWD/tools:/tools vllm-nightly-fi614:nvfp4-fi618 /tools/nvfp4_quant_unit.py
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open

MODEL = os.environ.get("MODEL", "/model")
dev = torch.device("cuda:0")

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.modelopt import _tq_nvfp4_quant_ckpt_layout  # noqa: E402
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E402
    break_fp4_bytes,
    dequantize_to_dtype,
    kE2M1ToFloat_handle,
)

kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(dev)

wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]


def get(k):
    with safe_open(f"{MODEL}/{wm[k]}", "pt") as f:
        return f.get_tensor(k)


def dequant_linear(packed, sf, gs):
    """[N,K/2] uint8, [N,K/16] fp8, scalar fp32 -> [N,K] fp32 (layout liniowy skal)."""
    N = packed.shape[0]
    vals = break_fp4_bytes(packed, torch.float32)  # [N, K]
    K = vals.shape[1]
    return (vals.reshape(N, K // 16, 16) * (sf.float() * gs).unsqueeze(-1)).reshape(N, K)


ok = True

# 1) round-trip na skwantyzowanym ekspercie targetu (warstwa 77): dequant -> quant musi odtworzyc bajty i skale
k = "model.layers.77.mlp.experts.0.gate_proj"
packed = get(k + ".weight").to(dev)
sf = get(k + ".weight_scale").to(dev)
gs = get(k + ".weight_scale_2").float().to(dev).reshape(())
print("ckpt expert:", tuple(packed.shape), packed.dtype, tuple(sf.shape), sf.dtype, "gs", gs.item())
dq = dequant_linear(packed, sf, gs)
dq_ref = dequantize_to_dtype(packed, sf, gs, dtype=torch.float32, swizzle=False)
print("  my dequant vs vllm dequant max|diff|:", (dq - dq_ref).abs().max().item())
q2, sf2 = _tq_nvfp4_quant_ckpt_layout(dq, gs)
sf_eq = torch.equal(sf2.view(torch.uint8), sf.view(torch.uint8))
mag_eq = torch.equal(q2 & 0x77, packed & 0x77)
# znak porownujemy tylko tam, gdzie magnitude != 0 (-0 vs +0 jest nieistotne)
lo_nz = (packed & 0x07) != 0
hi_nz = (packed & 0x70) != 0
sign_lo = ((q2 ^ packed) & 0x08)[lo_nz].any().item()
sign_hi = ((q2 ^ packed) & 0x80)[hi_nz].any().item()
print(f"  round-trip: scales equal={sf_eq} magnitudes equal={mag_eq} sign mismatch lo={sign_lo} hi={sign_hi}")
if not (sf_eq and mag_eq and not sign_lo and not sign_hi):
    ok = False
    nmis = ((q2 & 0x77) != (packed & 0x77)).sum().item()
    print("  MISMATCH bytes:", nmis, "of", packed.numel())

# 2) ekspert drafta (warstwa 78, BF16): blad rekonstrukcji + zgodnosc z kernelem vLLM scaled_fp4_quant
for name in ("gate_proj", "up_proj", "down_proj"):
    k = f"model.layers.78.mlp.experts.0.{name}.weight"
    w = get(k).to(dev)
    g = (w.float().abs().amax() / (6.0 * 448.0)).reshape(())
    t0 = time.time()
    q, s = _tq_nvfp4_quant_ckpt_layout(w, g)
    torch.cuda.synchronize()
    dt = time.time() - t0
    dq = dequant_linear(q, s, g)
    rel = ((dq - w.float()).norm() / w.float().norm()).item()
    # kernel vLLM: input_global_scale = 1/gs (konwencja odwrotna), skale wyjsciowe swizzlowane -> porownujemy dane
    qk, sk = ops.scaled_fp4_quant(w, (1.0 / g).reshape(1))
    same = (qk == q).float().mean().item()
    same_mag = ((qk & 0x77) == (q & 0x77)).float().mean().item()
    dq_k = dequantize_to_dtype(qk, sk, g, dtype=torch.float32, swizzle=True)
    rel_k = ((dq_k - w.float()).norm() / w.float().norm()).item()
    print(f"draft {name}: {tuple(w.shape)} {w.dtype} quant {dt*1000:.0f} ms; rel err mine={rel:.4f} kernel={rel_k:.4f}; "
          f"packed bytes identical to kernel: {same*100:.3f}% (magnitudes {same_mag*100:.3f}%)")
    if rel > 0.12 or same_mag < 0.995:
        ok = False

# 3) proxy skal aktywacji z checkpointu
from vllm.model_executor.layers.quantization.modelopt import _tq_nvfp4_moe_input_scales  # noqa: E402

a13, a2, src = _tq_nvfp4_moe_input_scales(MODEL, 77)
print(f"input_scale proxy: a13={a13:.5g} (amax {a13*2688:.1f}) a2={a2:.5g} (amax {a2*2688:.1f}) [{src}]")

print("UNIT-OK" if ok else "UNIT-FAIL")
sys.exit(0 if ok else 1)
