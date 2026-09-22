"""Patch vLLM (obraz nvfp4-fi618): B1 etap 2 - draft MTP i dokladnosc FP8.

1. Online NVFP4 dla WYKLUCZONYCH (BF16) warstw MoE - w praktyce eksperci drafta MTP (model.layers.78.mlp.experts):
   env VLLM_TQ_NVFP4_MOE="layers.78.mlp.experts" (podciagi prefixu). Wagi BF16 ladowane jak w UnquantizedFusedMoEMethod,
   w process_weights_after_loading kwantyzacja do layoutu checkpointu ModelOpt (FP4 e2m1 packed + skale blokowe E4M3
   per 16 + globalna per ekspert) i dalej NIEZMIENIONA sciezka ModelOptNvFp4FusedMoE (FLASHINFER_CUTLASS jak target).
   Skale aktywacji (statyczne, per warstwa) = proxy z ostatniej warstwy targetu w checkpoincie x margines
   (VLLM_TQ_NVFP4_MOE_AMAX_MARGIN, domyslnie 2.0) albo jawnie VLLM_TQ_NVFP4_MOE_ASCALE="a13_input_scale,a2_input_scale".
   Kwantyzacja drafta nie zmienia rozkladu wyjscia (rejection sampling) - wplywa tylko na akceptacje/predkosc.
2. Per-channel FP8 (skala per wiersz wagi) zamiast per-tensor dla warstw z VLLM_TQ_FP8_LINEARS: VLLM_TQ_FP8_CHANNEL=1.
   Kernel CUTLASS W8A8 (per-token aktywacje) wspiera to bez zmian; koszt identyczny, mniejszy blad kwantyzacji.
3. eh_proj drafta (nn.Linear 6144x12288 BF16, ~100 us/pass x5) -> ReplicatedLinear, wiec pattern "eh_proj"
   w VLLM_TQ_FP8_LINEARS daje mu online FP8. Bez patternu zachowanie jak dotad (nn.Linear).
Idempotentny, twarde anchory. Wymaga wczesniejszego patch_fp8_excluded.py.
"""
import ast
import os
import pathlib

SP = pathlib.Path(os.environ.get("TQ_VLLM_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"))
M = SP / "model_executor/layers/quantization/modelopt.py"
MTP = SP / "model_executor/models/deepseek_mtp.py"


def patch(path, old, new, count=1):
    s = path.read_text()
    if new in s:
        return
    assert old in s, f"ANCHOR NOT FOUND in {path}:\n{old[:200]}"
    assert s.count(old) == count, f"anchor x{s.count(old)} != {count} in {path}"
    path.write_text(s.replace(old, new))
    print(f"patched: {path.name}: {old[:60]!r}...")


# ---------------------------------------------------------------------------------------------------------------------
# 1) online NVFP4 dla wykluczonych MoE (draft MTP) - helper przed _tq_fp8_method_for_excluded
patch(
    M,
    "def _tq_fp8_method_for_excluded(prefix: str, layer):\n",
    '''_TQ_NVFP4_MOE_PATTERNS = [p.strip() for p in _tq_os.environ.get("VLLM_TQ_NVFP4_MOE", "").split(",") if p.strip()]
_TqOnlineNvFp4MoEMethod = None
_TQ_FP8_CHANNEL = _tq_os.environ.get("VLLM_TQ_FP8_CHANNEL", "0") == "1"


def _tq_nvfp4_quant_ckpt_layout(w, gs):
    """TQ_B1: kwantyzacja [N, K] (bf16/fp32, GPU) do layoutu checkpointu ModelOpt NVFP4.
    gs = globalna skala wagi (weight_scale_2 = amax / (6*448), fp32 scalar tensor).
    Zwraca (packed uint8 [N, K//2]: parzysty element w dolnym nibble'u, nieparzysty w gornym;
            skale blokowe float8_e4m3fn [N, K//16] w layoutcie liniowym - swizzle robi sciezka bazowa)."""
    import torch as _t

    N, K = w.shape
    assert K % 16 == 0, (N, K)
    x = w.to(_t.float32).reshape(N, K // 16, 16)
    bmax = x.abs().amax(dim=-1, keepdim=True)
    sf = (bmax / 6.0 / gs).clamp_(max=448.0).to(_t.float8_e4m3fn)
    denom = sf.to(_t.float32) * gs
    q = _t.where(denom > 0, x / denom, _t.zeros_like(x)).clamp_(-6.0, 6.0)
    mag = q.abs()
    # najblizszy punkt siatki e2m1 {0,.5,1,1.5,2,3,4,6}; remisy jak RNE (=cast_to_fp4 z vllm nvfp4_emulation_utils)
    idx = (
        (mag > 0.25).to(_t.uint8) + (mag >= 0.75) + (mag > 1.25) + (mag >= 1.75)
        + (mag > 2.5) + (mag >= 3.5) + (mag > 5.0)
    )
    nib = (idx | ((q < 0).to(_t.uint8) << 3)).reshape(N, K)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return packed.contiguous(), sf.reshape(N, K // 16).contiguous()


def _tq_nvfp4_moe_input_scales(model_path, last_layer):
    """TQ_B1: statyczne skale aktywacji (input_scale = amax/(6*448)) dla drafta: jawne z env albo proxy z checkpointu
    (ostatnia warstwa targetu; w checkpoincie incoai sa identyczne dla wszystkich ekspertow warstwy) x margines."""
    import glob as _glob
    import json as _json

    exp = _tq_os.environ.get("VLLM_TQ_NVFP4_MOE_ASCALE", "").strip()
    if exp:
        a13, a2 = (float(v) for v in exp.split(","))
        return a13, a2, "env"
    margin = float(_tq_os.environ.get("VLLM_TQ_NVFP4_MOE_AMAX_MARGIN", "2.0"))
    from safetensors import safe_open as _so

    keys = {
        "a13": f"model.layers.{last_layer}.mlp.experts.0.gate_proj.input_scale",
        "a13b": f"model.layers.{last_layer}.mlp.experts.0.up_proj.input_scale",
        "a2": f"model.layers.{last_layer}.mlp.experts.0.down_proj.input_scale",
    }
    idx_path = _tq_os.path.join(model_path, "model.safetensors.index.json")
    files = {}
    if _tq_os.path.isfile(idx_path):
        wm = _json.load(open(idx_path))["weight_map"]
        for k, name in keys.items():
            if name in wm:
                files[k] = _tq_os.path.join(model_path, wm[name])
    else:
        for f in _glob.glob(_tq_os.path.join(model_path, "*.safetensors")):
            with _so(f, "pt") as sf:
                ks = set(sf.keys())
            for k, name in keys.items():
                if name in ks:
                    files[k] = f
    vals = {}
    for k, f in files.items():
        with _so(f, "pt") as sf:
            vals[k] = float(sf.get_tensor(keys[k]).float().max())
    if "a13" not in vals or "a2" not in vals:
        raise RuntimeError(f"TQ_B1: no proxy input_scale for layer {last_layer} in {model_path}; set VLLM_TQ_NVFP4_MOE_ASCALE")
    a13 = max(vals["a13"], vals.get("a13b", 0.0)) * margin
    a2 = vals["a2"] * margin
    return a13, a2, f"ckpt layer {last_layer} x{margin}"


def _tq_make_online_nvfp4_moe_cls():
    import torch as _torch

    from vllm.config import get_current_vllm_config as _gcfg
    from vllm.model_executor.utils import replace_parameter as _replace
    from vllm.model_executor.utils import set_weight_attrs as _swa

    class TqOnlineNvFp4MoEMethod(ModelOptNvFp4FusedMoE):
        """TQ_B1: ModelOptNvFp4FusedMoE dla warstwy MoE z wagami BF16 w checkpoincie - kwantyzacja online przy ladowaniu."""

        def __init__(self, quant_config, moe_config):
            super().__init__(quant_config, moe_config)
            try:
                cfg = _gcfg()
                self._tq_model_path = cfg.model_config.model
                self._tq_last_layer = int(cfg.model_config.hf_config.num_hidden_layers) - 1
            except Exception as e:  # noqa: BLE001
                self._tq_model_path = _tq_os.environ.get("VLLM_TQ_MODEL_PATH", "/model")
                self._tq_last_layer = int(_tq_os.environ.get("VLLM_TQ_LAST_LAYER", "77"))
                logger.warning("TQ_B1: get_current_vllm_config failed (%s), using %s / layer %d", e,
                               self._tq_model_path, self._tq_last_layer)

        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                           **extra_weight_attrs):
            assert self.moe.is_act_and_mul, "TQ_B1: only gated (w1/w3) MoE supported"
            assert not self.moe.has_bias, "TQ_B1: MoE bias not supported"
            layer.num_experts = num_experts
            layer.params_dtype = params_dtype
            layer.quant_config = self.quant_config
            w13 = _torch.nn.Parameter(
                _torch.empty(num_experts, 2 * intermediate_size_per_partition, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight", w13)
            _swa(w13, extra_weight_attrs)
            w2 = _torch.nn.Parameter(
                _torch.empty(num_experts, hidden_size, intermediate_size_per_partition, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight", w2)
            _swa(w2, extra_weight_attrs)

        @_torch.no_grad()
        def process_weights_after_loading(self, layer):
            w13 = layer.w13_weight.data
            w2 = layer.w2_weight.data
            assert w13.dtype != _torch.uint8, "TQ_B1: weights already quantized (double processing?)"
            E, N13, H = w13.shape
            I = w2.shape[2]
            dev = w13.device
            gs = 6.0 * 448.0
            a13, a2, src = _tq_nvfp4_moe_input_scales(self._tq_model_path, self._tq_last_layer)
            w13_q = _torch.empty(E, N13, H // 2, dtype=_torch.uint8, device=dev)
            w13_sf = _torch.empty(E, N13, H // 16, dtype=_torch.float8_e4m3fn, device=dev)
            w13_gs = _torch.empty(E, 2, dtype=_torch.float32, device=dev)
            w2_q = _torch.empty(E, H, I // 2, dtype=_torch.uint8, device=dev)
            w2_sf = _torch.empty(E, H, I // 16, dtype=_torch.float8_e4m3fn, device=dev)
            w2_gs = _torch.empty(E, dtype=_torch.float32, device=dev)
            err_num = _torch.zeros((), dtype=_torch.float64, device=dev)
            err_den = _torch.zeros((), dtype=_torch.float64, device=dev)
            for e in range(E):
                for src_w, dst_q, dst_sf, dst_gs in ((w13[e], w13_q[e], w13_sf[e], w13_gs[e]), (w2[e], w2_q[e], w2_sf[e], w2_gs[e])):
                    g = (src_w.to(_torch.float32).abs().amax() / gs).clamp_(min=1e-12)
                    q, sf = _tq_nvfp4_quant_ckpt_layout(src_w, g)
                    dst_q.copy_(q)
                    dst_sf.copy_(sf)
                    dst_gs.fill_(g.item())
                    if e % 64 == 0:  # blad rekonstrukcji na probce ekspertow (log)
                        lo = (q & 0x0F).to(_torch.int64)
                        hi = (q >> 4).to(_torch.int64)
                        tab = _torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
                        vals = _torch.stack((tab[lo & 7] * _torch.where(lo & 8 > 0, -1.0, 1.0),
                                             tab[hi & 7] * _torch.where(hi & 8 > 0, -1.0, 1.0)), dim=-1).reshape(q.shape[0], -1)
                        dq = (vals.reshape(q.shape[0], -1, 16) * (sf.to(_torch.float32) * g).unsqueeze(-1)).reshape(q.shape[0], -1)
                        d = dq - src_w.to(_torch.float32)
                        err_num += (d * d).sum().double()
                        err_den += (src_w.to(_torch.float32) ** 2).sum().double()
            rel = (err_num / err_den.clamp(min=1e-30)).sqrt().item()
            logger.info("TQ_B1: online NVFP4 for %s: E=%d N13=%d H=%d I=%d; a13_input_scale=%.4g a2_input_scale=%.4g (%s); "
                        "rel. weight error (sample) %.4f", getattr(layer, "layer_name", "?"), E, N13, H, I, a13, a2, src, rel)
            _replace(layer, "w13_weight", w13_q)
            _replace(layer, "w13_weight_scale", w13_sf)
            _replace(layer, "w13_weight_scale_2", w13_gs)
            _replace(layer, "w13_input_scale", _torch.full((E, 2), a13, dtype=_torch.float32, device=dev))
            _replace(layer, "w2_weight", w2_q)
            _replace(layer, "w2_weight_scale", w2_sf)
            _replace(layer, "w2_weight_scale_2", w2_gs)
            _replace(layer, "w2_input_scale", _torch.full((E,), a2, dtype=_torch.float32, device=dev))
            del w13, w2
            _torch.cuda.empty_cache()
            super().process_weights_after_loading(layer)

    return TqOnlineNvFp4MoEMethod


def _tq_nvfp4_moe_method_for_excluded(quant_config, prefix: str, layer):
    """TQ_B1: dla wykluczonej (BF16) warstwy RoutedExperts pasujacej do VLLM_TQ_NVFP4_MOE zwroc metode NVFP4 online."""
    global _TqOnlineNvFp4MoEMethod
    if not _TQ_NVFP4_MOE_PATTERNS or not any(p in prefix for p in _TQ_NVFP4_MOE_PATTERNS):
        return None
    if _TqOnlineNvFp4MoEMethod is None:
        _TqOnlineNvFp4MoEMethod = _tq_make_online_nvfp4_moe_cls()
    logger.info("TQ_B1: %s -> online NVFP4 MoE (%s)", prefix, type(layer).__name__)
    return _TqOnlineNvFp4MoEMethod(quant_config=quant_config, moe_config=layer.moe_config)


def _tq_fp8_method_for_excluded(prefix: str, layer):
''',
)

# hook w ModelOptQuantConfigBase.get_quant_method (galaz wykluczen)
patch(
    M,
    """        # handle exclusion
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                _m = _tq_fp8_method_for_excluded(prefix, layer)  # TQ_FP8
                if _m is not None:
                    return _m
                return UnquantizedLinearMethod()
            return None
""",
    """        # handle exclusion
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                _m = _tq_fp8_method_for_excluded(prefix, layer)  # TQ_FP8
                if _m is not None:
                    return _m
                return UnquantizedLinearMethod()
            if isinstance(layer, RoutedExperts):
                _m = _tq_nvfp4_moe_method_for_excluded(self, prefix, layer)  # TQ_B1
                if _m is not None:
                    return _m
            return None
""",
)

# ---------------------------------------------------------------------------------------------------------------------
# 2) per-channel FP8 w TqOnlineFp8LinearMethod (VLLM_TQ_FP8_CHANNEL=1)
patch(
    M,
    """            self.fp8_linear = _fp8.init_fp8_linear_kernel(
                activation_quant_key=self.activation_quant_key,
                weight_quant_key=self.weight_quant_key,
                weight_shape=weight.shape,
""",
    """            if _TQ_FP8_CHANNEL:  # TQ_B1: skala per wiersz wagi (kernel CUTLASS: per-token akt. x per-channel wagi)
                from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticChannelSym as _kCh
                self.weight_quant_key = _kCh
            self.fp8_linear = _fp8.init_fp8_linear_kernel(
                activation_quant_key=self.activation_quant_key,
                weight_quant_key=self.weight_quant_key,
                weight_shape=weight.shape,
""",
)
patch(
    M,
    """            qweight, scale = _ops.scaled_fp8_quant(w.contiguous())  # per-tensor, scale [1]
""",
    """            if _TQ_FP8_CHANNEL and not self.use_marlin:  # TQ_B1: per-channel [N,1]; kernel chce (K, N)
                qweight, scale = _ops.scaled_fp8_quant(w.contiguous(), use_per_token_if_dynamic=True)
                _replace(layer, "weight", qweight.t())
                _replace(layer, "weight_scale", scale.reshape(-1, 1).contiguous())
                layer.input_scale = None
                self.fp8_linear.process_weights_after_loading(layer)
                return
            qweight, scale = _ops.scaled_fp8_quant(w.contiguous())  # per-tensor, scale [1]
""",
)

# ---------------------------------------------------------------------------------------------------------------------
# 3) eh_proj drafta jako ReplicatedLinear (tylko gdy pattern "eh_proj" w VLLM_TQ_FP8_LINEARS)
patch(
    MTP,
    "from .utils import get_pp_missing_layer_names, maybe_prefix\n",
    """from .utils import get_pp_missing_layer_names, maybe_prefix

import os as _tq_os

from vllm.model_executor.layers.linear import ReplicatedLinear as _TqReplicatedLinear

# TQ_B1: eh_proj (BF16 nn.Linear, ~100 us/pass przy decode) -> ReplicatedLinear, zeby online FP8 (patch_fp8_excluded)
# mogl go objac; aktywne tylko gdy prefix eh_proj pasuje do VLLM_TQ_FP8_LINEARS (ten sam matcher co w modelopt.py).
_TQ_FP8_PATTERNS = [p.strip() for p in _tq_os.environ.get("VLLM_TQ_FP8_LINEARS", "").split(",") if p.strip()]
""",
)
patch(
    MTP,
    """        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
""",
    """        if any(p in maybe_prefix(prefix, "eh_proj") for p in _TQ_FP8_PATTERNS):  # TQ_B1
            self.eh_proj = _TqReplicatedLinear(
                config.hidden_size * 2, config.hidden_size, bias=False,
                quant_config=quant_config, prefix=maybe_prefix(prefix, "eh_proj"),
            )
        else:
            self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
""",
)
patch(
    MTP,
    """        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
""",
    """        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        if isinstance(hidden_states, tuple):  # TQ_B1: ReplicatedLinear -> (out, bias)
            hidden_states = hidden_states[0]
""",
)

for p in (M, MTP):
    ast.parse(p.read_text())
src = M.read_text()
assert src.count("def _tq_nvfp4_moe_method_for_excluded") == 1
assert "quant_config = vllm_config.quant_config" in MTP.read_text()
print("SYNTAX-OK")
print("PATCH-B1S2-DONE")
