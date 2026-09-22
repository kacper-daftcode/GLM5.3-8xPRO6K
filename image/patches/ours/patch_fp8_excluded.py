"""Patch vLLM (obraz nvfp4-fi618): online FP8 dla warstw liniowych WYKLUCZONYCH z kwantyzacji w checkpoincie modelopt.

Checkpoint incoai/GLM-5.3-NVFP4 kwantyzuje tylko routowane eksperty; q_a/q_b/kv_a/kv_b/o_proj, indexer, shared experts,
lm_head sa w BF16 (exclude_modules). Przy decode (M=6) te GEMM-y sa ograniczone pasmem pamieci (~221 MB wag/warstwe/GPU
@TP4 => ~118 us/warstwe, ~10 ms z 30 ms kroku). FP8 (2x mniej bajtow) skraca je; przy prefillu daje ~1.4-1.6x FLOPS.
Sterowanie: VLLM_TQ_FP8_LINEARS="q_a_proj,q_b_proj,o_proj,shared_experts,indexer.wq_b[,lm_head]" (podciagi prefixu warstwy);
pusty/brak = zachowanie oryginalne. Kernel: domyslnie W8A8 (_scaled_mm + dynamiczna kwantyzacja aktywacji);
VLLM_TEST_FORCE_FP8_MARLIN=1 = Marlin W8A16 (bez kwantyzacji aktywacji, lepszy dla malych M).
Idempotentny, twarde anchory.
"""
import pathlib

M = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py")


def patch(path, old, new, count=1):
    s = path.read_text()
    if new in s:
        return
    assert old in s, f"ANCHOR NOT FOUND in {path}:\n{old[:200]}"
    assert s.count(old) == count, f"anchor x{s.count(old)} != {count} in {path}"
    path.write_text(s.replace(old, new))
    print(f"patched: {path.name}: {old[:60]!r}...")


# helper na poziomie modulu (po definicji loggera)
patch(
    M,
    "logger = init_logger(__name__)\n",
    """logger = init_logger(__name__)

import os as _tq_os

_TQ_FP8_PATTERNS = [p.strip() for p in _tq_os.environ.get("VLLM_TQ_FP8_LINEARS", "").split(",") if p.strip()]
_TQ_FP8_CFG = None
_TQ_FP8_SEEN: set = set()
_TqOnlineFp8LinearMethod = None


def _tq_make_online_fp8_cls():
    \"\"\"TQ_FP8: Fp8LinearMethod w tej wersji vLLM NIE kwantyzuje online (tworzy wage FP8 i oczekuje weight_scale
    z checkpointu -> dla warstw BF16 skala zostaje niezainicjalizowana -> NaN). Podklasa: waga BF16 jak w
    UnquantizedLinearMethod, a w process_weights_after_loading kwantyzacja per-tensor BF16->FP8 (ops.scaled_fp8_quant)
    i dalej standardowa sciezka bazowa (CUTLASS/_scaled_mm W8A8 albo Marlin W8A16).\"\"\"
    import torch as _torch

    from vllm import _custom_ops as _ops
    from vllm.model_executor.layers.quantization import fp8 as _fp8
    from vllm.model_executor.parameter import ModelWeightParameter as _MWP
    from vllm.model_executor.utils import replace_parameter as _replace

    class TqOnlineFp8LinearMethod(_fp8.Fp8LinearMethod):
        def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                           params_dtype, **extra_weight_attrs):
            weight_loader = extra_weight_attrs.get("weight_loader")
            layer.logical_widths = output_partition_sizes
            layer.input_size_per_partition = input_size_per_partition
            layer.output_size_per_partition = sum(output_partition_sizes)
            layer.orig_dtype = params_dtype
            layer.weight_block_size = None
            weight = _MWP(
                data=_torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype),
                input_dim=1, output_dim=0, weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)
            self.fp8_linear = _fp8.init_fp8_linear_kernel(
                activation_quant_key=self.activation_quant_key,
                weight_quant_key=self.weight_quant_key,
                weight_shape=weight.shape,
                input_dtype=self.input_dtype,
                out_dtype=self.out_dtype,
                module_name=self.__class__.__name__,
            )
            self.use_marlin = isinstance(self.fp8_linear, _fp8.MarlinFP8ScaledMMLinearKernel)

        def process_weights_after_loading(self, layer):
            w = layer.weight.data
            assert w.dtype != _torch.float8_e4m3fn, "TQ_FP8: weight already FP8 (double processing?)"
            qweight, scale = _ops.scaled_fp8_quant(w.contiguous())  # per-tensor, scale [1]
            n = max(1, len(layer.logical_widths))
            _replace(layer, "weight", qweight)
            _replace(layer, "weight_scale", scale.reshape(1).repeat(n).contiguous())
            super().process_weights_after_loading(layer)

    return TqOnlineFp8LinearMethod


def _tq_fp8_method_for_excluded(prefix: str, layer):
    \"\"\"TQ_FP8: dla wykluczonej (BF16) warstwy liniowej pasujacej do VLLM_TQ_FP8_LINEARS zwroc Fp8LinearMethod
    (kwantyzacja online przy ladowaniu), inaczej None (=> UnquantizedLinearMethod jak dotad).\"\"\"
    global _TQ_FP8_CFG, _TqOnlineFp8LinearMethod
    if not _TQ_FP8_PATTERNS or not any(p in prefix for p in _TQ_FP8_PATTERNS):
        return None
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if _TqOnlineFp8LinearMethod is None:
        _TqOnlineFp8LinearMethod = _tq_make_online_fp8_cls()
    if _TQ_FP8_CFG is None:
        _TQ_FP8_CFG = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
        logger.info("TQ_FP8: online FP8 for excluded linears matching %s", _TQ_FP8_PATTERNS)
    key = prefix.split(".layers.")[-1] if ".layers." in prefix else prefix
    key = ".".join(key.split(".")[1:]) if key[:1].isdigit() else key
    if key not in _TQ_FP8_SEEN:
        _TQ_FP8_SEEN.add(key)
        logger.info("TQ_FP8: %s -> FP8 (%s)", key, type(layer).__name__)
    m = _TqOnlineFp8LinearMethod(_TQ_FP8_CFG)
    if _tq_os.environ.get("VLLM_TQ_FP8_MARLIN", "0") == "1":
        # Marlin W8A16 tylko dla tych warstw: flaga VLLM_TEST_FORCE_FP8_MARLIN jest globalna (przestawia tez MoE na
        # MARLIN), wiec ustawiamy ja wylacznie na czas create_weights tej warstwy (tam wybierany jest kernel).
        _orig_cw = m.create_weights

        def _cw(*a, **kw):
            prev = _tq_os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN")
            _tq_os.environ["VLLM_TEST_FORCE_FP8_MARLIN"] = "1"
            try:
                return _orig_cw(*a, **kw)
            finally:
                if prev is None:
                    _tq_os.environ.pop("VLLM_TEST_FORCE_FP8_MARLIN", None)
                else:
                    _tq_os.environ["VLLM_TEST_FORCE_FP8_MARLIN"] = prev

        m.create_weights = _cw
    return m
""",
)

# ModelOptQuantConfigBase.get_quant_method (uzywany przez ModelOptNvFp4Config - ta klasa dziedziczy metode)
patch(
    M,
    """        # handle exclusion
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
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
            return None
""",
)
# ModelOptMixedPrecisionConfig.get_quant_method (dla kompletnosci)
# ModelOptQuantConfigBase.get_quant_method (uzywany przez ModelOptNvFp4Config - ta klasa dziedziczy metode)
patch(
    M,
    """        # handle exclusion
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
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
            return None
""",
)
# ModelOptMixedPrecisionConfig.get_quant_method (dla kompletnosci)
patch(
    M,
    """        # Excluded layers
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            return None

        quant_algo = self._resolve_quant_algo(prefix)
""",
    """        # Excluded layers
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                _m = _tq_fp8_method_for_excluded(prefix, layer)  # TQ_FP8
                if _m is not None:
                    return _m
                return UnquantizedLinearMethod()
            return None

        quant_algo = self._resolve_quant_algo(prefix)
""",
)

import ast

src = M.read_text()
ast.parse(src)
assert src.count("logger = init_logger(__name__)") == 1
print("SYNTAX-OK")
print("PATCH-FP8-EXCLUDED-DONE")
