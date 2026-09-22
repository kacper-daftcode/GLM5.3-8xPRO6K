"""Patch vLLM (obraz nvfp4-fi618): B12X MoE uzywalny z MTP + tryb hybrydowy B12X(decode)/CUTLASS(prefill).

1) oracle/unquantized.py: `--moe-backend flashinfer_b12x` nie jest wspierany przez niekwantyzowane MoE (warstwa MTP
   drafta w checkpoincie incoai jest BF16) -> zamiast wyjatku spadamy do auto (jak 'humming').
2) experts/flashinfer_b12x_moe.py: env VLLM_B12X_HYBRID_MAX_TOKENS=N (N>0) wlacza hybryde:
   - M <= N tokenow -> kernel B12X (static/micro, workspace tylko dla N tokenow, bez workspace "dynamic"),
   - M >  N tokenow -> FlashInfer CUTLASS grouped GEMM (ten sam, co domyslny backend FLASHINFER_CUTLASS).
   W hybrydzie NIE wypiekamy weight_scale_2 do skal blokowych (CUTLASS potrzebuje oryginalnych, swizzlowanych skal),
   B12X dostaje w1_alpha=weight_scale_2 i input_global_scale=1.0 (API FlashInfer >= 0.6.18). Bez env = stare zachowanie.
Idempotentny, twarde anchory.
"""
import pathlib

V = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")


def patch(path, old, new, count=1):
    s = path.read_text()
    if new in s:
        return
    assert old in s, f"ANCHOR NOT FOUND in {path}:\n{old[:200]}"
    assert s.count(old) == count, f"anchor x{s.count(old)} != {count} in {path}"
    path.write_text(s.replace(old, new))
    print(f"patched: {path.name}: {old[:60]!r}...")


# ---- 1. unquantized MoE: b12x -> auto -------------------------------------------
U = V / "model_executor/layers/fused_moe/oracle/unquantized.py"
patch(
    U,
    """    if runner_backend not in ["auto", "humming"]:
        requested_backend = map_unquantized_backend(runner_backend)""",
    """    # TQ_B12X: 'flashinfer_b12x' dotyczy tylko NVFP4 MoE; niekwantyzowane warstwy (np. MoE drafta MTP) -> auto.
    if runner_backend not in ["auto", "humming", "flashinfer_b12x"]:
        requested_backend = map_unquantized_backend(runner_backend)""",
)

# ---- 2. B12X hybryda ---------------------------------------------------------------
B = V / "model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"

patch(
    B,
    """from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    has_flashinfer_b12x_moe,
)
""",
    """from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    flashinfer_cutlass_fused_moe,
    has_flashinfer_b12x_moe,
)

import os as _os

# TQ_B12X: >0 => hybryda: B12X dla M <= N tokenow, FlashInfer CUTLASS dla M > N. 0 => czysty B12X.
_B12X_HYBRID_MAX_TOKENS = int(_os.environ.get("VLLM_B12X_HYBRID_MAX_TOKENS", "0"))
# TQ_B12X: statyczny workspace B12X wspoldzielony przez wszystkie warstwy MoE (klucz: ksztalt problemu)
_B12X_SHARED_WS: dict = {}
""",
)

# 2a. stan hybrydy w __init__
patch(
    B,
    """        # Lazily created on first apply() call.
        self._wrapper: Any | None = None
        self.w1_sf_mma: torch.Tensor | None = None
        self.w2_sf_mma: torch.Tensor | None = None
""",
    """        # Lazily created on first apply() call.
        self._wrapper: Any | None = None
        self.w1_sf_mma: torch.Tensor | None = None
        self.w2_sf_mma: torch.Tensor | None = None

        # TQ_B12X hybryda
        self.hybrid_max_tokens = _B12X_HYBRID_MAX_TOKENS
        self.tp_size = moe_config.moe_parallel_config.tp_size
        self.tp_rank = moe_config.moe_parallel_config.tp_rank
        self._b12x_w1_alpha: torch.Tensor | None = None
        self._b12x_w2_alpha: torch.Tensor | None = None
        self._b12x_ones: torch.Tensor | None = None
        self._c_g1_alphas: torch.Tensor | None = None
        self._c_g2_alphas: torch.Tensor | None = None
        self._c_a1_gscale: torch.Tensor | None = None
        self._c_a2_gscale: torch.Tensor | None = None
        self._b12x_ws: Any | None = None
        self._b12x_views: Any | None = None
""",
)

# 2b. process_weights_after_loading: w hybrydzie bez wypiekania skal
patch(
    B,
    """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Normalise block scales to absorb the per-expert weight global scale
        # (w_gs).  vLLM's NVFP4 convention stores:""",
    """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.hybrid_max_tokens > 0:
            return self._process_weights_hybrid(layer)
        # Normalise block scales to absorb the per-expert weight global scale
        # (w_gs).  vLLM's NVFP4 convention stores:""",
)

patch(
    B,
    """    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard
""",
    """    def _process_weights_hybrid(self, layer: torch.nn.Module) -> None:
        \"\"\"TQ_B12X: skale blokowe zostaja oryginalne (swizzlowane, dla CUTLASS); B12X dostaje widok MMA
        oraz w1_alpha = weight_scale_2 (bez wypiekania) i input_global_scale = 1.0 (dynamiczna kwantyzacja wejscia).\"\"\"
        assert self.w1_scale is not None and self.w2_scale is not None
        assert self.g1_alphas is not None and self.g2_alphas is not None
        assert self.a1_gscale is not None and self.a2_gscale is not None
        dev = layer.w13_weight.device
        e = self.num_local_experts
        # B12X: czyste weight_scale_2 (kopie - g1/g2_alphas to te same tensory co parametry warstwy)
        self._b12x_w1_alpha = self.g1_alphas.detach().float().clone().contiguous()
        self._b12x_w2_alpha = self.g2_alphas.detach().float().clone().contiguous()
        self._b12x_ones = torch.ones(e, device=dev, dtype=torch.float32)
        self._fc2_input_scale = self._b12x_ones
        # CUTLASS: jak FlashInferExperts.process_weights_after_loading (g = weight_scale_2 * input_scale),
        # ale na kopiach, zeby nie psuc alph B12X. a1/a2_gscale = 1/input_scale (globalne max po ekspertach).
        # ksztalt (E,) jak w FlashInferExperts (kernel CUTLASS wymaga skalara 0-dim albo (num_experts,))
        a1_gs = self.a1_gscale.detach().float().reshape(-1)
        a2_gs = self.a2_gscale.detach().float().reshape(-1)
        self._c_a1_gscale = a1_gs.min().expand(e).contiguous()
        self._c_a2_gscale = a2_gs.min().expand(e).contiguous()
        self._c_g1_alphas = (self._b12x_w1_alpha / self._c_a1_gscale).contiguous()
        self._c_g2_alphas = (self._b12x_w2_alpha / self._c_a2_gscale).contiguous()

        num_experts_w1, m1, k1_sf = self.w1_scale.shape
        self.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w1_scale.reshape(num_experts_w1 * m1, k1_sf), m=m1, k=k1_sf * 16, num_groups=num_experts_w1
        )
        num_experts_w2, m2, k2_sf = self.w2_scale.shape
        self.w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w2_scale.reshape(num_experts_w2 * m2, k2_sf), m=m2, k=k2_sf * 16, num_groups=num_experts_w2
        )
        # Widok MMA to strided view na tych samych bajtach co skale swizzlowane (zero kopii).
        assert self.w1_sf_mma.data_ptr() == self.w1_scale.data_ptr()

        # Jeden statyczny workspace B12X wspoldzielony przez wszystkie warstwy (B12xMoEWrapper alokuje ~540 MiB
        # NA WARSTWE -> 75 warstw = 40 GiB = OOM). Warstwy licza sie sekwencyjnie na jednym strumieniu.
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _get_weight_views,
            allocate_sm120_moe_workspace,
        )

        n = self.intermediate_size_per_partition
        k = self.hidden_dim
        ws_key = (e, k, n, self.topk, str(dev), self.hybrid_max_tokens, self._activation_str)
        ws = _B12X_SHARED_WS.get(ws_key)
        if ws is None:
            ws = allocate_sm120_moe_workspace(
                state_E=e, weight_E=e, max_rows=self.hybrid_max_tokens * self.topk, k=k, n=n, num_topk=self.topk,
                device=dev, quant_mode="nvfp4", backend="static", activation=self._activation_str,
            )
            _B12X_SHARED_WS[ws_key] = ws
        self._b12x_ws = ws
        # Widoki wag per warstwa (cache FlashInfer po data_ptr; skale bez kopii). w1_alpha juz "zfoldowane" (x1.0).
        self._b12x_views = _get_weight_views(
            w1_fp4=layer.w13_weight, w1_blockscale=self.w1_sf_mma, w2_fp4=layer.w2_weight, w2_blockscale=self.w2_sf_mma,
            w1_alphas=self._b12x_w1_alpha, w2_alphas=self._b12x_w2_alpha, n=n, k=k, activation_precision="fp4", quant_mode="nvfp4",
        )
        from vllm.logger import init_logger

        init_logger(__name__).info_once(
            "TQ_B12X hybrid: B12X for M<=%d tokens (shared static workspace %d rows), FlashInfer CUTLASS above (tp=%d/%d, experts=%d)",
            self.hybrid_max_tokens, self.hybrid_max_tokens * self.topk, self.tp_rank, self.tp_size, e,
        )

    def _apply_b12x_hybrid(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        \"\"\"TQ_B12X: decode = kernel B12X static/micro przez launch_sm120_moe ze wspolnym workspace i gotowymi widokami.\"\"\"
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import launch_sm120_moe

        launch_sm120_moe(
            a=hidden_states,
            topk_ids=topk_ids.to(torch.int32),
            topk_weights=topk_weights,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self._b12x_w1_alpha,
            fc2_input_scale=self._b12x_ones,
            input_global_scale=self._b12x_ones,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self._b12x_w2_alpha,
            num_experts=self.global_num_experts,
            top_k=self.topk,
            num_local_experts=self.num_local_experts,
            scatter_output=output,
            activation=self._activation_str,
            activation_precision="fp4",
            quant_mode="nvfp4",
            _workspace=self._b12x_ws,
            _weight_views=self._b12x_views,
        )

    def _apply_cutlass(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
    ) -> None:
        \"\"\"TQ_B12X: sciezka prefill = FlashInfer CUTLASS grouped GEMM (identyczna z backendem FLASHINFER_CUTLASS).\"\"\"
        from flashinfer.fused_moe.core import ActivationType

        from vllm import _custom_ops as ops

        act = {MoEActivation.SILU: ActivationType.Swiglu, MoEActivation.RELU2_NO_MUL: ActivationType.Relu2}[activation]
        a1q, a1q_sf = ops.scaled_fp4_quant(hidden_states, self._c_a1_gscale)
        assert self.w1_scale is not None and self.w2_scale is not None
        flashinfer_cutlass_fused_moe(
            input=a1q,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=w1.view(torch.long),
            fc2_expert_weights=w2.view(torch.long),
            output=output,
            output_dtype=self.out_dtype,
            quant_scales=[
                self._c_a1_gscale,
                self.w1_scale.view(torch.int32),
                self._c_g1_alphas,
                self._c_a2_gscale,
                self.w2_scale.view(torch.int32),
                self._c_g2_alphas,
            ],
            input_sf=a1q_sf,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=1,
            ep_rank=0,
            activation_type=act,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard
""",
)

# 2c. wrapper: w hybrydzie workspace tylko dla hybrid_max_tokens (bez "dynamic")
patch(
    B,
    """        self._wrapper = B12xMoEWrapper(
            num_experts=self.global_num_experts,
            top_k=self.topk,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size_per_partition,
            use_cuda_graph=True,
            max_num_tokens=self.max_num_tokens,""",
    """        self._wrapper = B12xMoEWrapper(
            num_experts=self.global_num_experts,
            top_k=self.topk,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size_per_partition,
            use_cuda_graph=True,
            max_num_tokens=(
                min(self.max_num_tokens, self.hybrid_max_tokens)
                if self.hybrid_max_tokens > 0
                else self.max_num_tokens
            ),  # TQ_B12X""",
)

# 2d. apply: rozgalezienie po liczbie tokenow + alphy hybrydy
patch(
    B,
    """        self._ensure_wrapper()
        wrapper = self._wrapper
        assert wrapper is not None

        wrapper_output = wrapper.run(
            x=hidden_states,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
        )
        output.copy_(wrapper_output)""",
    """        # TQ_B12X hybryda: duze M (prefill) -> CUTLASS, male M (decode) -> B12X ze wspolnym workspace
        if self.hybrid_max_tokens > 0:
            if hidden_states.shape[0] > self.hybrid_max_tokens:
                return self._apply_cutlass(output, hidden_states, w1, w2, topk_weights, topk_ids, activation)
            return self._apply_b12x_hybrid(output, hidden_states, w1, w2, topk_weights, topk_ids)

        self._ensure_wrapper()
        wrapper = self._wrapper
        assert wrapper is not None

        wrapper_output = wrapper.run(
            x=hidden_states,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
        )
        output.copy_(wrapper_output)""",
)

import ast

for f in (U, B):
    ast.parse(f.read_text())
print("SYNTAX-OK")
print("PATCH-B12X-HYBRID-DONE")
