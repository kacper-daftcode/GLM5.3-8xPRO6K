"""Unit test: Fp8LinearMethod (online quant BF16->FP8) vs BF16 dla Replicated/Column/Row/MergedColumn na 1 GPU (TP=1).
Uruchamiac w obrazie: python3 fp8_linear_unit.py [--marlin]"""
import os, sys, torch

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
if "--marlin" in sys.argv:
    os.environ["VLLM_TEST_FORCE_FP8_MARLIN"] = "1"
from vllm.config import VllmConfig, ModelConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel

mc = ModelConfig(model="/model", tokenizer="/model", trust_remote_code=True, dtype="bfloat16", seed=0, max_model_len=4096)
cfg = VllmConfig(model_config=mc)
with set_current_vllm_config(cfg):
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29555", local_rank=0, backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1)
    from vllm.model_executor.layers.linear import ReplicatedLinear, ColumnParallelLinear, RowParallelLinear, MergedColumnParallelLinear
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    q = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_dtype(torch.bfloat16)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    tests = [
        ("ReplicatedLinear 6144->576", lambda qc: ReplicatedLinear(6144, 576, bias=False, quant_config=qc, prefix="t.repl"), [torch.randn(576, 6144)]),
        ("ColumnParallelLinear 2048->3072", lambda qc: ColumnParallelLinear(2048, 3072, bias=False, quant_config=qc, prefix="t.col"), [torch.randn(3072, 2048)]),
        ("RowParallelLinear 4096->6144", lambda qc: RowParallelLinear(4096, 6144, bias=False, quant_config=qc, prefix="t.row"), [torch.randn(6144, 4096)]),
        ("MergedColumnParallelLinear 6144->[3072,3072]", lambda qc: MergedColumnParallelLinear(6144, [3072, 3072], bias=False, quant_config=qc, prefix="t.merged"), [torch.randn(3072, 6144), torch.randn(3072, 6144)]),
    ]
    for name, mk, ws in tests:
        with torch.device(dev):
            ref = mk(None)
            fp8 = mk(q)
        ws = [(w * 0.02).to(torch.bfloat16) for w in ws]
        for layer in (ref, fp8):
            p = layer.weight
            if len(ws) == 1:
                p.weight_loader(p, ws[0].to(dev)) if hasattr(p, "weight_loader") else p.data.copy_(ws[0])
            else:
                for i, w in enumerate(ws):
                    p.weight_loader(p, w.to(dev), i)
            layer.quant_method.process_weights_after_loading(layer)
        for M in (6, 512):
            x = torch.randn(M, ref.input_size, dtype=torch.bfloat16, device=dev)
            with torch.no_grad():
                y_ref = ref(x)[0].float(); y_fp8 = fp8(x)[0].float()
            err = (y_ref - y_fp8).abs().max().item(); rel = err / (y_ref.abs().max().item() + 1e-9)
            nan = torch.isnan(y_fp8).any().item() or torch.isinf(y_fp8).any().item()
            print(f"{name:46s} M={M:4d}: max|d|={err:.4f} rel={rel:.4f} nan/inf={nan}  kernel={type(getattr(fp8.quant_method, 'fp8_linear', None)).__name__}  w.dtype={fp8.weight.dtype} w.shape={tuple(fp8.weight.shape)} scale={tuple(fp8.weight_scale.shape) if hasattr(fp8,'weight_scale') else None}")
