"""All-reduce NCCL (torch.distributed) dla rozmiarow decode (48 KB) i prefill (6 MB @512 tok, 25 MB @2048 tok), hidden 6144 bf16.
torchrun --nproc_per_node 4 bench_ar_nccl.py ; env: NCCL_PROTO, NCCL_ALGO, NCCL_P2P_LEVEL, NCCL_P2P_DISABLE"""
import os, time, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local); dist.init_process_group("nccl", rank=rank, world_size=world)
res = []
for ntok in (4, 6, 96, 512, 2048):
    x = torch.randn(ntok, 6144, dtype=torch.bfloat16, device="cuda")
    for _ in range(20): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    iters = 200 if ntok < 1000 else 50
    t0 = time.perf_counter()
    for _ in range(iters): dist.all_reduce(x)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    nbytes = ntok * 6144 * 2
    res.append((ntok, nbytes, us, 2 * (world - 1) / world * nbytes / us / 1e3))  # GB/s efektywne (algbw*2(n-1)/n)
if rank == 0:
    env = {k: os.environ.get(k, "-") for k in ("NCCL_PROTO", "NCCL_ALGO", "NCCL_P2P_LEVEL", "NCCL_P2P_DISABLE")}
    print("env:", env)
    for ntok, nb, us, bw in res:
        print(f"  {ntok:>5} tok {nb/2**20:7.2f} MiB : {us:9.1f} us  ({bw:5.1f} GB/s busbw)")
dist.destroy_process_group()
