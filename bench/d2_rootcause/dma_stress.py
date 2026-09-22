#!/usr/bin/env python3
"""Synthetic reproducer for the D2 hang, independent of vLLM: on ONE GPU (CUDA_VISIBLE_DEVICES) run heavy matmuls on the
default stream while a side stream copies many small GPU segments (22528 B = one NVFP4 KV page) into pinned host memory
(cudaMemcpyAsync via torch copy_), plus periodic host->GPU loads. Prints progress every ~5 s; a stall = no output.
Usage: dma_stress.py <seconds> [segments_per_iter=512] [mode=d2h|h2d|both|compute]
"""
import os, sys, time, torch

dur = int(sys.argv[1]) if len(sys.argv) > 1 else 300
nseg = int(sys.argv[2]) if len(sys.argv) > 2 else 512
mode = sys.argv[3] if len(sys.argv) > 3 else "d2h"
SEG = 22528
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
gpu = torch.randint(0, 255, (65536, SEG), dtype=torch.uint8, device=dev)        # 1.4 GB "KV cache"
host = torch.empty((65536, SEG), dtype=torch.uint8, pin_memory=True)           # 1.4 GB pinned
a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
side = torch.cuda.Stream()
g = torch.Generator(device="cpu"); g.manual_seed(int(os.environ.get("SEED", "1")))
t0 = time.time(); last = t0; it = 0; nbytes = 0
name = torch.cuda.get_device_name(0)
print(f"start {name} mode={mode} nseg={nseg} dur={dur}s", flush=True)
while time.time() - t0 < dur:
    if mode != "copy_only":
        for _ in range(4):
            b = a @ a
    if mode in ("d2h", "both", "copy_only"):
        src = torch.randint(0, gpu.shape[0], (nseg,), generator=g).tolist()
        dst = torch.randint(0, host.shape[0], (nseg,), generator=g).tolist()
        with torch.cuda.stream(side):
            side.wait_stream(torch.cuda.current_stream())
            for i in range(nseg):
                host[dst[i]].copy_(gpu[src[i]], non_blocking=True)
        nbytes += nseg * SEG
    if mode in ("h2d", "both"):
        src = torch.randint(0, host.shape[0], (nseg,), generator=g).tolist()
        dst = torch.randint(0, gpu.shape[0], (nseg,), generator=g).tolist()
        with torch.cuda.stream(side):
            for i in range(nseg):
                gpu[dst[i]].copy_(host[src[i]], non_blocking=True)
        nbytes += nseg * SEG
    torch.cuda.synchronize()
    it += 1
    if time.time() - last >= 5:
        last = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] it={it} copied={nbytes/2**30:.1f} GiB ({nbytes/2**30/(time.time()-t0):.2f} GiB/s) matmul ok", flush=True)
print(f"done it={it} copied={nbytes/2**30:.1f} GiB", flush=True)
