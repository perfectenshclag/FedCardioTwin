#!/usr/bin/env python3
"""Measured compute footprint for the paper (Reviewer 3, comment 4).

Reports hardware/software, per-record inference latency (one 10 s 12-lead
record = 7 sliding windows) on GPU and CPU, batch throughput, peak GPU
memory of one local training step at batch 128, the per-patient twin update
time, and the cost of secure-aggregation masking for a federation of this
size. Per-round local training time at every hospital is logged by the
federated runs themselves (results/fed_*.json -> 'timing').

  python scripts/profile_compute.py --out results/profile.json
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from fedcardiotwin.configs import TwinConfig
from fedcardiotwin.data.dataset import CROP_LEN
from fedcardiotwin.models import PatientTwin, build_model
from fedcardiotwin.train.trainer import EVAL_STRIDE
from fedcardiotwin.twin.loop import _update_twin


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _windows(x):
    w = x.unfold(-1, CROP_LEN, EVAL_STRIDE)
    B, C, W, L = w.shape
    return w.permute(0, 2, 1, 3).reshape(B * W, C, L)


@torch.no_grad()
def latency(model, device, n_rep=200, threads=None):
    if threads:
        torch.set_num_threads(threads)
    model.to(device).eval()
    x = _windows(torch.randn(1, 12, 1000, device=device))
    for _ in range(20):
        model(x)
    _sync(device)
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        torch.sigmoid(model(x)).mean(0)
        _sync(device)
        ts.append(time.perf_counter() - t0)
    return {"median_ms": 1e3 * float(np.median(ts)),
            "p95_ms": 1e3 * float(np.percentile(ts, 95))}


@torch.no_grad()
def throughput(model, device, batch=128, n_rep=20):
    model.to(device).eval()
    x = _windows(torch.randn(batch, 12, 1000, device=device))
    ctx = (torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda"
           else torch.no_grad())
    with ctx:
        for _ in range(3):
            model(x)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(n_rep):
            model(x)
        _sync(device)
    return batch * n_rep / (time.perf_counter() - t0)


def train_step_memory(model, device, batch=128):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    x = torch.randn(batch, 12, CROP_LEN, device=device)
    y = torch.randint(0, 2, (batch, 26), device=device).float()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(12):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with (torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda"
              else torch.enable_grad()):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    out = {"step_ms_median": 1e3 * float(np.median(ts[2:]))}
    if device.type == "cuda":
        out["peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    elif device.type == "mps":
        out["driver_mem_mib"] = torch.mps.driver_allocated_memory() / 2 ** 20
    return out


def twin_update_time(device, n_rep=30):
    cfg = TwinConfig()
    base = build_model("inception1d", 26).to(device).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    twin = PatientTwin(base, hidden=cfg.adapter_hidden).to(device)
    buf = [(torch.randn(12, 1000), torch.randint(0, 2, (26,)).float())
           for _ in range(cfg.replay_size)]
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        _update_twin(twin, base, buf, device, cfg.update_steps, cfg.update_lr,
                     cfg.prox_weight)
        _sync(device)
        ts.append(time.perf_counter() - t0)
    n_tw = sum(p.numel() for p in twin.trainable_parameters())
    return {"update_ms_median": 1e3 * float(np.median(ts[3:])),
            "twin_trainable_params": int(n_tw)}


def secagg_masking(n_clients=6, n_params=485_000, n_rep=5):
    """Pairwise additive masking (Bonawitz et al.) in the 2^32 ring: each
    client expands K-1 shared seeds into masks of the update's size and adds
    or subtracts them; the server's sum cancels all masks (no dropouts)."""
    rng = np.random.default_rng(0)
    upd = [rng.integers(0, 2 ** 32, n_params, dtype=np.uint64) for _ in range(n_clients)]
    seeds = {(i, j): int(rng.integers(0, 2 ** 31)) for i in range(n_clients)
             for j in range(i + 1, n_clients)}
    t_mask = []
    masked = []
    for _ in range(n_rep):
        masked = []
        t0 = time.perf_counter()
        for i in range(n_clients):
            m = upd[i].copy()
            for j in range(n_clients):
                if i == j:
                    continue
                s = seeds[(min(i, j), max(i, j))]
                mask = np.random.default_rng(s).integers(0, 2 ** 32, n_params,
                                                         dtype=np.uint64)
                m = (m + mask) % 2 ** 32 if i < j else (m - mask) % 2 ** 32
            masked.append(m)
        t_mask.append((time.perf_counter() - t0) / n_clients)
    t0 = time.perf_counter()
    total = np.zeros(n_params, dtype=np.uint64)
    for m in masked:
        total = (total + m) % 2 ** 32
    t_sum = time.perf_counter() - t0
    ok = np.array_equal(total, sum(upd) % 2 ** 32)
    return {"clients": n_clients, "params": n_params,
            "client_mask_ms": 1e3 * float(np.median(t_mask)),
            "server_unmask_sum_ms": 1e3 * t_sum, "masks_cancel": bool(ok),
            "extra_upload_bytes_per_client": 32 * (n_clients - 1) * 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/profile.json")
    args = ap.parse_args()
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu_model"] = next(l.split(":", 1)[1].strip() for l in f
                                     if l.startswith("model name"))
    except Exception:
        pass
    model = build_model("inception1d", 26)
    out = {"env": info,
           "params": sum(p.numel() for p in model.parameters())}
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda"] = torch.version.cuda
        out["gpu_latency_per_record"] = latency(build_model("inception1d", 26), dev)
        out["gpu_throughput_records_s"] = throughput(build_model("inception1d", 26), dev)
        out["gpu_train_step_b128"] = train_step_memory(build_model("inception1d", 26), dev)
        out["gpu_twin_update"] = twin_update_time(dev)
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        import subprocess
        info["gpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True).stdout.strip() + " GPU (MPS, FP32)"
        out["gpu_latency_per_record"] = latency(build_model("inception1d", 26), dev)
        out["gpu_throughput_records_s"] = throughput(build_model("inception1d", 26), dev)
        out["gpu_train_step_b128"] = train_step_memory(build_model("inception1d", 26), dev)
        out["gpu_twin_update"] = twin_update_time(dev)
    cpu = torch.device("cpu")
    out["cpu_latency_per_record_1thread"] = latency(build_model("inception1d", 26), cpu,
                                                    n_rep=50, threads=1)
    out["cpu_latency_per_record_allthreads"] = latency(build_model("inception1d", 26), cpu,
                                                       n_rep=50,
                                                       threads=os.cpu_count())
    out["cpu_twin_update"] = twin_update_time(cpu, n_rep=10)
    out["secagg"] = secagg_masking()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
