#!/usr/bin/env python3
"""Run one revision batch unattended on this machine: NVIDIA GPUs (CUDA),
an Apple GPU (MPS) or the CPU.

The queues of the batch (scripts/batches.py) run in parallel: one per GPU on
CUDA machines, and two processes sharing one Apple GPU (about 1.3x the
throughput of a single process). Finished runs are skipped, so after an
interruption simply run the same command again. On macOS the machine is
kept awake with `caffeinate` while the batch runs.

  python scripts/run_batch.py --batch C1 --cache-dir /path/to/cache
  python scripts/run_batch.py --batch B  --cache-dir /path/to/cache --dry-run
"""
import argparse
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batches import make_batches  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, choices=["A1", "A2", "B", "C1", "C2", "C", "A_M5", "R_M5"])
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--parallel", type=int, default=2,
                    help="queues run at the same time on one MPS/CPU device")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    args = ap.parse_args()

    import torch
    dev = args.device
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if torch.backends.mps.is_available() else "cpu")
    if args.batch in ("C2", "C") and dev != "cuda":
        print("note: the DP runs of this batch are ~12x slower on an Apple GPU; "
              "run C2 on CUDA (Kaggle) and C1 here if possible.")

    paths = {k: os.path.abspath(getattr(args, k)) for k in
             ("cache_dir", "results_dir", "ckpt_dir", "logs_dir")}
    for k in ("results_dir", "ckpt_dir", "logs_dir"):
        os.makedirs(paths[k], exist_ok=True)
    queues = make_batches(sys.executable, paths["cache_dir"], paths["results_dir"],
                          paths["ckpt_dir"])[args.batch]

    env_base = {**os.environ, "PYTHONUNBUFFERED": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    if dev == "cuda":
        n_gpu = max(1, torch.cuda.device_count())
        envs = [{**env_base, "CUDA_VISIBLE_DEVICES": str(i % n_gpu)}
                for i in range(len(queues))]
    else:
        # one device shared by `parallel` processes: merge queues if needed
        k = max(1, min(args.parallel, len(queues)))
        merged = [[] for _ in range(k)]
        for i, q in enumerate(queues):
            merged[i % k] += q
        queues = merged
        envs = [{**env_base, "FCT_DEVICE": dev} for _ in queues]

    print(f"batch {args.batch} on {dev}: {len(queues)} parallel queue(s)")
    for i, q in enumerate(queues):
        print(f"  queue {i}: " + ", ".join(j[0] for j in q))
    if args.dry_run:
        print("\nfirst command:\n ", queues[0][0][1])
        return

    keepawake = None
    if sys.platform == "darwin":  # keep the Mac awake while this process runs
        keepawake = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])

    t0 = time.time()
    status = {}

    def run_queue(qi, jobs, env):
        for name, cmd, needs in jobs:
            while needs and not all(os.path.exists(p) for p in needs):
                time.sleep(60)
            log = os.path.join(paths["logs_dir"], f"{args.batch}_{name}.log")
            t = time.time()
            print(f"[{(t - t0) / 3600:5.2f} h] queue {qi} start {name}  (log: {log})",
                  flush=True)
            start = os.path.getsize(log) if os.path.exists(log) else 0
            with open(log, "a") as fh:
                rc = subprocess.call(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT,
                                     env=env, cwd=REPO)
            with open(log) as fh:  # only this attempt's output counts
                fh.seek(start)
                failed = rc != 0 or "FAILED" in fh.read()
            status[name] = ("FAILED" if failed else "ok") + f" ({(time.time() - t) / 3600:.2f} h)"
            print(f"[{(time.time() - t0) / 3600:5.2f} h] queue {qi} {name}: {status[name]}",
                  flush=True)

    threads = [threading.Thread(target=run_queue, args=(i, q, e))
               for i, (q, e) in enumerate(zip(queues, envs))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if keepawake:
        keepawake.terminate()
    print(f"\nbatch {args.batch} finished in {(time.time() - t0) / 3600:.2f} h")
    for k, v in status.items():
        print(f"  {k:14s} {v}")
    if any(v.startswith("FAILED") for v in status.values()):
        print("Some jobs failed: read their logs, fix, and run the same command again "
              "(finished runs are skipped).")


if __name__ == "__main__":
    main()
