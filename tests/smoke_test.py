#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data (CPU, ~2 min).
Verifies every code path the paper needs: cache format -> centralized ->
all FL strategies -> twin loop -> conformal. No dataset required.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch

from fedcardiotwin.configs import fast_preset
from fedcardiotwin.data.dataset import PTBXLStreams, load_clients
from fedcardiotwin.fl.engine import run_federated
from fedcardiotwin.models import build_model
from fedcardiotwin.train.trainer import evaluate, predict, train_model
from fedcardiotwin.twin.loop import run_twin_evaluation
from fedcardiotwin.conformal.crc import run_conformal
from fedcardiotwin.utils import seed_everything

N_CLASSES = 26
rng = np.random.RandomState(0)


def make_synthetic_cache(root):
    for client in ["SynthA", "SynthB", "SynthC"]:
        d = os.path.join(root, client)
        os.makedirs(d)
        n = 120
        X = rng.randn(n, 12, 1000).astype(np.float16)
        Y = (rng.rand(n, N_CLASSES) < 0.15).astype(np.uint8)
        Y[Y.sum(1) == 0, 0] = 1  # every record has >=1 label
        np.save(os.path.join(d, "X.npy"), X)
        np.save(os.path.join(d, "Y.npy"), Y)
    # Track-B synthetic: 12 patients x 3 records, 5 classes
    d = os.path.join(root, "PTBXL_TRACKB")
    os.makedirs(d)
    n = 60
    np.save(os.path.join(d, "X.npy"), rng.randn(n, 12, 1000).astype(np.float16))
    Y = (rng.rand(n, 5) < 0.3).astype(np.uint8)
    Y[Y.sum(1) == 0, 0] = 1
    np.save(os.path.join(d, "Y.npy"), Y)
    pd.DataFrame({
        "ecg_id": range(n),
        "patient_id": [i // 3 for i in range(n)],
        "date": [f"2020-01-{(i % 28) + 1:02d}" for i in range(n)],
        "strat_fold": [9 if i // 3 >= 8 else 1 for i in range(n)],
    }).to_csv(os.path.join(d, "meta.csv"), index=False)


def main():
    seed_everything(0)
    device = torch.device(os.environ.get("FCT_DEVICE", "cpu"))
    root = tempfile.mkdtemp(prefix="fct_smoke_")
    try:
        make_synthetic_cache(root)
        cfg = fast_preset()
        cfg.fl.batch_size = 32
        cfg.fl.rounds = 1
        cfg.fl.local_only_epochs = 1
        cfg.central.batch_size = 32
        cfg.central.epochs = 1
        cfg.twin.update_steps = 1

        clients = load_clients(root, ["SynthA", "SynthB", "SynthC"])

        # centralized path
        model = build_model("inception1d", N_CLASSES)
        train_model(model, clients[0].loader("train", 32, num_workers=0),
                    device, epochs=1)
        m, _ = evaluate(model, clients[0].loader("val", 32, num_workers=0),
                        clients[0].loader("test", 32, num_workers=0), device)
        assert np.isfinite(m["f1"]), m
        print(f"[ok] centralized inception1d: {m}")

        # resnet path
        rmodel = build_model("resnet1d", N_CLASSES)
        out = rmodel(torch.randn(4, 12, 250))
        assert out.shape == (4, N_CLASSES)
        print("[ok] resnet1d forward")

        # every FL strategy; keep the deployed states of the last one
        for strategy in ["local", "fedavg", "fedprox", "fedbn", "fedavgm",
                         "ditto", "fedper", "fedrep", "fedbabu"]:
            res, history, comm, gmodel, deployed, preds = run_federated(
                clients, N_CLASSES, cfg.fl, device, strategy=strategy, seed=0)
            assert "MEAN" in res and np.isfinite(res["MEAN"]["f1"])
            assert len(history) >= 1 and "cum_upload_mb" in history[-1]
            assert set(deployed) == {c.name for c in clients} == set(preds)
            assert "comm" in res and "timing" in res and "snapshots" in res
            print(f"[ok] {strategy}: mean F1={res['MEAN']['f1']:.3f} "
                  f"comm={comm:.1f}MB history={len(history)}")

        # twin loop — must produce every Table-3 metric
        sdb = PTBXLStreams(root)
        base = build_model("inception1d", 5)
        tres, arrays = run_twin_evaluation(base, sdb.eval_streams(), sdb.X, sdb.Y,
                                           device, cfg.twin)
        for key in ("cold_auroc", "warm_auroc", "personalization_gain",
                    "per_step", "alert_auroc", "backward_transfer"):
            assert key in tres, key
        assert tres["n_patients"] > 0
        assert len(arrays["alert"]) == len(arrays["changed"]) == tres["n_eval_records"]
        print(f"[ok] twin loop: {tres}")

        # conformal — evaluated on the *deployed* per-client models
        cal, test = {}, {}
        for c in clients:
            m_dep = build_model("inception1d", N_CLASSES)
            m_dep.load_state_dict(deployed[c.name])
            pv, yv = predict(m_dep, c.loader("val", 32, num_workers=0), device)
            pt, yt = predict(m_dep, c.loader("test", 32, num_workers=0), device)
            cal[c.name], test[c.name] = (pv, yv), (pt, yt)
        cres = run_conformal(cal, test, alpha=0.2)
        for name, r in cres["clients"].items():
            assert r["federated"]["fnr"] <= 0.65  # loose sanity bound
        print(f"[ok] conformal: fed lambda={cres['federated_lambda']:.3f}")

        print("\nALL SMOKE TESTS PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
