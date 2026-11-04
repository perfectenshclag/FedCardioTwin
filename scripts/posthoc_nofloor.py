#!/usr/bin/env python3
"""Conformal variants WITHOUT the federated floor (per-hospital calibration
only): local CRC at alpha_cal = 0.05 and the size-adaptive UCB rule at
alpha = 0.10, delta = 0.10. Same protocol as posthoc.py (3 seeds x 100
calibration/test re-splits of each hospital's pooled val+test predictions).

  python scripts/posthoc_nofloor.py --results-dir results --family fed_fedper_ref
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from fedcardiotwin.conformal import crc
from posthoc import load_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--family", default="fed_fedper_ref")
    ap.add_argument("--n-splits", type=int, default=100)
    args = ap.parse_args()
    rows = []
    for s in range(3):
        p = f"{args.results_dir}/preds/{args.family}_seed{s}.npz"
        if not os.path.exists(p):
            continue
        pr = load_preds(p)
        rng = np.random.RandomState(100 + s)
        pooled = {k: (np.concatenate([v[0], v[2]]),
                      crc.record_loss_matrix(np.concatenate([v[0], v[2]]),
                                             np.concatenate([v[1], v[3]])), len(v[0]))
                  for k, v in pr.items()}
        for _ in range(args.n_splits):
            for k, (P, L, nc) in pooled.items():
                perm = rng.permutation(len(P))
                cal, test = perm[:nc], perm[nc:]
                risk, n = crc._risk(L[cal])
                for mode, lam in (("local@0.05 (no floor)", crc.crc_from_curve(risk, n, 0.05)),
                                  ("UCB (no floor)", crc.ucb_from_curve(risk, n, 0.1, 0.1))):
                    j = int(round(lam * 200))
                    rows.append({"seed": s, "site": k, "mode": mode,
                                 "fnr": float(np.nanmean(L[test][:, j])),
                                 "size": crc._set_size(P[test], lam)})
    d = pd.DataFrame(rows)
    d["viol"] = d.fnr > 0.1
    out = f"{args.results_dir}/revision/conformal_nofloor.csv"
    d.to_csv(out, index=False)
    print(d.groupby(["mode", "site"]).agg(fnr=("fnr", "mean"), viol=("viol", "mean"),
                                          size=("size", "mean")).round(3))


if __name__ == "__main__":
    main()
