#!/usr/bin/env python3
"""CPU-only post-hoc analyses for the J-BHI revision.

Everything here runs from the saved predictions (results/preds/*.npz), the
result JSONs and the twin arrays, so no GPU and no retraining is needed:

  per-class metrics (AUROC, F1, sensitivity, PPV at the validation-tuned
  threshold) per hospital and pooled; key clinical classes; paired bootstrap
  CIs and Wilcoxon tests; repeated-split conformal analyses (calibration
  level and risk target sweeps, size-adaptive UCB rule, calibration-size
  curves, class-wise and weighted risk control); twin alert baselines and
  gain vs. number of prior records; communication, timing, round-budget and
  stopping-rule tables; PTB-XL patient-disjoint check.

  python scripts/posthoc.py --results-dir results --cache-dir data/cache \\
      --out-dir results/revision
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fedcardiotwin.conformal import crc
from fedcardiotwin.train.metrics import macro_auroc, tune_thresholds

SITES = ["CPSC", "CPSC-Extra", "Georgia", "Chapman", "Ningbo", "PTB-XL"]
KEY_CLASSES = ["AF", "AFL", "IAVB", "CLBBB|LBBB", "CRBBB|RBBB", "LAnFB",
               "PVC|VPB", "Brady", "SB", "LQT"]
# illustrative clinical tiers for the weighted-FNR example (set by clinicians
# in practice): rhythm/conduction findings that change management promptly
CRITICAL = ["AF", "AFL", "CLBBB|LBBB", "PVC|VPB", "LQT", "Brady"]


def load_preds(path):
    z = np.load(path)
    names = sorted({k.split("__")[0] for k in z.files})
    return {n: (z[f"{n}__pv"], z[f"{n}__yv"], z[f"{n}__pt"], z[f"{n}__yt"])
            for n in names}


def runs_by_family(preds_dir):
    fam = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(preds_dir, "*.npz"))):
        m = re.match(r"(.+)_seed(\d+)\.npz$", os.path.basename(p))
        if m:
            fam[m.group(1)][int(m.group(2))] = p
    return fam


def class_names(eval_repo):
    try:
        from fedcardiotwin.data.labels import ScoredLabelSpace
        return ScoredLabelSpace(eval_repo).classes
    except Exception:
        return [f"c{i}" for i in range(26)]


# ------------------------------------------------------------ per class

def per_class_rows(family, seed, preds, classes):
    rows = []
    for site, (pv, yv, pt, yt) in preds.items():
        th = tune_thresholds(yv, pv)
        hard = pt >= th[None, :]
        for c, cname in enumerate(classes):
            col = yt[:, c]
            npos = int(col.sum())
            tp = int((hard[:, c] & (col > 0)).sum())
            fp = int((hard[:, c] & (col == 0)).sum())
            auc = (roc_auc_score(col, pt[:, c]) if 0 < npos < len(col)
                   else np.nan)
            sens = tp / npos if npos else np.nan
            ppv = tp / (tp + fp) if (tp + fp) else np.nan
            f1 = (2 * tp / (2 * tp + fp + (npos - tp))) if npos else np.nan
            rows.append({"family": family, "seed": seed, "site": site,
                         "class": cname, "n_pos": npos, "auroc": auc,
                         "sens": sens, "ppv": ppv, "f1": f1,
                         "threshold": float(th[c])})
    return rows


def pooled_key_rows(family, seed, preds, classes):
    """Each record scored by its own hospital's deployed model at that
    hospital's validation-tuned threshold, then pooled across hospitals."""
    rows = []
    hard_all, prob_all, y_all = [], [], []
    for site, (pv, yv, pt, yt) in preds.items():
        th = tune_thresholds(yv, pv)
        hard_all.append(pt >= th[None, :])
        prob_all.append(pt)
        y_all.append(yt)
    H, P, Y = map(np.concatenate, (hard_all, prob_all, y_all))
    for c, cname in enumerate(classes):
        col = Y[:, c] > 0
        npos = int(col.sum())
        tp = int((H[:, c] & col).sum())
        fp = int((H[:, c] & ~col).sum())
        rows.append({"family": family, "seed": seed, "class": cname,
                     "n_pos": npos,
                     "auroc": roc_auc_score(col, P[:, c]) if 0 < npos < len(col) else np.nan,
                     "sens": tp / npos if npos else np.nan,
                     "ppv": tp / (tp + fp) if (tp + fp) else np.nan})
    return rows


# ------------------------------------------------------------ significance

def site_aurocs(preds):
    return {s: macro_auroc(p[3], p[2]) for s, p in preds.items()}


def paired_bootstrap(pa, pb, n_boot=1000, seed=0):
    """CI of the difference in mean and worst site macro-AUROC between two
    methods evaluated on the same test records (resampled per site)."""
    rng = np.random.RandomState(seed)
    d_mean, d_worst = [], []
    sites = sorted(set(pa) & set(pb))
    for _ in range(n_boot):
        a, b = [], []
        for s in sites:
            yt = pa[s][3]
            idx = rng.randint(0, len(yt), len(yt))
            a.append(macro_auroc(yt[idx], pa[s][2][idx]))
            b.append(macro_auroc(yt[idx], pb[s][2][idx]))
        d_mean.append(np.nanmean(a) - np.nanmean(b))
        d_worst.append(np.nanmin(a) - np.nanmin(b))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"d_mean_ci": q(d_mean), "d_worst_ci": q(d_worst)}


# ------------------------------------------------------------ twin

def _js(p, q, eps=1e-7):
    p, q = np.clip(p, eps, 1 - eps), np.clip(q, eps, 1 - eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    return np.mean(0.5 * kl(p, m) + 0.5 * kl(q, m), axis=-1)


def _delong(y, s1, s2):
    """Paired DeLong test for two correlated AUROCs (fast version)."""
    from scipy.stats import norm
    y = np.asarray(y).astype(bool)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    m, n = len(pos), len(neg)

    def comps(s):
        sp, sn = s[pos], s[neg]
        v10 = np.array([((sp_i > sn).sum() + 0.5 * (sp_i == sn).sum()) / n for sp_i in sp])
        v01 = np.array([((sp > sn_j).sum() + 0.5 * (sp == sn_j).sum()) / m for sn_j in sn])
        return v10.mean(), v10, v01

    a1, v10a, v01a = comps(s1)
    a2, v10b, v01b = comps(s2)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + \
          (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return float(a1), float(a2), float(2 * norm.sf(abs(z)))


def macro_auroc_w(y, p, w):
    """Macro AUROC with sample weights (classes absent/saturated skipped)."""
    aucs = []
    for c in range(y.shape[1]):
        col = y[:, c]
        pos, neg = (w * (col > 0)).sum(), (w * (col == 0)).sum()
        if pos > 0 and neg > 0:
            aucs.append(roc_auc_score(col, p[:, c], sample_weight=w))
    return float(np.mean(aucs)) if aucs else np.nan


def _bce_rows(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(axis=1)


def twin_analysis(arr_path, n_boot=1000, seed=0):
    """Alert baselines, gain by number of prior records and three guard
    variants, all from the logged per-record outputs of one twin run:
      unguarded   always the adapted twin;
      in_sample   the run's guard (twin vs global BCE on the records the
                  twin was trained on);
      prequential twin used at record t only if its loss on records
                  1..t-1, each predicted BEFORE the twin learned from it,
                  is lower than the global model's (out of sample)."""
    z = np.load(arr_path)
    if "pid_all" not in z.files:
        return None
    pid, t = z["pid_all"], z["t_all"]
    cold, twin, warm, y, guard = (z["cold_all"], z["twin_all"], z["warm_all"],
                                  z["y_all"], z["guard_all"])
    y = y.astype(float)
    # prequential guard, per patient
    preq = cold.copy()
    lc, lt = _bce_rows(cold, y), _bce_rows(twin, y)
    cum_c = cum_t = 0.0
    for i in range(len(pid)):
        if t[i] == 0:
            cum_c = cum_t = 0.0
            continue
        if t[i] >= 2 and cum_t < cum_c:
            preq[i] = twin[i]
        cum_c += lc[i]
        cum_t += lt[i]
    rows = []
    for i in range(1, len(pid)):
        if pid[i] != pid[i - 1] or t[i] == 0:
            continue
        rows.append({
            "i": i, "pid": int(pid[i]), "t": int(t[i]),
            "changed": int((y[i] != y[i - 1]).any()),
            "max_delta_cold": abs(cold[i].max() - cold[i - 1].max()),
            "max_delta_twin": abs(twin[i].max() - twin[i - 1].max()),
            "js_cold": float(_js(cold[i], cold[i - 1])),
            "guard_on": bool(guard[i]),
        })
    df = pd.DataFrame(rows)
    df["ours"] = z["alert"][:len(df)] if len(z["alert"]) == len(df) else np.nan
    out = {"n_pairs": int(len(df)), "n_changed": int(df.changed.sum()),
           "n_patients": int(df.pid.nunique()),
           "guard_accept_rate": float(df.guard_on.mean())}
    rng = np.random.RandomState(seed)
    upid, pidx = np.unique(df.pid.values, return_inverse=True)

    def boot_weights():
        cnt = rng.multinomial(len(upid), np.full(len(upid), 1.0 / len(upid)))
        return cnt[pidx].astype(float)

    W = [boot_weights() for _ in range(n_boot)]
    scores = ["ours", "max_delta_cold", "max_delta_twin", "js_cold"]
    for sc in scores:
        if df[sc].isna().any():
            continue
        yv, sv = df.changed.values, df[sc].values
        out[f"auroc_{sc}"] = float(roc_auc_score(yv, sv))
        b = [roc_auc_score(yv, sv, sample_weight=w) for w in W
             if 0 < (w * yv).sum() < w.sum()]
        out[f"ci_{sc}"] = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    if not df["ours"].isna().any():
        for sc in scores[1:]:
            _, _, pv = _delong(df.changed.values, df["ours"].values, df[sc].values)
            out[f"delong_p_ours_vs_{sc}"] = pv
    # gains (macro AUROC on records t >= 1) for the three guard variants
    sel = np.array(df.i.values)
    yy, cc = y[sel], cold[sel]
    variants = {"unguarded": twin[sel], "in_sample": warm[sel], "prequential": preq[sel]}
    base = macro_auroc(yy, cc)
    out["cold_auroc"] = base
    for name, pv_ in variants.items():
        g = macro_auroc(yy, pv_) - base
        bs = [macro_auroc_w(yy, pv_, w) - macro_auroc_w(yy, cc, w) for w in W[:300]]
        out[f"gain_{name}"] = float(g)
        out[f"gain_{name}_ci"] = (float(np.nanpercentile(bs, 2.5)),
                                  float(np.nanpercentile(bs, 97.5)))
    out["prequential_accept_rate"] = float(
        np.mean([not np.allclose(preq[i], cold[i]) for i in sel]))
    # gain by number of prior records
    per_t = {}
    for k in sorted(set(df.t.tolist())):
        m = (df.t.values == k) if k < 3 else (df.t.values >= 3)
        if k > 3 or m.sum() < 10:
            continue
        ii = sel[m]
        per_t[f"t{k}" if k < 3 else "t>=3"] = {
            "n": int(m.sum()),
            **{f"gain_{n}": float(macro_auroc(y[ii], v[ii]) - macro_auroc(y[ii], cold[ii]))
               for n, v in (("unguarded", twin), ("in_sample", warm), ("prequential", preq))},
            "in_sample_accept": float(df.guard_on.values[m].mean())}
    out["per_t"] = per_t
    return out


# ------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--eval-repo", default="external/evaluation-2021")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="few splits/boots")
    args = ap.parse_args()
    if args.quick:
        args.n_splits, args.n_boot = 5, 20
    out = args.out_dir or os.path.join(args.results_dir, "revision")
    os.makedirs(out, exist_ok=True)
    classes = class_names(args.eval_repo)
    fam = runs_by_family(os.path.join(args.results_dir, "preds"))
    numbers = {}
    print("families:", {k: sorted(v) for k, v in fam.items()})
    PER = "fed_fedper" if "fed_fedper" in fam else "fed_fedper_ref"
    AVG = "fed_fedavg" if "fed_fedavg" in fam else "fed_fedavg_ref"
    numbers["primary_families"] = [PER, AVG]

    # ---- per-class + key classes
    pc, pk = [], []
    TAB3 = ("fed_fedper", "fed_fedavg", "fed_fedrep", "fed_fedbabu",
            "centralized", "ft_fedavg", "ft_central", "fed_local",
            "fed_fedbn", "fed_ditto", "fed_fedprox", "fed_fedavgm")
    for f in TAB3 + tuple(t + "_ref" for t in TAB3):
        for seed, p in fam.get(f, {}).items():
            pr = load_preds(p)
            pc += per_class_rows(f, seed, pr, classes)
            pk += pooled_key_rows(f, seed, pr, classes)
    if pc:
        pc, pk = pd.DataFrame(pc), pd.DataFrame(pk)
        pc.to_csv(f"{out}/per_class_site.csv", index=False)
        pk.to_csv(f"{out}/per_class_pooled.csv", index=False)
        agg = (pc.groupby(["family", "site", "class"])
               [["auroc", "sens", "ppv", "f1", "n_pos"]].mean().reset_index())
        agg.to_csv(f"{out}/per_class_site_mean.csv", index=False)
        key = (pk[pk["class"].isin(KEY_CLASSES)]
               .groupby(["family", "class"])
               .agg(n_pos=("n_pos", "mean"), auroc=("auroc", "mean"),
                    auroc_sd=("auroc", "std"), sens=("sens", "mean"),
                    sens_sd=("sens", "std"), ppv=("ppv", "mean"),
                    ppv_sd=("ppv", "std")).reset_index())
        key.to_csv(f"{out}/key_classes.csv", index=False)
        # which classes FedPer improves over FedAvg (per site, seed-mean)
        if {PER, AVG} <= set(agg.family):
            a = agg[agg.family == PER].set_index(["site", "class"]).auroc
            b = agg[agg.family == AVG].set_index(["site", "class"]).auroc
            d = (a - b).dropna()
            d.reset_index().rename(columns={"auroc": "delta"}).to_csv(
                f"{out}/per_class_delta_fedper_fedavg.csv", index=False)
            numbers["per_class_frac_improved"] = float((d > 0).mean())
            numbers["per_class_median_delta"] = float(d.median())

    # ---- paired bootstrap + Wilcoxon vs FedAvg (and B variants vs FedPer)
    from scipy.stats import wilcoxon
    B_TAGS = ("_ref", "_depth", "_eqw", "_fedla", "_part50", "_drop30", "_leads",
              "_twinfed")
    BOOT = {PER, "fed_fedrep", "fed_fedbabu", "ft_fedavg", "ft_central", "centralized"}
    BOOT |= {b + "_ref" for b in BOOT if not b.endswith("_ref")}
    # Table III rows of the same-machine reference batch, also compared with FedPer
    VS_PER = {t + "_ref" for t in TAB3} - {"fed_fedper_ref", "fed_fedavg_ref"}

    def compare(f, ref, boot):
        seeds = sorted(set(fam[f]) & set(fam[ref]))
        if not seeds:
            return None
        pairs, boots, site_d = [], [], defaultdict(list)
        for s_ in seeds:
            pa, pb = load_preds(fam[f][s_]), load_preds(fam[ref][s_])
            aa, ab = site_aurocs(pa), site_aurocs(pb)
            for site in aa:
                pairs.append((aa[site], ab[site]))
                site_d[site].append(aa[site] - ab[site])
            if boot and s_ == seeds[0] and args.n_boot:
                boots.append(paired_bootstrap(pa, pb, args.n_boot, s_))
        a, b = np.array(pairs).T
        res = {"ref": ref, "n_pairs": len(a), "mean_diff": float((a - b).mean())}
        if len(a) >= 6 and not np.allclose(a, b):
            res["wilcoxon_p_site_seed"] = float(wilcoxon(a, b).pvalue)
        sd = np.array([np.mean(v) for v in site_d.values()])
        if len(sd) >= 6 and not np.allclose(sd, 0):
            res["wilcoxon_p_site_level"] = float(wilcoxon(sd).pvalue)
        res["site_diff"] = {k: float(np.mean(v)) for k, v in site_d.items()}
        if boots:
            res["bootstrap_seed0"] = boots[0]
        return res

    sig = {}
    for f in sorted(fam):
        if not (f.startswith("fed_") or f.startswith("ft_") or f.startswith("centralized")):
            continue
        tagged = any(tg in f for tg in B_TAGS)
        ref = "fed_fedavg_ref" if tagged else "fed_fedavg"
        if ref not in fam or f == ref:
            continue
        r = compare(f, ref, f in BOOT)
        if r:
            sig[f] = r
        if tagged and (f.startswith("fed_fedper_") or f in VS_PER) \
                and f != "fed_fedper_ref" and "fed_fedper_ref" in fam:
            r2 = compare(f, "fed_fedper_ref", f in VS_PER)
            if r2:
                sig[f + "__vs_fedper_ref"] = r2
    json.dump(sig, open(f"{out}/significance.json", "w"), indent=2)

    # ---- conformal: repeated re-splits on the deployed models
    conf_fams = [f for f in (PER, AVG, "fed_fedbn",
                             "fed_fedper_part50", "fed_fedper_drop30",
                             "fed_fedper_leads2", "fed_fedper_leads1",
                             "fed_fedrep", "fed_fedrep_ref") if f in fam]
    conf_rows = []
    for f in conf_fams:
        for seed, p in sorted(fam[f].items()):
            rows = crc.multi_split(load_preds(p),
                                   alphas=(0.05, 0.10, 0.15, 0.20),
                                   cal_alphas=(0.03, 0.04, 0.05, 0.06, 0.07,
                                               0.08, 0.09, 0.10),
                                   n_splits=args.n_splits, seed=seed,
                                   ucb_delta=0.1)
            for r in rows:
                r.update(family=f, seed=seed)
            conf_rows += rows
    if conf_rows:
        cf = pd.DataFrame(conf_rows)
        cf["violation"] = cf.fnr > cf.alpha
        cf.to_csv(f"{out}/conformal_splits.csv.gz", index=False)
        summ = (cf.groupby(["family", "alpha", "mode", "cal_alpha", "site"],
                           dropna=False)
                .agg(fnr_mean=("fnr", "mean"), fnr_sd=("fnr", "std"),
                     fnr_p05=("fnr", lambda v: np.percentile(v, 5)),
                     fnr_p95=("fnr", lambda v: np.percentile(v, 95)),
                     viol_freq=("violation", "mean"),
                     size_mean=("size", "mean"), lam_mean=("lambda", "mean"),
                     lam_sd=("lambda", "std"), n_cal=("n_cal", "mean"))
                .reset_index())
        summ.to_csv(f"{out}/conformal_summary.csv", index=False)

    # calibration-size curves on large hospitals (FedPer, seed 0)
    if PER in fam:
        pr = load_preds(fam[PER][min(fam[PER])])
        cs = []
        for site in ("Ningbo", "PTB-XL", "Chapman", "Georgia"):
            if site not in pr:
                continue
            pv, yv, pt, yt = pr[site]
            rows = crc.cal_size_curve(np.concatenate([pv, pt]),
                                      np.concatenate([yv, yt]),
                                      sizes=(50, 100, 200, 500, 1000, 2000),
                                      n_rep=max(20, args.n_splits),
                                      alpha=0.1, cal_alphas=(0.05,),
                                      ucb_delta=0.1, seed=0)
            for r in rows:
                r["site"] = site
            cs += rows
        if cs:
            cs = pd.DataFrame(cs)
            cs["violation"] = cs.fnr > 0.1
            (cs.groupby(["site", "n_cal", "mode", "target"])
             .agg(fnr_mean=("fnr", "mean"), viol_freq=("violation", "mean"),
                  size_mean=("size", "mean")).reset_index()
             .to_csv(f"{out}/cal_size_curve.csv", index=False))

        # class-wise and weighted risk control (FedPer)
        crit = [classes.index(c) for c in CRITICAL if c in classes]
        cw = []
        for seed, p in sorted(fam[PER].items()):
            rows = crc.classwise_multi_split(load_preds(p), crit, alpha=0.1,
                                             alpha_c=0.05, cal_alpha=0.05,
                                             n_splits=args.n_splits, seed=seed)
            for r in rows:
                r["seed"] = seed
            cw += rows
        cw = pd.DataFrame(cw)
        cw.to_csv(f"{out}/classwise_splits.csv.gz", index=False)
        miss_cols = [f"miss_{c}" for c in crit]
        g = cw.groupby(["rule"])[["fnr", "size"] + miss_cols].mean()
        g.columns = ["fnr", "size"] + [classes[c] for c in crit]
        g.to_csv(f"{out}/classwise_summary.csv")
        w = np.ones(len(classes))
        w[crit] = 3.0
        wr = []
        for seed, p in sorted(fam[PER].items()):
            rows = crc.multi_split(load_preds(p), alphas=(0.1,), cal_alphas=(0.05,),
                                   n_splits=args.n_splits, seed=seed,
                                   ucb_delta=0.1, class_weights=w)
            for r in rows:
                r["seed"] = seed
            wr += rows
        wr = pd.DataFrame(wr)
        wr["violation"] = wr.fnr > wr.alpha
        (wr.groupby(["mode", "site"]).agg(wfnr_mean=("fnr", "mean"),
                                          viol_freq=("violation", "mean"),
                                          size_mean=("size", "mean"))
         .reset_index().to_csv(f"{out}/weighted_fnr_summary.csv", index=False))

    # ---- twin
    tw = {}
    for p in sorted(glob.glob(os.path.join(args.results_dir, "twin*_arrays.npz"))):
        r = twin_analysis(p, n_boot=args.n_boot)
        if r is not None:
            tw[os.path.basename(p)] = r
    if tw:
        json.dump(tw, open(f"{out}/twin_analysis.json", "w"), indent=2)

    # ---- JSON-based tables: snapshots, comm, timing, sensitivity rows
    res_rows, snap_rows, hist_rows = [], [], []
    for p in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        name = os.path.basename(p)[:-5]
        m = re.match(r"(.+)_seed(\d+)$", name)
        if not m:
            continue
        r = json.load(open(p))
        if "MEAN" not in r:
            continue
        fam_name, seed = m.group(1), int(m.group(2))
        row = {"family": fam_name, "seed": seed,
               "mean_auroc": r["MEAN"]["auroc"], "worst_auroc": r["WORST"]["auroc"],
               "mean_f1": r["MEAN"]["f1"], "worst_f1": r["WORST"]["f1"]}
        for s in SITES:
            if s in r:
                row[f"auroc_{s}"] = r[s]["auroc"]
                row[f"f1_{s}"] = r[s]["f1"]
        c = r.get("comm", {})
        row.update({"up_mb_round": c.get("per_client_round_upload_mb"),
                    "down_mb_round": c.get("per_client_round_download_mb"),
                    "mb_round_fp16": c.get("per_client_round_mb_fp16"),
                    "cum_up_mb": c.get("cum_upload_mb_per_client"),
                    "cum_down_mb": c.get("cum_upload_mb_per_client"),
                    "cum_total_federation_mb": (c.get("cum_upload_mb_total") or 0)
                    + (c.get("cum_download_mb_total") or 0),
                    "rounds": c.get("rounds")})
        t = r.get("timing", {})
        for s in SITES:
            row[f"time_{s}"] = (t.get("local_train_s_median") or {}).get(s)
            row[f"mem_{s}"] = (t.get("peak_mem_mib") or {}).get(s)
        row["agg_s"] = t.get("aggregation_s_mean")
        if "dp" in r:
            row["dp_eps_achieved_max"] = max(r["dp"]["achieved_epsilon"].values())
        res_rows.append(row)
        for rr, sm in (r.get("snapshots") or {}).items():
            snap_rows.append({"family": fam_name, "seed": seed, "round": int(rr),
                              "mean_auroc": sm["MEAN"]["auroc"],
                              "worst_auroc": sm["WORST"]["auroc"]})
        for h in r.get("history") or []:
            vals = [h[s] for s in SITES if s in h]
            hist_rows.append({"family": fam_name, "seed": seed,
                              "round": h["round"],
                              "val_mean": float(np.mean(vals)) if vals else np.nan,
                              "cum_up_mb": h.get("cum_upload_mb"),
                              "cum_down_mb": h.get("cum_download_mb")})
    if res_rows:
        rd = pd.DataFrame(res_rows)
        rd.to_csv(f"{out}/runs.csv", index=False)
        summ = rd.drop(columns=["seed"]).groupby("family").agg(["mean", "std"])
        summ.columns = ["_".join(c) for c in summ.columns]
        summ.to_csv(f"{out}/runs_summary.csv")
    if snap_rows:
        sd = pd.DataFrame(snap_rows)
        (sd.groupby(["family", "round"]).agg(
            mean=("mean_auroc", "mean"), mean_sd=("mean_auroc", "std"),
            worst=("worst_auroc", "mean"), worst_sd=("worst_auroc", "std"))
         .reset_index().to_csv(f"{out}/round_budget.csv", index=False))
    if hist_rows:
        hd = pd.DataFrame(hist_rows)
        hd.to_csv(f"{out}/val_history.csv", index=False)
        # patience-based early stopping on the mean validation AUROC
        stops = []
        for (f, s), g in hd.groupby(["family", "seed"]):
            g = g.sort_values("round")
            if len(g) < 3:
                continue
            best, best_r, stop = -1, None, None
            for _, e in g.iterrows():
                if e.val_mean > best + 1e-3:
                    best, best_r = e.val_mean, e["round"]
                elif e["round"] - best_r >= 20:
                    stop = e["round"]
                    break
            stops.append({"family": f, "seed": s, "best_round": best_r,
                          "stop_round": stop if stop is not None else g["round"].max(),
                          "stopped_early": stop is not None})
        pd.DataFrame(stops).to_csv(f"{out}/early_stopping.csv", index=False)

    # ---- PTB-XL patient-disjoint check (main split, seed of the cache)
    try:
        from fedcardiotwin.data.dataset import ClientData, ptbxl_client_meta
        meta = ptbxl_client_meta(args.cache_dir)
        cd = ClientData(args.cache_dir, "PTB-XL")
        train_pat = set(meta.patient_id.values[cd.train_idx])
        test_pat = meta.patient_id.values[cd.test_idx]
        disjoint = np.array([p not in train_pat for p in test_pat])
        chk = {"test_records": int(len(test_pat)),
               "disjoint_records": int(disjoint.sum())}
        for f in (PER, AVG, "centralized", "centralized_ref"):
            vals = []
            for seed, p in sorted(fam.get(f, {}).items()):
                pv, yv, pt, yt = load_preds(p)["PTB-XL"]
                vals.append((macro_auroc(yt, pt),
                             macro_auroc(yt[disjoint], pt[disjoint])))
            if vals:
                v = np.array(vals)
                chk[f] = {"all": float(v[:, 0].mean()),
                          "patient_disjoint": float(v[:, 1].mean())}
        json.dump(chk, open(f"{out}/ptbxl_patient_disjoint.json", "w"), indent=2)
    except Exception as e:  # cache not available
        print("patient-disjoint check skipped:", e)

    json.dump(numbers, open(f"{out}/numbers.json", "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
