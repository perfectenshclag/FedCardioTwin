#!/usr/bin/env python3
"""Figures of the revised manuscript, drawn from the result files.

  fig_rounds.pdf     120 round runs: (a) mean validation AUROC over rounds,
                     (b) test AUROC at the round snapshots vs cumulative
                     traffic (up+down); skipped until those runs exist
  fig_loho.pdf       held-out hospital AUROC + onboarding curve
  fig_conformal.pdf  worst-hospital violation frequency and set size vs the
                     calibration level (repeated re-splits), UCB rule marked
  fig_calsize.pdf    (supplement) violation frequency vs calibration size

  python scripts/make_revision_figures.py --results-dir results \\
      --out-dir figures
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
SITES = ["CPSC", "CPSC-Extra", "Georgia", "Chapman", "Ningbo", "PTB-XL"]
METHODS = [("fed_fedavg", "FedAvg"), ("fed_fedprox", "FedProx"),
           ("fed_fedavgm", "FedAvgM"), ("fed_fedbn", "FedBN"),
           ("fed_ditto", "Ditto"), ("fed_fedrep", "FedRep"),
           ("fed_fedbabu", "FedBABU"), ("fed_fedper", "FedPer")]


R120 = [("fed_fedavg_r120", "FedAvg"), ("fed_fedrep_r120", "FedRep"),
        ("fed_fedper_r120", "FedPer")]


SIXTY = [  # (family, label, colour, line style)
    ("fed_fedavg_ref", "FedAvg, all hospitals", "#4c72b0", "-"),
    ("fed_fedper_ref", "FedPer, all hospitals", "#dd8452", "-"),
    ("fed_fedavg_part50", "FedAvg, 3 of 6 per round", "#4c72b0", "--"),
    ("fed_fedper_part50", "FedPer, 3 of 6 per round", "#dd8452", "--"),
    ("fed_fedavg_drop30", "FedAvg, 30% dropout", "#4c72b0", ":"),
    ("fed_fedper_drop30", "FedPer, 30% dropout", "#dd8452", ":"),
]


def fig_rounds_60(h, out):
    """Round budget and two way traffic from the 60 round runs (3 seeds):
    (a) mean validation AUROC per logged round, (b) the same against the
    average cumulative traffic per hospital (upload plus download)."""
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.85))
    for f, lab, col, ls in SIXTY:
        d = h[h.family == f]
        if not len(d):
            continue
        g = d.groupby("round").agg(v=("val_mean", "mean"), s=("val_mean", "std"),
                                   up=("cum_up_mb", "mean"), down=("cum_down_mb", "mean"))
        lw = 1.6 if "all" in lab else 1.1
        ax[0].plot(g.index, g.v, color=col, ls=ls, lw=lw, marker="o", ms=2.5, label=lab)
        ax[0].fill_between(g.index, g.v - g.s, g.v + g.s, color=col, alpha=.10, lw=0)
        traffic = (g.up + g.down) / len(SITES)  # federation total -> per hospital
        ax[1].plot(traffic, g.v, color=col, ls=ls, lw=lw, marker="o", ms=2.5)
    ax[0].set(xlabel="Communication round", ylabel="Mean validation AUROC", title="(a)",
              xticks=range(10, 61, 10))
    ax[1].set(xlabel="Cumulative traffic per hospital (MB, up+down)",
              ylabel="Mean validation AUROC", title="(b)")
    ax[0].legend(frameon=False, ncol=1, fontsize=6, loc="lower right")
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(f"{out}/fig_rounds.pdf")
    plt.close(fig)


def fig_rounds(rev, out):
    """Round budget (Reviewer 2, comment 4): from the 120 round runs when they
    exist, otherwise from the 60 round runs."""
    vh, rb = f"{rev}/val_history.csv", f"{rev}/round_budget.csv"
    if not (os.path.exists(vh) and os.path.exists(rb)):
        return
    h, r = pd.read_csv(vh), pd.read_csv(rb)
    if not any((h.family == f).any() for f, _ in R120):
        fig_rounds_60(h, out)
        return
    runs = pd.read_csv(f"{rev}/runs.csv") if os.path.exists(f"{rev}/runs.csv") else None
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for f, lab in R120:
        d = h[h.family == f]
        if len(d):
            g = d.groupby("round").val_mean.mean()
            ax[0].plot(g.index, g.values, label=lab, lw=1.2 if lab != "FedPer" else 2)
        d = r[r.family == f].sort_values("round")
        if len(d) and runs is not None:
            per_round = runs[runs.family == f][["up_mb_round", "down_mb_round"]].mean().sum()
            ax[1].plot(d["round"] * per_round, d["mean"], marker="o", ms=3,
                       lw=1.2 if lab != "FedPer" else 2, label=lab)
    ax[0].axvline(60, color="gray", ls=":", lw=.8)
    ax[0].set(xlabel="Communication round", ylabel="Mean validation AUROC", title="(a)")
    ax[1].set(xlabel="Cumulative traffic per hospital (MB, up+down)",
              ylabel="Mean test AUROC", title="(b)")
    ax[0].legend(frameon=False)
    fig.savefig(f"{out}/fig_rounds.pdf")
    plt.close(fig)


T3_FIG = [("fed_local", "Local"), ("fed_fedavg", "FedAvg"), ("fed_fedprox", "FedProx"),
          ("fed_fedavgm", "FedAvgM"), ("fed_fedbn", "FedBN"), ("fed_ditto", "Ditto"),
          ("fed_fedrep", "FedRep"), ("ft_fedavg", "FedAvg+FT"), ("fed_fedper", "FedPer"),
          ("centralized", "Central."), ("ft_central", "Central.+FT")]


def _t3(results):
    """Per seed site AUROC of the Table III runs (results/original)."""
    out = []
    for f, lab in T3_FIG:
        R = [json.load(open(p)) for p in sorted(glob.glob(f"{results}/original/{f}_seed[0-9].json"))]
        if R:
            a = np.array([[r[s]["auroc"] for s in SITES] for r in R])
            out.append((lab, a))
    return out


def fig_main_auroc(results, out):
    """Per hospital macro AUROC of every Table III strategy (mean +- sd over seeds)."""
    t = _t3(results)
    if not t:
        return
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 3.1))
    k, w = len(t), 0.8 / len(t)
    x = np.arange(len(SITES))
    for i, (lab, a) in enumerate(t):
        ax.bar(x + (i - (k - 1) / 2) * w, a.mean(0), w, yerr=a.std(0, ddof=1) if len(a) > 1 else None,
               capsize=1.2, error_kw={"lw": .6}, label=lab, color=cmap(i * 2 % 20 + (i * 2 >= 20)),
               edgecolor="black" if lab == "FedPer" else "none", lw=.6)
    ax.set_xticks(x, [s.replace("-", " ") for s in SITES])
    ax.set(ylim=(0.70, 1.0), ylabel="Macro AUROC")
    ax.grid(axis="y", alpha=.3)
    ax.legend(ncol=6, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(f"{out}/fig_main_auroc.pdf")
    plt.close(fig)


def fig_mean_worst(results, out):
    """Mean and worst hospital AUROC of every Table III strategy."""
    t = _t3(results)
    if not t:
        return
    labs = [l for l, _ in t]
    mean = np.array([a.mean(1).mean() for _, a in t]); msd = np.array([a.mean(1).std(ddof=1) for _, a in t])
    worst = np.array([a.min(1).mean() for _, a in t]); wsd = np.array([a.min(1).std(ddof=1) for _, a in t])
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.bar(x - .2, mean, .4, yerr=msd, capsize=1.2, error_kw={"lw": .6}, color="#4c72b0", label="Mean")
    ax.bar(x + .2, worst, .4, yerr=wsd, capsize=1.2, error_kw={"lw": .6}, color="#c44e52", label="Worst hospital")
    ax.set_xticks(x, labs, rotation=45, ha="right", fontsize=6.5)
    ax.set(ylim=(0.70, 0.95), ylabel="Macro AUROC")
    ax.grid(axis="y", alpha=.3)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    fig.savefig(f"{out}/fig_mean_worst.pdf")
    plt.close(fig)



def fig_benchmark(results, out):
    """(a) per hospital macro AUROC of every Table III strategy, (b) mean and
    worst hospital AUROC, in one two column figure."""
    t = _t3(results)
    if not t:
        return
    import matplotlib.gridspec as gs
    cmap = plt.get_cmap("tab20")
    cols = [cmap(i * 2 % 20 + (i * 2 >= 20)) for i in range(len(t))]
    fig = plt.figure(figsize=(10, 2.45))
    g = gs.GridSpec(1, 2, width_ratios=[2.6, 1], wspace=0.18)
    ax = fig.add_subplot(g[0])
    k, w = len(t), 0.8 / len(t)
    x = np.arange(len(SITES))
    for i, (lab, a) in enumerate(t):
        ax.bar(x + (i - (k - 1) / 2) * w, a.mean(0), w, yerr=a.std(0, ddof=1) if len(a) > 1 else None,
               capsize=1.0, error_kw={"lw": .5}, label=lab, color=cols[i],
               edgecolor="black" if lab == "FedPer" else "none", lw=.6)
    ax.set_xticks(x, [s.replace("-", " ") for s in SITES])
    ax.set(ylim=(0.70, 1.065), ylabel="Macro AUROC", title="(a) per hospital", yticks=np.arange(0.70, 1.001, 0.05))
    ax.grid(axis="y", alpha=.3)
    ax.legend(ncol=11, frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.0, columnspacing=0.6, handletextpad=0.3)
    ax2 = fig.add_subplot(g[1])
    labs = [l for l, _ in t]
    xx = np.arange(len(labs))
    mean = [a.mean(1).mean() for _, a in t]; msd = [a.mean(1).std(ddof=1) for _, a in t]
    worst = [a.min(1).mean() for _, a in t]; wsd = [a.min(1).std(ddof=1) for _, a in t]
    ax2.bar(xx - .2, mean, .4, yerr=msd, capsize=1.0, error_kw={"lw": .5}, color="#4c72b0", label="Mean")
    ax2.bar(xx + .2, worst, .4, yerr=wsd, capsize=1.0, error_kw={"lw": .5}, color="#c44e52", label="Worst hospital")
    ax2.set_xticks(xx, labs, rotation=50, ha="right", fontsize=6)
    ax2.set(ylim=(0.70, 1.0), title="(b) mean and worst hospital")
    ax2.grid(axis="y", alpha=.3)
    ax2.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper left")
    fig.savefig(f"{out}/fig_benchmark.pdf")
    plt.close(fig)


def fig_loho(results, out):
    files = [p for p in glob.glob(f"{results}/loho_*_seed0.json")
             if not p.endswith("loho_seed0.json")]
    if not files:
        return
    zs, ob = {}, []
    for p in files:
        d = json.load(open(p))
        site = [k for k in d if k in SITES]
        if not site:
            continue
        zs[site[0]] = d[site[0]]["auroc"]
        for o in d.get("onboarding", []):
            ob.append({"site": site[0], **o})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.85), gridspec_kw={"width_ratios": [1, 1.35]})
    order = [s for s in SITES if s in zs]
    ax[0].bar(range(len(order)), [zs[s] for s in order], color="#4c72b0")
    ax[0].set_xticks(range(len(order)), [s.replace("-", " ") for s in order],
                     rotation=30, ha="right")
    ax[0].set(ylim=(0.6, 1.0), ylabel="Macro AUROC (unseen hospital)", title="(a) zero shot")
    if ob:
        ob = pd.DataFrame(ob)
        for s in order:
            d = ob[ob.site == s].groupby("n").auroc.agg(["mean", "std"]).reset_index()
            if not len(d):
                continue
            x = np.r_[0, d["n"].values]
            y = np.r_[zs[s], d["mean"].values]
            ax[1].plot(x[1:], y[1:], marker="o", ms=3, label=s)
            ax[1].axhline(zs[s], lw=.5, ls=":", color=ax[1].lines[-1].get_color())
        ax[1].set_xscale("log")
        ax[1].set(xlabel="Labelled records at the new hospital ($n$)",
                  ylabel="Macro AUROC", title="(b) onboarding (private head)")
        ax[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(f"{out}/fig_loho.pdf")
    plt.close(fig)


def fig_conformal(rev, out):
    cs = f"{rev}/conformal_summary.csv"
    if not os.path.exists(cs):
        return
    c = pd.read_csv(cs)
    per = "fed_fedper" if (c.family == "fed_fedper").any() else "fed_fedper_ref"
    c = c[c.family == per]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.85))
    for alpha, mk in ((0.05, "s"), (0.10, "o"), (0.15, "^"), (0.20, "v")):
        d = c[(c["mode"] == "personalized") & np.isclose(c.alpha, alpha)]
        if not len(d):
            continue
        g = d.groupby("cal_alpha").agg(v=("viol_freq", "max"), z=("size_mean", "mean"))
        g = g[g.index <= alpha + 1e-9]
        l, = ax[0].plot(g.index, 100 * g.v, marker=mk, ms=3, label=f"$\\alpha$={alpha:.2f}")
        ax[1].plot(g.index, g.z, marker=mk, ms=3, color=l.get_color())
        u = c[(c["mode"] == "ucb") & np.isclose(c.alpha, alpha)]
        if len(u):
            ax[0].axhline(100 * u.viol_freq.max(), color=l.get_color(), ls=":", lw=.8)
            ax[1].axhline(u.size_mean.mean(), color=l.get_color(), ls=":", lw=.8)
    ax[0].set(xlabel="Calibration level $\\alpha_{cal}$",
              ylabel="Re-splits with FNR > $\\alpha$ (%, worst hospital)", title="(a)")
    ax[1].set(xlabel="Calibration level $\\alpha_{cal}$", ylabel="Mean set size", title="(b)")
    ax[0].legend(frameon=False)
    fig.savefig(f"{out}/fig_conformal.pdf")
    plt.close(fig)


def fig_calsize(rev, out):
    p = f"{rev}/cal_size_curve.csv"
    if not os.path.exists(p):
        return
    c = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for (mode, tgt), d in c.groupby(["mode", "target"]):
        g = d.groupby("n_cal").agg(v=("viol_freq", "mean"), z=("size_mean", "mean"))
        lab = f"{mode} at {tgt:.2f}" if mode == "local" else "size adaptive (UCB)"
        ax[0].plot(g.index, 100 * g.v, marker="o", ms=3, label=lab)
        ax[1].plot(g.index, g.z, marker="o", ms=3, label=lab)
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("Calibration records $n$")
    ax[0].set(ylabel="Draws with FNR > 0.10 (%)")
    ax[1].set(ylabel="Mean set size")
    ax[0].legend(frameon=False)
    fig.savefig(f"{out}/fig_calsize.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    rev = os.path.join(args.results_dir, "revision")
    os.makedirs(args.out_dir, exist_ok=True)
    for fn in (lambda: fig_benchmark(args.results_dir, args.out_dir),
               lambda: fig_main_auroc(args.results_dir, args.out_dir),
               lambda: fig_mean_worst(args.results_dir, args.out_dir),
               lambda: fig_rounds(rev, args.out_dir),
               lambda: fig_loho(args.results_dir, args.out_dir),
               lambda: fig_conformal(rev, args.out_dir),
               lambda: fig_calsize(rev, args.out_dir)):
        try:
            fn()
        except Exception as e:  # one missing input must not block the rest
            print("figure skipped:", e)
    print("figures ->", args.out_dir)


if __name__ == "__main__":
    main()
