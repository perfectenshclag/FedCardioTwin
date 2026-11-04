"""Federated conformal risk control for multilabel ECG prediction sets.

Per client, conformal risk control (Angelopoulos et al.) picks the smallest
threshold lambda such that the calibrated false-negative rate is <= alpha.
The federated variant averages client risk curves (weighted by calibration
size) on a shared lambda grid, yielding one global lambda with group-level
guarantees — the practical recipe from federated conformal prediction work.

Prediction set for a record = {classes with sigmoid prob >= 1 - lambda}.
Outputs per client: empirical FNR (coverage), mean set size.

Revision additions:
  * vectorized per-record loss matrices (all lambdas at once), which make
    repeated random calibration/test re-splits cheap (multi_split);
  * a size-adaptive upper-confidence-bound rule (RCPS-style, Hoeffding-
    Bentkus bound) that controls the FNR with probability >= 1 - delta
    over the calibration draw instead of only in expectation;
  * clinically weighted FNR (any class weights; still a valid CRC loss);
  * class-conditional CRC for critical diagnoses (classwise_multi_split);
  * calibration-size curves (cal_size_curve).
"""
import numpy as np

LAMBDA_GRID = np.linspace(0.0, 1.0, 201)


def _fnr_at(probs, labels, lam):
    """Empirical false-negative rate of sets {c: p_c >= 1-lam} per record."""
    keep = probs >= (1.0 - lam)
    pos = labels.sum(axis=1)
    missed = (labels * (~keep)).sum(axis=1)
    mask = pos > 0
    if mask.sum() == 0:
        return 0.0
    return float((missed[mask] / pos[mask]).mean())


def crc_lambda(cal_probs, cal_labels, alpha, grid=LAMBDA_GRID):
    """Smallest lambda with upper-corrected risk <= alpha (B=1 bound)."""
    n = len(cal_probs)
    for lam in grid:
        risk = _fnr_at(cal_probs, cal_labels, lam)
        if (n / (n + 1)) * risk + 1.0 / (n + 1) <= alpha:
            return float(lam)
    return 1.0


def federated_lambda(client_cal, alpha, grid=LAMBDA_GRID):
    """client_cal: list of (probs, labels). Weighted-average risk curve."""
    sizes = np.array([len(p) for p, _ in client_cal], dtype=float)
    w = sizes / sizes.sum()
    n_tot = sizes.sum()
    for lam in grid:
        risk = sum(wi * _fnr_at(p, l, lam) for wi, (p, l) in zip(w, client_cal))
        if (n_tot / (n_tot + 1)) * risk + 1.0 / (n_tot + 1) <= alpha:
            return float(lam)
    return 1.0


def evaluate_sets(test_probs, test_labels, lam):
    keep = test_probs >= (1.0 - lam)
    return {
        "lambda": float(lam),
        "fnr": _fnr_at(test_probs, test_labels, lam),
        "mean_set_size": float(keep.sum(axis=1).mean()),
    }


def run_conformal(client_val, client_test, alpha=0.1, cal_alpha=None):
    """client_val/test: dicts name -> (probs, labels).

    Three modes per client:
      local       — lambda calibrated on the client's own data only.
      federated   — one shared lambda (group-level guarantee; can violate a
                    given hospital, e.g. high-prevalence sites).
      personalized— lambda_h = max(federated lambda, conservative local
                    lambda_h): the federated value acts as a shared floor,
                    raised to each hospital's own CRC threshold calibrated at
                    a tighter cal_alpha (<= alpha). The tighter target leaves
                    finite-sample headroom so even small-calibration sites
                    test under alpha, while larger sites benefit from the
                    federated prior.

    cal_alpha defaults to alpha (no conservatism). Set it below alpha (e.g.
    0.05 for alpha=0.1) to robustly hit FNR_h <= alpha at every hospital.
    """
    cal_alpha = alpha if cal_alpha is None else cal_alpha
    fed_lam = federated_lambda(list(client_val.values()), alpha)
    out = {}
    for name in client_val:
        pv, yv = client_val[name]
        pt, yt = client_test[name]
        local_lam = crc_lambda(pv, yv, alpha)            # reported "local"
        pers_local = crc_lambda(pv, yv, cal_alpha)       # conservative
        pers_lam = max(fed_lam, pers_local)
        out[name] = {
            "local": evaluate_sets(pt, yt, local_lam),
            "federated": evaluate_sets(pt, yt, fed_lam),
            "personalized": evaluate_sets(pt, yt, pers_lam),
        }
    return {"alpha": alpha, "cal_alpha": cal_alpha,
            "federated_lambda": fed_lam, "clients": out}


# ---------------------------------------------------------------- vectorized

def inclusion_index(probs, grid=LAMBDA_GRID):
    """Smallest grid index j at which each class enters {p >= 1 - grid[j]}
    (len(grid) if never). Uses exactly the comparison of _fnr_at."""
    neg_thr = -(1.0 - grid)                  # increasing
    return np.searchsorted(neg_thr, -probs, side="left")


def record_loss_matrix(probs, labels, grid=LAMBDA_GRID, class_weights=None):
    """(n_records, n_grid) per-record (weighted) false-negative proportion
    at every lambda. Rows without positives are NaN."""
    n, C = labels.shape
    w = np.ones(C) if class_weights is None else np.asarray(class_weights, float)
    idx = inclusion_index(probs, grid)
    rows, cols = np.nonzero(labels > 0)
    M = np.zeros((n, len(grid) + 1))
    np.add.at(M, (rows, idx[rows, cols]), w[cols])
    inc = np.cumsum(M, axis=1)[:, :len(grid)]
    tot = (labels * w[None, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        L = 1.0 - inc / tot[:, None]
    L[tot <= 0] = np.nan
    return np.clip(L, 0.0, 1.0, out=L, where=np.isfinite(L))


def _risk(L):
    valid = np.isfinite(L[:, 0])
    return (np.nanmean(L[valid], axis=0) if valid.any()
            else np.zeros(L.shape[1])), int(valid.sum())


def crc_from_curve(risk, n, alpha, grid=LAMBDA_GRID):
    """CRC lambda from a risk curve with n calibration points (B = 1)."""
    ok = (n / (n + 1.0)) * risk + 1.0 / (n + 1.0) <= alpha
    j = np.argmax(ok) if ok.any() else None
    return float(grid[j]) if j is not None else 1.0


def _h1(a, b):
    a = min(max(a, 0.0), 1.0)
    out = 0.0
    if a > 0:
        out += a * np.log(a / b)
    if a < 1:
        out += (1 - a) * np.log((1 - a) / (1 - b))
    return out


def hb_ucb(rhat, n, delta, tol=1e-7):
    """Hoeffding-Bentkus upper confidence bound on the mean of n i.i.d.
    [0,1]-bounded losses with empirical mean rhat (Bates et al., RCPS)."""
    from scipy.stats import binom

    def pval(R):
        hoeff = np.exp(-n * _h1(min(rhat, R), R))
        bent = np.e * binom.cdf(np.ceil(n * rhat), n, R)
        return min(hoeff, bent)

    lo, hi = rhat, 1.0 - 1e-12
    if pval(hi) >= delta:
        return 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if pval(mid) >= delta:
            lo = mid
        else:
            hi = mid
    return hi


_UCB_CACHE = {}


def ucb_risk_target(n, alpha, delta):
    """Largest empirical risk r with hb_ucb(r, n, delta) <= alpha: the
    size-adaptive calibration target (small n -> much tighter than alpha)."""
    key = (int(n), float(alpha), float(delta))
    if key in _UCB_CACHE:
        return _UCB_CACHE[key]
    lo, hi = 0.0, alpha
    if hb_ucb(0.0, n, delta) > alpha:
        _UCB_CACHE[key] = -1.0          # unattainable: always full sets
        return -1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if hb_ucb(mid, n, delta) <= alpha:
            lo = mid
        else:
            hi = mid
    _UCB_CACHE[key] = lo
    return lo


def ucb_from_curve(risk, n, alpha, delta, grid=LAMBDA_GRID):
    """RCPS lambda: smallest lambda whose UCB on the risk is <= alpha.
    (Risk is non-increasing in lambda, so this is the RCPS threshold.)"""
    r = ucb_risk_target(n, alpha, delta)
    ok = risk <= r
    return float(grid[np.argmax(ok)]) if (r >= 0 and ok.any()) else 1.0


def _set_size(P, lam):
    return float((P >= 1.0 - lam).sum(axis=1).mean())


def multi_split(client_preds, alphas=(0.1,), cal_alphas=(0.05,), n_splits=100,
                seed=0, ucb_delta=0.1, class_weights=None, grid=LAMBDA_GRID):
    """Repeated random calibration/test re-splits.

    client_preds: {name: (pv, yv, pt, yt)} from the deployed models. The
    validation and test records of each hospital are pooled and re-split
    n_splits times keeping the original calibration size. Modes:
      local (alpha), federated (alpha), personalized (max(fed, local at
      each cal_alpha)), ucb (max(fed, RCPS-UCB at alpha, ucb_delta)).
    Returns a list of row dicts: split, site, alpha, cal_alpha, mode,
    lambda, fnr (realized on the test part), size, n_cal."""
    rng = np.random.RandomState(seed)
    pooled = {}
    for name, (pv, yv, pt, yt) in client_preds.items():
        P = np.concatenate([pv, pt])
        Y = np.concatenate([yv, yt])
        pooled[name] = (P, record_loss_matrix(P, Y, grid, class_weights), len(pv))
    rows = []
    for s in range(n_splits):
        split = {}
        for name, (P, L, nc) in pooled.items():
            perm = rng.permutation(len(P))
            cal, test = perm[:nc], perm[nc:]
            risk, n = _risk(L[cal])
            split[name] = (cal, test, risk, n)
        sizes = np.array([split[k][3] for k in pooled], float)
        fed_curve = sum(sz * split[k][2] for sz, k in zip(sizes, pooled)) / sizes.sum()
        for alpha in alphas:
            fed = crc_from_curve(fed_curve, sizes.sum(), alpha, grid)
            for name, (P, L, nc) in pooled.items():
                cal, test, risk, n = split[name]
                Lt, Pt = L[test], P[test]
                lam_modes = [("local", None, crc_from_curve(risk, n, alpha, grid)),
                             ("federated", None, fed)]
                for ca in cal_alphas:
                    lam_modes.append(("personalized", ca,
                                      max(fed, crc_from_curve(risk, n, ca, grid))))
                if ucb_delta:
                    lam_modes.append(("ucb", None,
                                      max(fed, ucb_from_curve(risk, n, alpha,
                                                              ucb_delta, grid))))
                for mode, ca, lam in lam_modes:
                    j = int(round(lam * (len(grid) - 1)))
                    rows.append({"split": s, "site": name, "alpha": alpha,
                                 "cal_alpha": ca, "mode": mode, "lambda": lam,
                                 "fnr": float(np.nanmean(Lt[:, j])),
                                 "size": _set_size(Pt, lam), "n_cal": n})
    return rows


def cal_size_curve(P, Y, sizes, n_rep=200, alpha=0.1, cal_alphas=(0.05,),
                   ucb_delta=0.1, seed=0, grid=LAMBDA_GRID):
    """Local calibration with n records drawn from one hospital's pooled
    data; the rest is the test set. Shows how many calibration records a
    (new) hospital needs for stable per-site control."""
    rng = np.random.RandomState(seed)
    L = record_loss_matrix(P, Y, grid)
    rows = []
    for n_cal in sizes:
        if n_cal >= len(P) - 50:
            continue
        for r in range(n_rep):
            perm = rng.permutation(len(P))
            cal, test = perm[:n_cal], perm[n_cal:]
            risk, n = _risk(L[cal])
            modes = [("local", alpha, crc_from_curve(risk, n, alpha, grid))]
            modes += [("local", ca, crc_from_curve(risk, n, ca, grid))
                      for ca in cal_alphas]
            if ucb_delta:
                modes.append(("ucb", alpha,
                              ucb_from_curve(risk, n, alpha, ucb_delta, grid)))
            for mode, a, lam in modes:
                j = int(round(lam * (len(grid) - 1)))
                rows.append({"n_cal": n_cal, "rep": r, "mode": mode,
                             "target": a, "lambda": lam,
                             "fnr": float(np.nanmean(L[test][:, j])),
                             "size": _set_size(P[test], lam)})
    return rows


def classwise_multi_split(client_preds, critical, alpha=0.1, alpha_c=0.05,
                          cal_alpha=0.05, cal_alpha_c=None, n_splits=100,
                          seed=0, grid=LAMBDA_GRID):
    """Class-conditional risk control for critical diagnoses.

    For each critical class c, lambda_c is calibrated on the calibration
    records that carry c (loss = c missed), federated across hospitals as a
    floor and personalized per hospital exactly like the record-level rule;
    the final set uses max(lambda_h, lambda_c) for critical classes and
    lambda_h otherwise. Returns rows with the realized per-class miss rate
    (on test positives of c) and the overall FNR / set size, for the plain
    personalized rule and the class-conditional rule."""
    cal_alpha_c = alpha_c * cal_alpha / alpha if cal_alpha_c is None else cal_alpha_c
    rng = np.random.RandomState(seed)
    pooled = {}
    for name, (pv, yv, pt, yt) in client_preds.items():
        P = np.concatenate([pv, pt])
        Y = np.concatenate([yv, yt])
        pooled[name] = (P, Y, record_loss_matrix(P, Y, grid), len(pv))
    rows = []
    for s in range(n_splits):
        sp = {}
        for name, (P, Y, L, nc) in pooled.items():
            perm = rng.permutation(len(P))
            sp[name] = (perm[:nc], perm[nc:])
        # record-level personalized lambda (as in the main analysis)
        curves = {k: _risk(pooled[k][2][sp[k][0]]) for k in pooled}
        n_all = np.array([curves[k][1] for k in pooled], float)
        fed = crc_from_curve(sum(n * curves[k][0] for n, k in zip(n_all, pooled))
                             / n_all.sum(), n_all.sum(), alpha, grid)
        lam_h = {k: max(fed, crc_from_curve(curves[k][0], curves[k][1],
                                            cal_alpha, grid)) for k in pooled}
        # per-class lambdas: miss indicator of class c among its positives
        lam_c = {k: {} for k in pooled}
        for c in critical:
            per_site = {}
            for k, (P, Y, L, nc) in pooled.items():
                cal = sp[k][0]
                pos = cal[Y[cal, c] > 0]
                if len(pos) == 0:
                    per_site[k] = (np.zeros(len(grid)), 0)
                    continue
                miss = (inclusion_index(P[pos, c], grid)[:, None]
                        > np.arange(len(grid))[None, :]).mean(axis=0)
                per_site[k] = (miss, len(pos))
            n_c = np.array([per_site[k][1] for k in pooled], float)
            fed_c = (crc_from_curve(sum(n * per_site[k][0] for n, k in
                                        zip(n_c, pooled)) / n_c.sum(),
                                    n_c.sum(), alpha_c, grid)
                     if n_c.sum() > 0 else 1.0)
            for k in pooled:
                miss, n = per_site[k]
                loc = crc_from_curve(miss, n, cal_alpha_c, grid) if n > 0 else 1.0
                lam_c[k][c] = max(fed_c, loc)
        for k, (P, Y, L, nc) in pooled.items():
            test = sp[k][1]
            Pt, Yt = P[test], Y[test]
            for rule in ("personalized", "classwise"):
                lam_vec = np.full(P.shape[1], lam_h[k])
                if rule == "classwise":
                    for c in critical:
                        lam_vec[c] = max(lam_h[k], lam_c[k][c])
                keep = Pt >= (1.0 - lam_vec[None, :])
                pos = Yt.sum(axis=1)
                fnr = float(((Yt * ~keep).sum(axis=1)[pos > 0] / pos[pos > 0]).mean())
                row = {"split": s, "site": k, "rule": rule, "fnr": fnr,
                       "size": float(keep.sum(axis=1).mean())}
                for c in critical:
                    m = Yt[:, c] > 0
                    row[f"miss_{c}"] = float((~keep[m, c]).mean()) if m.any() else np.nan
                    row[f"npos_{c}"] = int(m.sum())
                rows.append(row)
    return rows
