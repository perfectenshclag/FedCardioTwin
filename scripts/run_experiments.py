#!/usr/bin/env python3
"""Single entry point for every experiment in the paper.

Examples:
  python scripts/run_experiments.py --stage centralized --preset fast
  python scripts/run_experiments.py --stage federated --strategies fedavg fedbn
  python scripts/run_experiments.py --stage twin
  python scripts/run_experiments.py --stage conformal
Results land in results/<stage>_<strategy><tag>_seed<k>.json; validation and
test predictions of every deployed model land in results/preds/*.npz so all
post-hoc analyses (per-class metrics, conformal re-splits) run on CPU.

Revision examples:
  # main table + round-budget curves: 120 rounds, Table III at round 60
  ... --stage federated --rounds 120 --main-round 60 \\
      --snapshot-rounds 20 40 60 80 100 120 --eval-every 5
  # personalization depth / weighting / participation / leads / DP
  ... --stage federated --strategies fedper --local-blocks 1 --tag _depth1
  ... --stage federated --strategies fedavg fedper --weighting equal --tag _eqw
  ... --stage federated --strategies fedper --participation 0.5 --tag _part50
  ... --stage federated --strategies fedper --leads 2 --tag _leads2
  ... --stage federated --strategies fedper --dp-epsilon 3 --tag _dp3
  ... --stage finetune   # FedAvg+FT and Centralized+FT
  ... --stage loho       # held-out hospital + onboarding curve
  ... --stage twin_fed   # twin on the deployed federated model
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from fedcardiotwin.configs import LEAD_SETS, fast_preset, full_preset
from fedcardiotwin.data.dataset import (CinCPTBXLStreams, PTBXLStreams,
                                        default_workers, load_clients,
                                        twin_heldout_rows)
from fedcardiotwin.data.labels import ScoredLabelSpace
from fedcardiotwin.fl.engine import _bn_keys, _head_keys, run_federated
from fedcardiotwin.models import build_model
from fedcardiotwin.train.metrics import summarize_clients
from fedcardiotwin.train.trainer import (evaluate, local_finetune, predict,
                                         train_model)
from fedcardiotwin.twin.loop import run_twin_evaluation
from fedcardiotwin.conformal.crc import run_conformal
from fedcardiotwin.utils import (get_device, get_logger, load_json, save_json,
                                 seed_everything)

log = get_logger()

STRATEGIES = ["local", "fedavg", "fedprox", "fedbn", "fedavgm", "ditto", "fedper"]
ALL_STRATEGIES = STRATEGIES + ["fedrep", "fedbabu"]


def save_preds(path, preds):
    """preds: {client: (pv, yv, pt, yt)} -> compressed npz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrs = {}
    for name, (pv, yv, pt, yt) in preds.items():
        arrs[f"{name}__pv"], arrs[f"{name}__yv"] = pv.astype(np.float32), yv.astype(np.uint8)
        arrs[f"{name}__pt"], arrs[f"{name}__yt"] = pt.astype(np.float32), yt.astype(np.uint8)
    np.savez_compressed(path, **arrs)


def load_preds(path):
    z = np.load(path)
    names = sorted({k.split("__")[0] for k in z.files})
    return {n: (z[f"{n}__pv"], z[f"{n}__yv"], z[f"{n}__pt"], z[f"{n}__yt"])
            for n in names}


def eval_clients(model_or_states, clients, build, device, batch_size):
    """Evaluate one model (or per-client states) on every client."""
    res, preds = {}, {}
    for c in clients:
        if isinstance(model_or_states, dict):
            m = build()
            m.load_state_dict(model_or_states[c.name])
        else:
            m = model_or_states
        metrics, (pt, yt, _), (pv, yv) = evaluate(
            m, c.loader("val", batch_size), c.loader("test", batch_size),
            device, return_preds=True)
        res[c.name] = metrics
        preds[c.name] = (pv, yv, pt, yt)
    return res, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["centralized", "federated", "twin", "twin_fed",
                             "conformal", "loho", "finetune"])
    ap.add_argument("--preset", default="full", choices=["fast", "full"])
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--hospitals", nargs="+", default=None,
                    help="loho: restrict which hospitals to hold out (for "
                         "splitting the 6 hold-outs across GPUs/processes)")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--eval-repo", default="external/evaluation-2021")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="override the preset's seeds (e.g. --seeds 0 to "
                         "regenerate only seed 0)")
    # --- revision options (defaults reproduce the submitted protocol) ---
    ap.add_argument("--tag", default="", help="suffix of result/ckpt names")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--main-round", type=int, default=None)
    ap.add_argument("--snapshot-rounds", nargs="+", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--weighting", default=None,
                    choices=["samples", "equal", "label"])
    ap.add_argument("--participation", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--local-blocks", type=int, default=None)
    ap.add_argument("--leads", default=None, choices=sorted(LEAD_SETS))
    ap.add_argument("--norm", default=None, choices=["bn", "gn"])
    ap.add_argument("--dp-epsilon", type=float, default=None)
    ap.add_argument("--prox-mu", type=float, default=None)
    ap.add_argument("--ditto-lambda", type=float, default=None)
    ap.add_argument("--twin-split", action="store_true",
                    help="withhold PTB-XL fold 9-10 patients from the "
                         "federation (they are the federated twin's streams)")
    ap.add_argument("--conformal-strategy", default="fedper")
    ap.add_argument("--ft-bases", nargs="+", default=["fedavg", "central"])
    ap.add_argument("--onboard-sizes", nargs="+", type=int,
                    default=[50, 100, 250, 500, 1000, 0],
                    help="loho onboarding: labelled records (0 = all)")
    ap.add_argument("--max-records", type=int, default=None,
                    help="per-client subsample (smoke tests only)")
    args = ap.parse_args()

    cfg = fast_preset() if args.preset == "fast" else full_preset()
    if args.seeds is not None:
        cfg.seeds = tuple(args.seeds)
    fl = cfg.fl
    for attr, val in (("rounds", args.rounds), ("main_round", args.main_round),
                      ("eval_every", args.eval_every),
                      ("weighting", args.weighting),
                      ("participation", args.participation),
                      ("dropout", args.dropout),
                      ("local_blocks", args.local_blocks),
                      ("dp_epsilon", args.dp_epsilon),
                      ("prox_mu", args.prox_mu),
                      ("ditto_lambda", args.ditto_lambda)):
        if val is not None:
            setattr(fl, attr, val)
    if args.snapshot_rounds is not None:
        fl.snapshot_rounds = tuple(args.snapshot_rounds)
    leads = LEAD_SETS[args.leads] if args.leads else None
    model_kw = {}
    if leads is not None:
        model_kw["in_ch"] = len(leads)
    if args.norm:
        model_kw["norm"] = args.norm
    tag = args.tag

    def build(n_classes=None):
        return build_model(cfg.model, n_classes or space.num_classes, **model_kw)

    device = get_device()
    log.info(f"device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    preds_dir = os.path.join(args.results_dir, "preds")

    if args.stage in ("centralized", "federated", "conformal", "loho", "finetune"):
        space = ScoredLabelSpace(args.eval_repo)
        exclude = ({"PTB-XL": twin_heldout_rows(args.cache_dir)}
                   if args.twin_split else None)
        clients = load_clients(args.cache_dir, cfg.clients, leads=leads,
                               exclude=exclude, max_records=args.max_records)
        log.info(f"clients: {[(c.name, len(c.train_idx), len(c.val_idx), len(c.test_idx)) for c in clients]}")

    if args.stage == "centralized":
        # Pooled reference: concatenate all clients' train data.
        from torch.utils.data import ConcatDataset, DataLoader
        from fedcardiotwin.data.dataset import ECGDataset
        for seed in cfg.seeds:
            result_path = f"{args.results_dir}/centralized{tag}_seed{seed}.json"
            if os.path.exists(result_path):
                log.info(f"[centralized] seed{seed} already done, skipping")
                continue
            seed_everything(seed)
            tr = ConcatDataset([ECGDataset(c.X, c.Y, c.train_idx, True,
                                           augment=cfg.central.augment,
                                           leads=leads)
                                for c in clients])
            loader = DataLoader(tr, batch_size=cfg.central.batch_size, shuffle=True,
                                num_workers=default_workers(), pin_memory=torch.cuda.is_available(),
                                drop_last=True)
            model = build()
            progress_path = f"{args.ckpt_dir}/central{tag}_seed{seed}_progress.pt"
            if os.path.exists(progress_path):
                ck = torch.load(progress_path, map_location="cpu")
                model.load_state_dict(ck["model"])
                log.info(f"[centralized] seed{seed} resumed from epoch {ck['epoch']}")
            train_model(model, loader, device, cfg.central.epochs, lr=cfg.central.lr,
                        mixup_alpha=cfg.central.mixup_alpha,
                        ema_decay=cfg.central.ema_decay, log_fn=log.info,
                        ckpt_path=progress_path, ckpt_every=5)
            res, preds = eval_clients(model, clients, build, device,
                                      cfg.central.batch_size)
            torch.save(model.state_dict(), f"{args.ckpt_dir}/central{tag}_seed{seed}.pt")
            save_preds(f"{preds_dir}/centralized{tag}_seed{seed}.npz", preds)
            save_json(summarize_clients(res), result_path)
            if os.path.exists(progress_path):
                os.remove(progress_path)

    elif args.stage == "federated":
        for strategy in args.strategies:
            for seed in cfg.seeds:
                run = f"fed_{strategy}{tag}_seed{seed}"
                result_path = f"{args.results_dir}/{run}.json"
                if os.path.exists(result_path):
                    log.info(f"[federated] {run} already done, skipping")
                    continue
                seed_everything(seed)
                progress_path = f"{args.ckpt_dir}/{run}_progress.pt"
                try:
                    res, history, comm_mb, gmodel, deployed, preds = run_federated(
                        clients, space.num_classes, fl, device,
                        model_name=cfg.model, strategy=strategy, seed=seed,
                        ckpt_path=progress_path, model_kw=model_kw)
                except Exception as e:
                    # One strategy/seed diverging or erroring must not abort the
                    # remaining runs in this process. Drop its (possibly poisoned)
                    # progress so a future invocation restarts it cleanly.
                    log.error(f"[federated] {run} FAILED: {e!r}")
                    if os.path.exists(progress_path):
                        os.remove(progress_path)
                    continue
                res["comm_mb_per_round"] = comm_mb
                res["history"] = history  # convergence / comm-budget curves
                res["config"] = {"strategy": strategy, "seed": seed, "tag": tag,
                                 "rounds": fl.rounds, "main_round": res["main_round"],
                                 "weighting": fl.weighting,
                                 "participation": fl.participation,
                                 "dropout": fl.dropout,
                                 "local_blocks": fl.local_blocks,
                                 "leads": args.leads or "12",
                                 "norm": ("gn" if fl.dp_epsilon > 0 else model_kw.get("norm", "bn")),
                                 "dp_epsilon": fl.dp_epsilon,
                                 "prox_mu": fl.prox_mu,
                                 "ditto_lambda": fl.ditto_lambda,
                                 "twin_split": args.twin_split}
                torch.save(gmodel.state_dict(), f"{args.ckpt_dir}/{run}.pt")
                # deployed per-client models: what conformal must evaluate
                torch.save(deployed, f"{args.ckpt_dir}/{run}_clients.pt")
                save_preds(f"{preds_dir}/{run}.npz", preds)
                save_json(res, result_path)
                if os.path.exists(progress_path):
                    os.remove(progress_path)

    elif args.stage == "finetune":
        # Local fine-tuning baselines: every hospital fine-tunes the shared
        # model (all parameters) on its own training split; the epoch is
        # selected on its validation split (epoch 0 = no fine-tuning).
        for base in args.ft_bases:
            for seed in cfg.seeds:
                run = f"ft_{base}{tag}_seed{seed}"
                result_path = f"{args.results_dir}/{run}.json"
                if os.path.exists(result_path):
                    log.info(f"[finetune] {run} already done, skipping")
                    continue
                src = (f"{args.ckpt_dir}/central{tag}_seed{seed}.pt" if base == "central"
                       else f"{args.ckpt_dir}/fed_{base}{tag}_seed{seed}.pt")
                if not os.path.exists(src):
                    log.error(f"[finetune] missing {src}, skipping")
                    continue
                seed_everything(seed)
                base_state = torch.load(src, map_location="cpu")
                res, preds, deployed, info = {}, {}, {}, {}
                for c in clients:
                    m = build()
                    m.load_state_dict(base_state)
                    m, inf = local_finetune(m, c.loader("train", fl.batch_size),
                                            c.loader("val", fl.batch_size),
                                            device, fl.ft_epochs, fl.ft_lr)
                    metrics, (pt, yt, _), (pv, yv) = evaluate(
                        m, c.loader("val", fl.batch_size),
                        c.loader("test", fl.batch_size), device, return_preds=True)
                    res[c.name], preds[c.name], info[c.name] = metrics, (pv, yv, pt, yt), inf
                    deployed[c.name] = {k: v.cpu() for k, v in m.state_dict().items()}
                    log.info(f"[finetune] {run} {c.name}: AUROC={metrics['auroc']:.4f} "
                             f"(epoch {inf['best_epoch']})")
                out = summarize_clients(res)
                out["finetune"] = info
                torch.save(deployed, f"{args.ckpt_dir}/{run}_clients.pt")
                save_preds(f"{preds_dir}/{run}.npz", preds)
                save_json(out, result_path)

    elif args.stage == "twin":
        streams_db = PTBXLStreams(args.cache_dir)
        from torch.utils.data import DataLoader
        from fedcardiotwin.data.dataset import ECGDataset
        for seed in cfg.seeds:
            result_path = f"{args.results_dir}/twin{tag}_seed{seed}.json"
            if os.path.exists(result_path):
                log.info(f"[twin] seed{seed} already done, skipping")
                continue
            seed_everything(seed)
            base_idx = streams_db.base_indices()
            loader = DataLoader(
                ECGDataset(streams_db.X, streams_db.Y, base_idx, True,
                           augment=cfg.central.augment),
                batch_size=cfg.central.batch_size, shuffle=True,
                num_workers=default_workers(), pin_memory=torch.cuda.is_available(),
                                drop_last=True)
            base = build_model(cfg.model, streams_db.Y.shape[1])
            progress_path = f"{args.ckpt_dir}/twin{tag}_seed{seed}_progress.pt"
            if os.path.exists(progress_path):
                ck = torch.load(progress_path, map_location="cpu")
                base.load_state_dict(ck["model"])
                log.info(f"[twin] seed{seed} resumed from epoch {ck['epoch']}")
            train_model(base, loader, device, cfg.central.epochs,
                        lr=cfg.central.lr, mixup_alpha=cfg.central.mixup_alpha,
                        ema_decay=cfg.central.ema_decay, log_fn=log.info,
                        ckpt_path=progress_path, ckpt_every=5)
            torch.save(base.state_dict(), f"{args.ckpt_dir}/twin_base{tag}_seed{seed}.pt")
            streams = streams_db.eval_streams()
            res, arrays = run_twin_evaluation(base, streams, streams_db.X,
                                              streams_db.Y, device, cfg.twin)
            np.savez(f"{args.results_dir}/twin{tag}_seed{seed}_arrays.npz", **arrays)
            save_json(res, result_path)
            log.info(f"twin seed{seed}: {res}")
            if os.path.exists(progress_path):
                os.remove(progress_path)

    elif args.stage == "twin_fed":
        # Twin on the DEPLOYED federated model: the PTB-XL hospital's FedPer
        # model from a federation that never saw the fold 9-10 patients
        # (run: --stage federated --strategies fedper --twin-split
        #        --tag _twinfed).
        space = ScoredLabelSpace(args.eval_repo)
        sdb = CinCPTBXLStreams(args.cache_dir)
        streams = sdb.eval_streams()
        log.info(f"[twin_fed] {len(streams)} patients, "
                 f"{sum(len(r) for _, r in streams)} records")
        src_tag = tag or "_twinfed"
        for seed in cfg.seeds:
            result_path = f"{args.results_dir}/twinfed_seed{seed}.json"
            if os.path.exists(result_path):
                log.info(f"[twin_fed] seed{seed} already done, skipping")
                continue
            src = f"{args.ckpt_dir}/fed_fedper{src_tag}_seed{seed}_clients.pt"
            if not os.path.exists(src):
                log.error(f"[twin_fed] missing {src}; run the twin-split "
                          f"federation first")
                continue
            seed_everything(seed)
            base = build_model(cfg.model, space.num_classes)
            base.load_state_dict(torch.load(src, map_location="cpu")["PTB-XL"])
            res, arrays = run_twin_evaluation(base, streams, sdb.X, sdb.Y,
                                              device, cfg.twin)
            np.savez(f"{args.results_dir}/twinfed_seed{seed}_arrays.npz", **arrays)
            save_json(res, result_path)
            log.info(f"twin_fed seed{seed}: {res}")

    elif args.stage == "conformal":
        # Single-split conformal on the *deployed* per-client models of the
        # primary strategy (FedPer). The repeated re-split analysis runs on
        # CPU from the saved predictions (scripts/posthoc.py).
        strat = args.conformal_strategy
        seed = cfg.seeds[0]
        ppath = f"{preds_dir}/fed_{strat}{tag}_seed{seed}.npz"
        if os.path.exists(ppath):
            preds = load_preds(ppath)
        else:
            deployed_path = f"{args.ckpt_dir}/fed_{strat}{tag}_seed{seed}_clients.pt"
            if not os.path.exists(deployed_path):
                raise FileNotFoundError(f"Run --stage federated for {strat} first "
                                        f"({deployed_path} not found).")
            deployed = torch.load(deployed_path, map_location="cpu")
            _, preds = eval_clients(deployed, clients, build, device, fl.batch_size)
        log.info(f"conformal: using {strat}{tag} seed{seed}")
        cal = {n: (p[0], p[1]) for n, p in preds.items()}
        test = {n: (p[2], p[3]) for n, p in preds.items()}
        res = run_conformal(cal, test, alpha=cfg.conformal.alpha,
                            cal_alpha=cfg.conformal.cal_alpha)
        res["strategy"] = f"{strat}{tag}"
        save_json(res, f"{args.results_dir}/conformal{tag}.json")
        log.info(f"conformal: federated lambda={res['federated_lambda']:.3f}")

    elif args.stage == "loho":
        # Leave-one-hospital-out: train fedavg on 5 clients, deploy on the
        # unseen 6th. Thresholds come from the training clients' pooled val
        # (the held-out hospital contributes nothing to model or tuning).
        # Then the onboarding curve: the new hospital trains a private head
        # (+ its normalization layers) on n labelled records.
        from fedcardiotwin.train.metrics import (macro_auroc, macro_f1,
                                                 tune_thresholds)
        seed = cfg.seeds[0]
        targets = list(clients)
        if args.hospitals:
            targets = [c for c in clients if c.name in set(args.hospitals)]
        for held in targets:
            out_path = f"{args.results_dir}/loho_{held.name}{tag}_seed{seed}.json"
            if os.path.exists(out_path):
                log.info(f"[loho] {held.name} already done, skipping")
                continue
            train_clients = [c for c in clients if c.name != held.name]
            seed_everything(seed)
            gpath = f"{args.ckpt_dir}/loho_{held.name}{tag}_seed{seed}.pt"
            if os.path.exists(gpath):
                gmodel = build()
                gmodel.load_state_dict(torch.load(gpath, map_location="cpu"))
            else:
                log.info(f"[loho] holding out {held.name}")
                _, _, _, gmodel, _, _ = run_federated(
                    train_clients, space.num_classes, fl, device,
                    model_name=cfg.model, strategy="fedavg", seed=seed,
                    model_kw=model_kw)
                torch.save(gmodel.state_dict(), gpath)
            pv_all, yv_all = [], []
            for c in train_clients:
                pv, yv = predict(gmodel, c.loader("val", fl.batch_size), device)
                pv_all.append(pv)
                yv_all.append(yv)
            th = tune_thresholds(np.concatenate(yv_all), np.concatenate(pv_all))
            pt, yt = predict(gmodel, held.loader("test", fl.batch_size), device)
            res = {"auroc": macro_auroc(yt, pt), "f1": macro_f1(yt, pt, th)}
            log.info(f"[loho] {held.name}: {res}")

            # onboarding: private head + normalization layers, body shared
            trainable = ({n for n, _ in gmodel.named_parameters()}
                         & (_head_keys(gmodel) | _bn_keys(gmodel)))
            onboarding = []
            rng = np.random.RandomState(seed)
            for n_lab in args.onboard_sizes:
                full = n_lab <= 0 or n_lab >= len(held.train_idx)
                reps = 1 if full else 3
                for r in range(reps):
                    idx = (held.train_idx if full else
                           rng.choice(held.train_idx, n_lab, replace=False))
                    m = build()
                    m.load_state_dict(gmodel.state_dict())
                    epochs = 10 if len(idx) >= 1000 else 20
                    m, inf = local_finetune(
                        m, held.loader("train", fl.batch_size, indices=idx),
                        held.loader("val", fl.batch_size), device, epochs,
                        fl.lr, trainable=trainable)
                    metrics, _ = evaluate(m, held.loader("val", fl.batch_size),
                                          held.loader("test", fl.batch_size),
                                          device)
                    onboarding.append({"n": int(len(idx)), "rep": r,
                                       "best_epoch": inf["best_epoch"], **metrics})
                    log.info(f"[loho] {held.name} onboard n={len(idx)} rep{r}: "
                             f"AUROC={metrics['auroc']:.4f}")
            save_json({held.name: res, "onboarding": onboarding}, out_path)

    log.info("stage complete.")


if __name__ == "__main__":
    main()
