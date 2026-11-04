"""Serial federated simulation engine (single-GPU friendly).

Strategies: local-only, fedavg, fedprox, fedbn (local normalization),
fedavgm (server momentum), ditto, and the shared-encoder/private-head family
fedper (head private), fedrep (alternating head/body updates) and fedbabu
(head frozen during FL, local fine-tuning at deployment). For the private-
head family all normalization layers also stay local, as in FedBN.

A serial loop is deliberate: it is exactly reproducible, runs anywhere a
notebook runs, and matches how FL papers simulate cross-silo settings.

Revision features (all default to the submitted protocol): aggregation
weighting (samples | equal | label-aware FedLA), partial participation and
client dropout, personalization depth (extra private blocks), round-budget
snapshots, upload AND download accounting, per-client timing/peak memory,
and optional record-level DP-SGD.

Outputs per run: per-client metrics of the deployed models at the main
round, a per-round validation-AUROC history, communication cost, the
deployed per-client state dicts (global shared weights composed with each
client's private tensors) and their val/test predictions, so downstream
stages (conformal, per-class analysis) evaluate exactly what a hospital
would run.
"""
import copy
import os
import time

import numpy as np
import torch

from ..models import build_model
from ..train.metrics import macro_auroc, summarize_clients
from ..train.trainer import (cache_features, evaluate, local_finetune, predict,
                             train_head_on_features, train_model, train_model_dp)
from ..utils import count_bytes, get_logger

log = get_logger()

NORM_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.GroupNorm)
PRIVATE_HEAD = ("fedper", "fedrep", "fedbabu")


def _bn_keys(model):
    """State keys of every normalization layer (BatchNorm or GroupNorm)."""
    keys = set()
    for name, mod in model.named_modules():
        if isinstance(mod, NORM_TYPES):
            for suffix in ("weight", "bias", "running_mean", "running_var",
                           "num_batches_tracked"):
                keys.add(f"{name}.{suffix}")
    return keys & set(model.state_dict())


def _head_keys(model):
    return {k for k in model.state_dict() if k.startswith("head.")}


def _block_keys(model, n_blocks):
    """The last n_blocks Inception blocks plus the residual shortcut that
    closes them: the extra depth kept private in the depth ablation."""
    if n_blocks <= 0:
        return set()
    depth = len(model.blocks)
    prefixes = [f"blocks.{depth - i}." for i in range(1, n_blocks + 1)]
    prefixes.append(f"shortcuts.{len(model.shortcuts) - 1}.")
    return {k for k in model.state_dict() if k.startswith(tuple(prefixes))}


def private_keys(model, strategy, local_blocks=0):
    """Tensors that never leave the hospital for this strategy."""
    bn = _bn_keys(model)
    if strategy == "fedbn":
        return bn
    if strategy in PRIVATE_HEAD:
        return bn | _head_keys(model) | _block_keys(model, local_blocks)
    return set()


def client_weights(clients, scheme="samples"):
    """Aggregation weights of Eq. (1).

    samples: n_k (FedAvg).  equal: 1 per hospital.
    label:   FedLA (Dege & Hoog Antink, CinC 2026): for every label, each
             client's share of that label's total count; shares summed over
             labels and normalized across clients."""
    if scheme == "equal":
        return [1.0] * len(clients)
    if scheme == "label":
        counts = np.stack([c.label_counts() for c in clients])  # K x C
        tot = counts.sum(axis=0)
        share = counts[:, tot > 0] / tot[tot > 0]
        w = share.sum(axis=1)
        return (w / w.sum()).tolist()
    if scheme != "samples":
        raise ValueError(f"unknown weighting {scheme}")
    return [float(len(c.train_idx)) for c in clients]


def participation_schedule(n_clients, rounds, participation=1.0, dropout=0.0,
                           seed=0):
    """Participants of every round, fixed up front (resume-safe).
    participation < 1 samples that fraction of hospitals uniformly per round;
    dropout > 0 then removes each sampled hospital independently with that
    probability (at least one hospital always reports)."""
    rng = np.random.RandomState(10007 + seed)
    sched = []
    for _ in range(rounds):
        idx = list(range(n_clients))
        if participation < 1.0:
            m = max(1, int(round(participation * n_clients)))
            idx = sorted(rng.choice(n_clients, m, replace=False).tolist())
        if dropout > 0:
            keep = [i for i in idx if rng.rand() >= dropout]
            idx = keep if keep else [idx[rng.randint(len(idx))]]
        sched.append(idx)
    return sched


def _avg_states(states, weights, keys):
    out = {}
    total = sum(weights)
    for k in keys:
        acc = None
        for s, w in zip(states, weights):
            t = s[k].float() * (w / total)
            acc = t if acc is None else acc + t
        out[k] = acc.to(states[0][k].dtype)
    return out


def _fp16_bytes(state_dict, keys):
    return sum(v.numel() * (2 if v.is_floating_point() else v.element_size())
               for k, v in state_dict.items() if k in keys)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        import platform
        import subprocess
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        except Exception:
            chip = platform.processor()
        return f"{chip} GPU (MPS, FP32)"
    return str(device)


def run_federated(clients, num_classes, cfg, device, model_name="inception1d",
                  strategy="fedavg", seed=0, ckpt_path=None, resume=True,
                  model_kw=None):
    """Returns (summary, history, comm_mb_per_round, global_model,
    deployed_states {client: state_dict}, preds {client: (pv, yv, pt, yt)}).

    summary holds per-client metrics at the main round (cfg.main_round, or
    the last round) plus 'snapshots', 'comm', 'timing', 'weights',
    'participation' and, for DP runs, 'dp'.

    If ckpt_path is given, progress (global model, local/personal states,
    momentum, history, completed round) is saved after every round and
    reloaded on resume, so an interrupted run continues instead of
    restarting from round 0."""
    model_kw = dict(model_kw or {})
    dp = cfg.dp_epsilon > 0
    if dp:
        model_kw["norm"] = "gn"
        assert strategy in ("fedavg", "fedper"), "DP runs: fedavg/fedper only"

    def build():
        return build_model(model_name, num_classes, **model_kw)

    torch.manual_seed(seed)
    global_model = build()
    all_keys = set(global_model.state_dict().keys())
    param_keys = {n for n, _ in global_model.named_parameters()}  # learnable only
    private = private_keys(global_model, strategy, cfg.local_blocks)
    shared_keys = all_keys - private

    local_states = [copy.deepcopy(global_model.state_dict()) for _ in clients]
    personal = ([copy.deepcopy(global_model.state_dict()) for _ in clients]
                if strategy == "ditto" else None)
    momentum = None
    history = []
    gst0 = global_model.state_dict()
    is_local = strategy == "local"
    shared_bytes = 0 if is_local else count_bytes(gst0, shared_keys)
    comm_mb = shared_bytes / 1e6
    comm_mb_fp16 = 0.0 if is_local else _fp16_bytes(gst0, shared_keys) / 1e6
    n_params = sum(p.numel() for p in global_model.parameters())

    train_loaders = [c.loader("train", cfg.batch_size) for c in clients]
    weights = client_weights(clients, cfg.weighting)

    rounds = 1 if is_local else cfg.rounds
    local_epochs = cfg.local_only_epochs if is_local else cfg.local_epochs
    main_round = rounds if is_local else min(cfg.main_round or rounds, rounds)
    snap_rounds = sorted(set(int(r) for r in cfg.snapshot_rounds if r <= rounds)
                         | {main_round}) if not is_local else [rounds]
    schedule = (participation_schedule(len(clients), rounds, cfg.participation,
                                       cfg.dropout, seed)
                if not is_local else [list(range(len(clients)))])

    # record-level DP: per-client noise multiplier for the whole run's steps
    accountants, sigmas = None, None
    if dp:
        from opacus.accountants import RDPAccountant
        from opacus.accountants.utils import get_noise_multiplier
        accountants = [RDPAccountant() for _ in clients]
        sigmas = []
        for ci, dl in enumerate(train_loaders):
            n_parts = sum(ci in p for p in schedule)
            sigmas.append(get_noise_multiplier(
                target_epsilon=cfg.dp_epsilon, target_delta=cfg.dp_delta,
                sample_rate=1.0 / len(dl), steps=n_parts * local_epochs * len(dl),
                accountant="rdp"))
        log.info(f"[{strategy}] DP eps={cfg.dp_epsilon}: sigma per client "
                 f"{[round(s, 3) for s in sigmas]}")

    timing = {c.name: [] for c in clients}
    peak_mem = {c.name: 0.0 for c in clients}
    agg_time = []
    comm_log = []
    snapshots = {}
    main_stash = None
    cum_up = cum_down = 0.0

    start_round = 0
    if ckpt_path and resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        global_model.load_state_dict(ck["global_state"])
        local_states = ck["local_states"]
        personal = ck["personal"]
        momentum = ck["momentum"]
        history = ck["history"]
        start_round = ck["round"]
        ex = ck.get("extra", {})
        timing = ex.get("timing", timing)
        peak_mem = ex.get("peak_mem", peak_mem)
        agg_time = ex.get("agg_time", agg_time)
        comm_log = ex.get("comm_log", comm_log)
        snapshots = ex.get("snapshots", snapshots)
        main_stash = ex.get("main_stash", main_stash)
        cum_up, cum_down = ex.get("cum", (cum_up, cum_down))
        if dp and ex.get("acc_history") is not None:
            for a, h in zip(accountants, ex["acc_history"]):
                a.history = list(h)
        log.info(f"[{strategy}] resumed from round {start_round}/{rounds}")

    def deployed_state(ci):
        """What client ci actually runs in production (before any
        deployment-time fine-tuning)."""
        if is_local:
            return copy.deepcopy(local_states[ci])
        if strategy == "ditto":
            return copy.deepcopy(personal[ci])
        st = copy.deepcopy(global_model.state_dict())
        for k in private:
            st[k] = local_states[ci][k].clone()
        return st

    def deploy_all():
        """Deployed states of every client; FedBABU fine-tunes locally."""
        out = []
        for ci, client in enumerate(clients):
            st = deployed_state(ci)
            if strategy == "fedbabu":
                m = build()
                m.load_state_dict(st)
                m, _ = local_finetune(m, train_loaders[ci],
                                      client.loader("val", cfg.batch_size),
                                      device, cfg.ft_epochs, cfg.lr)
                st = {k: v.detach().cpu().clone()
                      for k, v in m.state_dict().items()}
            out.append(st)
        return out

    def evaluate_all(states, keep_preds):
        res, preds = {}, {}
        for ci, client in enumerate(clients):
            m = build()
            m.load_state_dict(states[ci])
            metrics, (pt, yt, _), (pv, yv) = evaluate(
                m, client.loader("val", cfg.batch_size),
                client.loader("test", cfg.batch_size), device,
                return_preds=True)
            res[client.name] = metrics
            if keep_preds:
                preds[client.name] = (pv, yv, pt, yt)
        return summarize_clients(res), preds

    def history_val_auroc(ci, client):
        st = deployed_state(ci)
        m = build()
        m.load_state_dict(st)
        if strategy == "fedbabu":
            # cheap proxy for FedBABU's fine-tuned deployment: refit only the
            # (frozen at init) head on cached features of the current body
            feats, ys = cache_features(m, train_loaders[ci], device, passes=1)
            train_head_on_features(m.head, feats, ys, device,
                                   epochs=cfg.rep_head_epochs, lr=cfg.lr)
        pv, yv = predict(m, client.loader("val", cfg.batch_size), device)
        return macro_auroc(yv, pv)

    for rnd in range(start_round, rounds):
        part = schedule[rnd]
        # round-start global parameters: prox target for fedprox AND ditto
        g0 = [p.detach().clone() for p in global_model.parameters()]
        for ci in part:
            client = clients[ci]
            model = build()
            st = copy.deepcopy(global_model.state_dict())
            for k in private:
                st[k] = local_states[ci][k]
            if is_local:
                st = local_states[ci]
            model.load_state_dict(st)

            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            if dp:
                train_model_dp(model, train_loaders[ci], device, local_epochs,
                               cfg.lr, sigmas[ci], cfg.dp_max_grad_norm,
                               accountants[ci])
            elif strategy == "fedrep":
                # head phase: body frozen, head fit on cached features
                feats, ys = cache_features(model, train_loaders[ci], device,
                                           passes=cfg.rep_feature_passes)
                train_head_on_features(model.head, feats, ys, device,
                                       epochs=cfg.rep_head_epochs, lr=cfg.lr)
                # body phase: head frozen, local epoch(s) on the representation
                for p in model.head.parameters():
                    p.requires_grad_(False)
                train_model(model, train_loaders[ci], device,
                            epochs=local_epochs, lr=cfg.lr,
                            eval_modules=[model.head])
                for p in model.head.parameters():
                    p.requires_grad_(True)
            elif strategy == "fedbabu":
                # body-only FL: the head stays frozen at its shared random init
                for p in model.head.parameters():
                    p.requires_grad_(False)
                train_model(model, train_loaders[ci], device,
                            epochs=local_epochs, lr=cfg.lr,
                            eval_modules=[model.head])
                for p in model.head.parameters():
                    p.requires_grad_(True)
            else:
                train_model(model, train_loaders[ci], device,
                            epochs=local_epochs, lr=cfg.lr,
                            prox_mu=cfg.prox_mu if strategy == "fedprox" else 0.0,
                            global_params=g0 if strategy == "fedprox" else None)

            if strategy == "ditto":  # personal model, prox to round-start global
                pm = build()
                pm.load_state_dict(personal[ci])
                train_model(pm, train_loaders[ci], device, epochs=local_epochs,
                            lr=cfg.lr, prox_mu=cfg.ditto_lambda, global_params=g0)
                personal[ci] = {k: v.cpu() for k, v in pm.state_dict().items()}
            _sync(device)
            timing[client.name].append(time.perf_counter() - t0)
            if device.type == "cuda":
                peak_mem[client.name] = max(
                    peak_mem[client.name],
                    torch.cuda.max_memory_allocated() / 2 ** 20)
            elif device.type == "mps":  # no peak counter on MPS: driver total
                peak_mem[client.name] = max(
                    peak_mem[client.name],
                    torch.mps.driver_allocated_memory() / 2 ** 20)

            local_states[ci] = {k: v.cpu() for k, v in model.state_dict().items()}

        if not is_local:
            t0 = time.perf_counter()
            avg = _avg_states([local_states[ci] for ci in part],
                              [weights[ci] for ci in part], shared_keys)
            gst = global_model.state_dict()
            if strategy == "fedavgm":
                if momentum is None:
                    momentum = {k: torch.zeros_like(v.float()) for k, v in avg.items()}
                for k in avg:
                    # Only momentum-update learnable params. Applying momentum
                    # to BN buffers (running_var) can drive variance negative
                    # -> sqrt(<0) at eval -> NaN; average those directly like
                    # FedAvg does. (num_batches_tracked is non-float, also here.)
                    if k not in param_keys:
                        gst[k] = avg[k]
                        continue
                    delta = gst[k].float() - avg[k].float()
                    momentum[k] = cfg.server_momentum * momentum[k] + delta
                    # Server learning rate damps the momentum step. Without it
                    # the steady-state update is delta/(1-momentum) ~= 10x a
                    # FedAvg step, which diverges. server_lr=(1-momentum) makes
                    # the steady-state step equal a vanilla FedAvg step.
                    gst[k] = (gst[k].float()
                              - cfg.server_lr * momentum[k]).to(gst[k].dtype)
            else:
                for k in avg:
                    gst[k] = avg[k]
            global_model.load_state_dict(gst)
            agg_time.append(time.perf_counter() - t0)

        # traffic: each participant uploads its shared tensors and receives
        # the broadcast of the new shared tensors (same size, both FP32)
        up = down = len(part) * comm_mb
        cum_up += up
        cum_down += down
        comm_log.append({"round": rnd + 1, "participants": len(part),
                         "upload_mb": up, "download_mb": down,
                         "cum_upload_mb": cum_up, "cum_download_mb": cum_down})
        # (federation totals; per-hospital values are comm_mb per round)

        # periodic validation AUROC -> convergence / comm-budget curves
        if cfg.eval_every and ((rnd + 1) % cfg.eval_every == 0 or rnd == rounds - 1):
            entry = {"round": rnd + 1, "cum_upload_mb": cum_up,
                     "cum_download_mb": cum_down, "participants": part}
            for ci, client in enumerate(clients):
                entry[client.name] = history_val_auroc(ci, client)
            history.append(entry)

        # round-budget snapshots (test-evaluated deployed models)
        if rnd + 1 in snap_rounds:
            states = deploy_all()
            summ, preds = evaluate_all(states, keep_preds=(rnd + 1 == main_round))
            snapshots[str(rnd + 1)] = summ
            log.info(f"[{strategy}] snapshot r{rnd + 1}: "
                     f"MEAN={summ['MEAN']['auroc']:.4f} "
                     f"WORST={summ['WORST']['auroc']:.4f}")
            if rnd + 1 == main_round:
                main_stash = {
                    "summary": summ, "preds": preds,
                    "deployed": {c.name: states[ci]
                                 for ci, c in enumerate(clients)},
                    "global_state": copy.deepcopy(global_model.state_dict())}
        log.info(f"[{strategy}] round {rnd + 1}/{rounds} done "
                 f"(participants {[clients[i].name for i in part]})")

        if ckpt_path:
            torch.save({
                "round": rnd + 1,
                "global_state": global_model.state_dict(),
                "local_states": local_states,
                "personal": personal,
                "momentum": momentum,
                "history": history,
                "extra": {"timing": timing, "peak_mem": peak_mem,
                          "agg_time": agg_time, "comm_log": comm_log,
                          "snapshots": snapshots, "main_stash": main_stash,
                          "cum": (cum_up, cum_down),
                          "acc_history": ([a.history for a in accountants]
                                          if dp else None)},
            }, ckpt_path)

    # The main-round deployed models are the result of this run.
    summary = dict(main_stash["summary"])
    deployed = main_stash["deployed"]
    preds = main_stash["preds"]
    global_model.load_state_dict(main_stash["global_state"])
    for name, m in summary.items():
        if name not in ("MEAN", "WORST"):
            log.info(f"[{strategy}] {name}: AUROC={m['auroc']:.4f} F1={m['f1']:.4f}")

    summary["model_params"] = n_params
    summary["snapshots"] = snapshots
    summary["main_round"] = main_round
    summary["weights"] = {c.name: w / sum(weights) for c, w in zip(clients, weights)}
    summary["participation"] = {"fraction": cfg.participation,
                                "dropout": cfg.dropout,
                                "rounds_participated": {
                                    c.name: sum(ci in p for p in schedule[:main_round])
                                    for ci, c in enumerate(clients)}}
    upto = [e for e in comm_log if e["round"] <= main_round]
    summary["comm"] = {
        "precision": "fp32",
        "per_client_round_upload_mb": comm_mb,
        "per_client_round_download_mb": comm_mb,
        "per_client_round_mb_fp16": comm_mb_fp16,
        "shared_params": int(sum(global_model.state_dict()[k].numel()
                                 for k in shared_keys)) if not is_local else 0,
        "rounds": main_round,
        # whole federation (sum over participating hospitals)
        "cum_upload_mb_total": upto[-1]["cum_upload_mb"] if upto else 0.0,
        "cum_download_mb_total": upto[-1]["cum_download_mb"] if upto else 0.0,
        # one hospital, averaged over hospitals (differs from rounds x per
        # round only under partial participation / dropout)
        "cum_upload_mb_per_client": comm_mb * float(np.mean(
            [sum(ci in p for p in schedule[:main_round])
             for ci in range(len(clients))])) if not is_local else 0.0,
        "log": comm_log,
    }
    summary["timing"] = {
        "device": _device_name(device),
        "local_train_s_mean": {k: float(np.mean(v)) if v else None
                               for k, v in timing.items()},
        "local_train_s_median": {k: float(np.median(v)) if v else None
                                 for k, v in timing.items()},
        "peak_mem_mib": peak_mem,
        "aggregation_s_mean": float(np.mean(agg_time)) if agg_time else 0.0,
        "train_sizes": {c.name: int(len(c.train_idx)) for c in clients},
    }
    if dp:
        summary["dp"] = {"target_epsilon": cfg.dp_epsilon, "delta": cfg.dp_delta,
                         "max_grad_norm": cfg.dp_max_grad_norm,
                         "noise_multiplier": {c.name: s for c, s in zip(clients, sigmas)},
                         "achieved_epsilon": {
                             c.name: float(a.get_epsilon(cfg.dp_delta))
                             for c, a in zip(clients, accountants)}}
    return summary, history, comm_mb, global_model, deployed, preds
