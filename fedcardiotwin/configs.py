"""Experiment configuration. Two presets:
FAST  — pipeline sanity on a subset (minutes on a T4).
FULL  — the paper runs (overnight-class on a T4).

Every field added for the J-BHI revision defaults to the submitted protocol,
so a run with default settings reproduces the original experiments.
"""
from dataclasses import dataclass, field


@dataclass
class FLConfig:
    rounds: int = 60
    local_epochs: int = 1
    local_only_epochs: int = 20      # for the local-only baseline
    batch_size: int = 128
    lr: float = 1e-3
    prox_mu: float = 0.01            # fedprox
    ditto_lambda: float = 0.1        # ditto prox strength
    server_momentum: float = 0.9     # fedavgm
    server_lr: float = 0.1           # fedavgm server step damping (~1-momentum)
    eval_every: int = 10             # rounds between val-AUROC history points
                                     # (0 disables; last round always logged)
    # --- revision additions (defaults = submitted protocol) ---
    weighting: str = "samples"       # aggregation weights: samples | equal |
                                     # label (FedLA: per-label client shares)
    participation: float = 1.0       # fraction of clients sampled per round
    dropout: float = 0.0             # P(sampled client unavailable this round)
    local_blocks: int = 0            # fedper/fedrep: extra trailing Inception
                                     # blocks kept private (personalization depth)
    rep_head_epochs: int = 10        # fedrep: head epochs per round (frozen body)
    rep_feature_passes: int = 2      # fedrep: random crops per record cached
    ft_epochs: int = 5               # local fine-tuning epochs (fedbabu, *+FT)
    ft_lr: float = 1e-4              # local fine-tuning peak learning rate
    main_round: int = 0              # round whose deployed models are THE result
                                     # (0 = last round)
    snapshot_rounds: tuple = ()      # rounds at which deployed models are
                                     # val/test evaluated (round-budget curves)
    dp_epsilon: float = 0.0          # >0: record-level DP-SGD at every client
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0


@dataclass
class CentralConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    mixup_alpha: float = 0.2     # mixup on signals+labels (multilabel-safe)
    ema_decay: float = 0.999     # EMA weights are the evaluated weights
    augment: bool = True         # ECG augmentation pipeline (train only)


@dataclass
class TwinConfig:
    adapter_hidden: int = 64
    replay_size: int = 16        # larger buffer + reservoir keeps early records
    update_steps: int = 3        # fewer, gentler updates
    update_lr: float = 2e-4      # conservative LR (was 1e-3 -> caused drift)
    prox_weight: float = 1.0     # anchor twin logits to the global model
    guard: bool = True           # deploy the adapted twin only when it beats
                                  # the global model on the patient's own
                                  # already-seen records


@dataclass
class ConformalConfig:
    alpha: float = 0.1
    cal_alpha: float = 0.05  # conservative per-hospital calibration target:
                             # leaves finite-sample headroom so small sites
                             # (e.g. CPSC-Extra) still test under alpha
    n_splits: int = 100      # random calibration/test re-splits per seed
    ucb_delta: float = 0.1   # confidence level of the size-adaptive UCB rule


@dataclass
class ExperimentConfig:
    clients: tuple = ("CPSC", "CPSC-Extra", "Georgia", "Chapman", "Ningbo", "PTB-XL")
    model: str = "inception1d"
    seeds: tuple = (0, 1, 2)
    leads: tuple = None          # None = all 12 leads; else lead indices
    fl: FLConfig = field(default_factory=FLConfig)
    central: CentralConfig = field(default_factory=CentralConfig)
    twin: TwinConfig = field(default_factory=TwinConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)


# Standard lead order I, II, III, aVR, aVL, aVF, V1..V6 (CinC 2021 headers).
LEAD_SETS = {
    "12": tuple(range(12)),
    "6": (0, 1, 2, 3, 4, 5),        # limb leads (CinC 2021 6-lead set)
    "4": (0, 1, 2, 7),              # I, II, III, V2
    "3": (0, 1, 7),                 # I, II, V2
    "2": (0, 1),                    # I, II
    "1": (0,),                      # lead I (wearable single lead)
}


def fast_preset():
    cfg = ExperimentConfig(seeds=(0,))
    cfg.fl.rounds = 3
    cfg.fl.local_only_epochs = 2
    cfg.central.epochs = 2
    return cfg


def full_preset():
    return ExperimentConfig()
