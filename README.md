# FedCardioTwin

**Continually Updated, Uncertainty-Aware Federated Digital Twins for
Patient-Specific Cardiac Monitoring Across Heterogeneous Hospitals.**

Target venue: IEEE J-BHI special issue *"Federated Learning and Digital
Twins for Smart Healthcare"* (submission 30 June 2026).

## What this is

- **Track A (federated):** 6 real institutions from the PhysioNet/CinC 2021
  corpus (CPSC, CPSC-Extra, Georgia, Chapman, Ningbo, PTB-XL) as FL clients —
  real inter-hospital non-IID, 26 official scored classes, multilabel.
- **Track B (twin):** original PTB-XL multi-record patients replayed
  chronologically; per-patient adapter twins run predict → compare → alert →
  update, with replay-based forgetting prevention.
- **Conformal layer:** federated conformal risk control gives per-hospital
  prediction sets with FNR ≤ α guarantees.

All data is public; no hardware; everything runs on a single T4.

## Layout

```
fedcardiotwin/
  data/        labels (parsed from official CSVs), preprocessing, datasets
  models/      Inception1d (primary), ResNet1d, patient twin adapters
  train/       trainer (AMP + OneCycle + sliding-window eval), metrics
  fl/          serial FL engine: fedavg/fedprox/fedbn/fedavgm/ditto/fedper/local
  twin/        the digital-twin loop (Track B)
  conformal/   federated conformal risk control
  configs.py   fast/full presets
scripts/       build_cache, run_experiments, make_tables
tests/         smoke_test.py — full pipeline on synthetic data, no download
notebooks/     FedCardioTwin_T4.ipynb — end-to-end on Kaggle/Colab
external/      cloned reference repos (helme benchmark, official CinC eval)
```

## Quickstart

```bash
pip install -r requirements.txt
python tests/smoke_test.py                      # verify install, no data needed

# data (one-time, ~15 GB raw):
wget -r -N -c -np -P data/raw/ptbxl_dl https://physionet.org/files/ptb-xl/1.0.3/
wget -r -N -c -np -P data/raw/cinc_dl  https://physionet.org/files/challenge-2021/1.0.3/training/
# (see notebook for the exact post-download directory arrangement)

python scripts/build_cache.py --track a --raw-dir data/raw/cinc2021
python scripts/build_cache.py --track b --ptbxl-dir data/raw/ptbxl

# experiments (fast preset first, then full):
python scripts/run_experiments.py --stage centralized --preset fast
python scripts/run_experiments.py --stage federated  --preset full
python scripts/run_experiments.py --stage twin       --preset full
python scripts/run_experiments.py --stage conformal
python scripts/make_tables.py
```

## J-BHI revision (JBHI-04805-2026): reproducing every revised table

All revision features default to the submitted protocol, so the commands
above still reproduce the original setup. Revision runs (Kaggle, 2 x T4;
`notebooks/FedCardioTwin_Revision_Kaggle.ipynb` runs them unattended in four
batches A1, A2, B, C):

| Revised result | Command (add `--cache-dir/--results-dir/--ckpt-dir`) |
|---|---|
| Table III (round 60) + round-budget curves | `--stage federated --strategies <s> --rounds 120 --main-round 60 --snapshot-rounds 20 40 60 80 100 120 --eval-every 5` for every strategy incl. `fedrep fedbabu`; `--stage centralized`; `--stage finetune` (FedAvg+FT, Centralized+FT); `scripts/ensemble_eval.py --fed-clients ...` |
| Personalization depth | `--strategies fedper --local-blocks 1 --tag _depth1` (and `2`) |
| Aggregation weights | `--weighting equal --tag _eqw`, `--weighting label --tag _fedla` |
| Partial participation / dropout | `--participation 0.5 --tag _part50`, `--dropout 0.3 --tag _drop30` |
| Reduced leads | `--leads 6|3|2|1 --tag _leads<k>` |
| Differential privacy | `--norm gn --tag _gn`, `--dp-epsilon 8|3|1 --tag _dp<k>` (needs `opacus==1.5.4`) |
| LOHO + onboarding curve | `--stage loho` |
| Twin (Track B, full logging) / federated twin | `--stage twin`; `--strategies fedper --twin-split --tag _twinfed` then `--stage twin_fed` |
| Measured compute | `scripts/profile_compute.py` |
| All post-hoc analyses (CPU) | `scripts/posthoc.py` (per-class, key classes, bootstrap, repeated-split conformal, UCB rule, class-wise/weighted risk, calibration-size curves, alert baselines, comm/timing, stopping rule, PTB-XL patient-disjoint check) |
| Result figures | `scripts/make_revision_figures.py --results-dir results --out-dir figures` |

Local CPU tests: `FCT_DEVICE=cpu FCT_WORKERS=0 python scripts/run_experiments.py ... --preset fast --max-records 200`.

## Experiment matrix (the paper's tables)

| Table | Command | What it shows |
|---|---|---|
| Main comparison | `--stage federated` (local/fedavg/fedprox/fedbn/fedavgm/ditto/fedper × 3 seeds) | per-client + worst-client AUROC/F1 under real non-IID |
| Upper bound | `--stage centralized` | pooled-data ceiling |
| Headline | `scripts/ensemble_eval.py` | 3-seed ensemble |
| Generalization | `--stage loho` | leave-one-hospital-out: deploy on an unseen institution |
| Twin gain | `--stage twin` | cold vs warm AUROC, per-step gain curve, alert-detection AUROC, forgetting (+ raw `.npz` arrays) |
| Uncertainty | `--stage conformal` | FNR coverage + set size per hospital, on deployed per-client models |
| Communication | `history` field in federated results | AUROC vs rounds / vs cumulative upload MB |
| Significance | `scripts/make_tables.py` | mean ± std + Wilcoxon signed-rank vs FedAvg |

## Model recipe (best-results configuration)

Primary model: **SE-InceptionTime-1d** — InceptionTime (bottleneck 32,
kernels 39/19/9, depth 6, residual every 3) with squeeze-and-excitation
channel attention per block. Training: AdamW + OneCycle, AMP, random 2.5 s
crops, **ECG augmentation** (noise, amplitude scale, time mask, baseline
wander, lead dropout), **mixup** (alpha=0.2), **EMA weights** (0.999).
Evaluation: sliding-window probability averaging (stride 125) +
per-class thresholds tuned on validation. Final headline number:
**seed ensemble** via `scripts/ensemble_eval.py` (average sigmoid probs of
the 3 seed checkpoints; add resnet1d checkpoints for an architecture
ensemble). Each component is independently ablatable (`se=False`,
`augment=False`, `mixup_alpha=0`, `ema_decay=0`) for the ablation table.

## Notes

- Label space is parsed at runtime from the official challenge
  `weights.csv` / `dx_mapping_scored.csv` (cloned in `external/`) — no
  hardcoded medical codes anywhere.
- The FL engine is deliberately serial (one GPU, exact reproducibility);
  this matches standard cross-silo FL simulation practice.
