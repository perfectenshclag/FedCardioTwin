"""Experiment batches of the J-BHI revision: single source of truth for the
Kaggle notebook and for scripts/run_batch.py (local GPU / Apple MPS).

Each batch is a list of queues; a queue is a list of jobs run one after the
other, and the queues of a batch run in parallel (one per GPU on Kaggle,
two processes on one Apple GPU). Every job skips runs whose result file
already exists, so a batch can be restarted at any time.

  A1  main table (part 1): centralized, local, FedAvg, FedPer, FedRep,
      FedBABU (120 rounds, Table III = round 60), fine-tuning baselines,
      FedPer ensemble
  A2  main table (part 2): FedProx, FedBN, FedAvgM, Ditto (120 rounds)
  B   sensitivity analyses + their reference runs on the same machine
  C1  leave-one-hospital-out + onboarding, twin (Track B), federated twin
  C2  baseline hyperparameter sensitivity + differential privacy (CUDA only:
      DP-SGD is ~12x slower on Apple GPUs)
  C   C1 and C2 together (Kaggle: one queue each)

A1, A2, B and C2 must run on the same kind of hardware (they are compared
with each other); C1 is self-contained and can run anywhere.
"""

MAIN = "--rounds 120 --main-round 60 --snapshot-rounds 20 40 60 80 100 120 --eval-every 5"


def make_batches(py, cache, results, ckpt, repo_scripts="scripts"):
    from shlex import quote  # paths may contain spaces (e.g. external drives)
    py, cache, results, ckpt = map(quote, (py, cache, results, ckpt))
    base = (f"{py} -u {repo_scripts}/run_experiments.py --cache-dir {cache} "
            f"--results-dir {results} --ckpt-dir {ckpt}")

    def fed(strats, seeds="0 1 2", extra=""):
        return f"{base} --stage federated --strategies {strats} --seeds {seeds} {extra}"

    def J(name, cmd, needs=()):
        return (name, cmd, list(needs))

    ensemble = (f"{py} -u {repo_scripts}/ensemble_eval.py --fed-clients "
                + " ".join(quote(f"{ckpt}/fed_fedper_seed{s}_clients.pt".replace("'", ""))
                           for s in range(3))
                + f" --out-name ensemble_fedper --cache-dir {cache} --results-dir {results}")

    c1 = [
        [J("loho", f"{base} --stage loho")],
        [J("twin", f"{base} --stage twin"),
         J("twinfed_fl", fed("fedper", extra="--twin-split --tag _twinfed")),
         J("twinfed", f"{base} --stage twin_fed")],
    ]
    c2 = [
        [J("hp_mu0.001", fed("fedprox", seeds="0", extra="--prox-mu 0.001 --tag _mu0.001")),
         J("hp_mu0.1", fed("fedprox", seeds="0", extra="--prox-mu 0.1 --tag _mu0.1")),
         J("hp_lam0.01", fed("ditto", seeds="0", extra="--ditto-lambda 0.01 --tag _lam0.01")),
         J("hp_lam1", fed("ditto", seeds="0", extra="--ditto-lambda 1.0 --tag _lam1"))],
        [J("gn", fed("fedper", seeds="0", extra="--norm gn --tag _gn")),
         J("dp8", fed("fedper", seeds="0", extra="--dp-epsilon 8 --tag _dp8")),
         J("dp3", fed("fedper", seeds="0", extra="--dp-epsilon 3 --tag _dp3")),
         J("dp1", fed("fedper", seeds="0", extra="--dp-epsilon 1 --tag _dp1"))],
    ]
    return {
        "A1": [
            [J("central", f"{base} --stage centralized"),
             J("fedavg", fed("fedavg", extra=MAIN)),
             J("fedper", fed("fedper", extra=MAIN)),
             J("finetune", f"{base} --stage finetune"),
             J("ensemble", ensemble),
             J("fedbabu_s2", fed("fedbabu", seeds="2", extra=MAIN))],
            [J("local", fed("local")),
             J("fedrep", fed("fedrep", extra=MAIN)),
             J("fedbabu_s01", fed("fedbabu", seeds="0 1", extra=MAIN))]],
        "A2": [
            [J("ditto", fed("ditto", extra=MAIN)),
             J("fedavgm_s2", fed("fedavgm", seeds="2", extra=MAIN))],
            [J("fedprox", fed("fedprox", extra=MAIN)),
             J("fedbn", fed("fedbn", extra=MAIN)),
             J("fedavgm_s01", fed("fedavgm", seeds="0 1", extra=MAIN))]],
        "B": [
            [J("ref", fed("fedavg fedper", extra="--tag _ref")),
             J("depth1", fed("fedper", extra="--local-blocks 1 --tag _depth1")),
             J("depth2", fed("fedper", extra="--local-blocks 2 --tag _depth2")),
             J("eqw", fed("fedavg fedper", extra="--weighting equal --tag _eqw")),
             J("fedla", fed("fedavg fedper", extra="--weighting label --tag _fedla"))],
            [J("part50", fed("fedavg fedper", extra="--participation 0.5 --tag _part50")),
             J("drop30", fed("fedavg fedper", extra="--dropout 0.3 --tag _drop30")),
             J("leads6", fed("fedper", extra="--leads 6 --tag _leads6")),
             J("leads3", fed("fedper", extra="--leads 3 --tag _leads3")),
             J("leads2", fed("fedper", extra="--leads 2 --tag _leads2")),
             J("leads1", fed("fedper", extra="--leads 1 --tag _leads1"))]],
        "C1": c1,
        "C2": c2,
        "C": [c1[0] + c1[1] + c2[0], c2[1]],
        # Main table on the same machine as batch B (60 rounds, tag _ref, so
        # it extends the reference FedAvg/FedPer runs of batch B)
        "A_M5": [
            [J("central_ref", f"{base} --stage centralized --tag _ref"),
             J("fedrep_ref", fed("fedrep", extra="--tag _ref")),
             J("finetune_ref", f"{base} --stage finetune --tag _ref"),
             J("fedprox_ref", fed("fedprox", extra="--tag _ref")),
             J("ditto_ref_s0", fed("ditto", seeds="0", extra="--tag _ref"))],
            [J("local_ref", fed("local", extra="--tag _ref")),
             J("fedbabu_ref", fed("fedbabu", extra="--tag _ref")),
             J("fedbn_ref", fed("fedbn", extra="--tag _ref")),
             J("fedavgm_ref", fed("fedavgm", extra="--tag _ref")),
             J("ditto_ref_s12", fed("ditto", seeds="1 2", extra="--tag _ref"))]],
        # Round budget: 120 rounds with test snapshots every 20 rounds
        "R_M5": [
            [J("r120_fedper", fed("fedper", seeds="0", extra=MAIN + " --tag _r120")),
             J("r120_fedrep", fed("fedrep", seeds="0", extra=MAIN + " --tag _r120"))],
            [J("r120_fedavg", fed("fedavg", seeds="0", extra=MAIN + " --tag _r120"))]],
    }
