"""Datasets over the preprocessed caches.

Training recipe follows the PTB-XL benchmarking setup that tops the public
leaderboard: random 2.5 s crops during training, sliding-window aggregation
at evaluation time (handled in the trainer).
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CROP_LEN = 250  # 2.5 s @ 100 Hz


def default_workers(requested=2):
    """DataLoader workers. FCT_WORKERS overrides. On macOS workers are
    spawned, which pickles the memory-mapped caches into every worker (the
    Ningbo cache alone is 0.8 GB), so the default there is 0 (in-process
    loading, which is fast enough next to an Apple GPU)."""
    if "FCT_WORKERS" in os.environ:
        return int(os.environ["FCT_WORKERS"])
    import sys
    return 0 if sys.platform == "darwin" else requested


class ECGDataset(Dataset):
    def __init__(self, X, Y, indices, train: bool, crop_len: int = CROP_LEN,
                 augment: bool = True, leads=None):
        self.X, self.Y = X, Y
        self.indices = np.asarray(indices)
        self.train = train
        self.crop_len = crop_len
        self.leads = None if leads is None else list(leads)
        self.augment = None
        if train and augment:
            from .augment import ECGAugment
            n_leads = 12 if self.leads is None else len(self.leads)
            self.augment = ECGAugment(p=0.5, allow_lead_dropout=n_leads > 3)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x = self.X[idx].astype(np.float32)
        if self.leads is not None:
            x = x[self.leads]
        y = self.Y[idx].astype(np.float32)
        if self.train:
            start = np.random.randint(0, x.shape[-1] - self.crop_len + 1)
            x = x[:, start:start + self.crop_len]
            if self.augment is not None:
                x = self.augment(x)
        return torch.from_numpy(x.copy()), torch.from_numpy(y)


class ClientData:
    """One hospital client: memmapped cache + deterministic 70/10/20 split.

    exclude:     cache rows withheld from this client entirely (e.g. the
                 PTB-XL patients reserved for the federated twin streams).
    max_records: subsample (smoke tests only).
    """

    def __init__(self, cache_dir, name, seed=0, leads=None, exclude=None,
                 max_records=None):
        d = os.path.join(cache_dir, name)
        self.name = name
        self.leads = leads
        self.X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
        self.Y = np.load(os.path.join(d, "Y.npy"))
        n = len(self.Y)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        if exclude is not None and len(exclude):
            perm = perm[~np.isin(perm, np.asarray(list(exclude)))]
        if max_records:
            perm = perm[:max_records]
        m = len(perm)
        n_tr, n_va = int(0.7 * m), int(0.1 * m)
        self.train_idx = perm[:n_tr]
        self.val_idx = perm[n_tr:n_tr + n_va]
        self.test_idx = perm[n_tr + n_va:]

    @property
    def n_leads(self):
        return 12 if self.leads is None else len(self.leads)

    def loader(self, split, batch_size, num_workers=2, indices=None,
               shuffle=None):
        """split in {train, val, test}; `indices` overrides the split's rows
        (e.g. an onboarding subset) while keeping its train/eval transform."""
        from torch.utils.data import DataLoader
        num_workers = default_workers(num_workers)
        idx = {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[split]
        if indices is not None:
            idx = np.asarray(indices)
        train = split == "train"
        shuffle = train if shuffle is None else shuffle
        return DataLoader(ECGDataset(self.X, self.Y, idx, train=train,
                                     leads=self.leads),
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(),
                          drop_last=train and len(idx) > batch_size)

    def label_counts(self):
        """Per-class positive counts on the training split (FedLA weights)."""
        return self.Y[self.train_idx].sum(axis=0).astype(np.float64)

    def __len__(self):
        return len(self.Y)


def load_clients(cache_dir, names, seed=0, leads=None, exclude=None,
                 max_records=None):
    """exclude: {client_name: rows to withhold}."""
    out = []
    for n in names:
        if os.path.exists(os.path.join(cache_dir, n, "X.npy")):
            out.append(ClientData(cache_dir, n, seed=seed, leads=leads,
                                  exclude=(exclude or {}).get(n),
                                  max_records=max_records))
    if not out:
        raise FileNotFoundError(f"No client caches found in {cache_dir}")
    return out


def ptbxl_db_path(cache_dir):
    """ptbxl_database.csv (PhysioNet PTB-XL metadata) next to the caches."""
    for p in (os.path.join(cache_dir, "ptbxl_database.csv"),
              os.path.join(cache_dir, "PTBXL_TRACKB", "ptbxl_database.csv")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "ptbxl_database.csv not found next to the caches. Download it from "
        "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv")


def ptbxl_client_meta(cache_dir):
    """Patient metadata for every row of the CinC PTB-XL client cache.

    CinC 2021 names PTB-XL records HR<ecg_id>; joining on ecg_id recovers
    patient_id, recording date and the official stratified fold. (Verified
    by signal correlation against the original PTB-XL waveforms.)"""
    man = pd.read_csv(os.path.join(cache_dir, "PTB-XL", "manifest.csv"))
    man["ecg_id"] = man.record.str.replace("HR", "", regex=False).astype(int)
    db = pd.read_csv(ptbxl_db_path(cache_dir), index_col="ecg_id",
                     usecols=["ecg_id", "patient_id", "recording_date",
                              "strat_fold"])
    meta = man.join(db, on="ecg_id")
    meta["row"] = np.arange(len(meta))
    return meta


def twin_heldout_rows(cache_dir, folds=(9, 10)):
    """PTB-XL client rows of every patient in the held-out folds. These are
    withheld from federated training so the federated twin is evaluated on
    patients the federation never saw."""
    meta = ptbxl_client_meta(cache_dir)
    return meta[meta.strat_fold.isin(list(folds))].row.values


class CinCPTBXLStreams:
    """Longitudinal streams over the CinC PTB-XL client (26 scored classes),
    for the twin built on the deployed federated model."""

    def __init__(self, cache_dir, folds=(9, 10), min_records=2):
        d = os.path.join(cache_dir, "PTB-XL")
        self.X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
        self.Y = np.load(os.path.join(d, "Y.npy"))
        self.meta = ptbxl_client_meta(cache_dir)
        self.folds, self.min_records = folds, min_records

    def eval_streams(self):
        m = self.meta[self.meta.strat_fold.isin(list(self.folds))]
        streams = []
        for pid, g in m.groupby("patient_id"):
            if len(g) >= self.min_records:
                g = g.sort_values("recording_date")
                streams.append((int(pid), g.row.values))
        return streams


class PTBXLStreams:
    """Track B: chronological per-patient record streams for the twin loop.

    Patients in folds 1-8 train the base model; multi-record patients in
    folds 9-10 are the held-out longitudinal evaluation streams.
    """

    def __init__(self, cache_dir):
        d = os.path.join(cache_dir, "PTBXL_TRACKB")
        self.X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
        self.Y = np.load(os.path.join(d, "Y.npy"))
        self.meta = pd.read_csv(os.path.join(d, "meta.csv"))
        self.meta["row"] = np.arange(len(self.meta))

    def base_indices(self, folds=range(1, 9)):
        m = self.meta[self.meta.strat_fold.isin(list(folds))]
        return m.row.values

    def eval_streams(self, folds=(9, 10), min_records=2):
        m = self.meta[self.meta.strat_fold.isin(list(folds))]
        streams = []
        for pid, g in m.groupby("patient_id"):
            if len(g) >= min_records:
                g = g.sort_values("date")
                streams.append((int(pid), g.row.values))
        return streams
