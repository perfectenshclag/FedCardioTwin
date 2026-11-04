"""ECG-specific training augmentations (applied on (12, L) numpy arrays).

These are the transforms with consistent gains in the 12-lead ECG
literature: gaussian noise, per-lead amplitude scaling, time masking,
baseline wander, and lead dropout. All cheap, all label-preserving.
"""
import numpy as np


def gaussian_noise(x, rng, max_sigma=0.05):
    return x + rng.uniform(0, max_sigma) * rng.standard_normal(x.shape).astype(x.dtype)


def amplitude_scale(x, rng, lo=0.8, hi=1.2):
    return x * rng.uniform(lo, hi, size=(x.shape[0], 1)).astype(x.dtype)


def time_mask(x, rng, max_frac=0.1):
    L = x.shape[-1]
    w = int(rng.uniform(0, max_frac) * L)
    if w > 0:
        s = rng.randint(0, L - w + 1)
        x = x.copy()
        x[:, s:s + w] = 0.0
    return x


def baseline_wander(x, rng, fs=100, max_amp=0.3):
    L = x.shape[-1]
    t = np.arange(L) / fs
    f = rng.uniform(0.05, 0.3)
    phase = rng.uniform(0, 2 * np.pi)
    wander = (rng.uniform(0, max_amp) * np.sin(2 * np.pi * f * t + phase)).astype(x.dtype)
    return x + wander[None, :]


def lead_dropout(x, rng, p=0.2, max_leads=2):
    if rng.uniform() < p:
        x = x.copy()
        k = rng.randint(1, max_leads + 1)
        x[rng.choice(x.shape[0], size=k, replace=False)] = 0.0
    return x


_TRANSFORMS = (gaussian_noise, amplitude_scale, time_mask, baseline_wander,
               lead_dropout)


class ECGAugment:
    """Each transform fires independently with probability `p`.
    lead_dropout is skipped for reduced-lead inputs (<= 3 leads), where
    zeroing up to two leads could blank the whole signal."""

    def __init__(self, p=0.5, seed=None, allow_lead_dropout=True):
        self.p = p
        self.rng = np.random.RandomState(seed)
        self.transforms = tuple(t for t in _TRANSFORMS
                                if allow_lead_dropout or t is not lead_dropout)

    def __call__(self, x):
        for t in self.transforms:
            if self.rng.uniform() < self.p:
                x = t(x, self.rng)
        return np.ascontiguousarray(x)
