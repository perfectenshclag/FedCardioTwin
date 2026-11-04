import os
import json
import random
import logging

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    forced = os.environ.get("FCT_DEVICE")  # e.g. FCT_DEVICE=cpu for smoke tests
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = "fedcardiotwin"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def count_bytes(state_dict, keys=None):
    """Bytes required to transmit `keys` (default: all) of a state dict.
    Used for the communication-cost accounting reported in the paper."""
    total = 0
    for k, v in state_dict.items():
        if keys is not None and k not in keys:
            continue
        total += v.numel() * v.element_size()
    return total


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)
