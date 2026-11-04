"""Training/eval loops shared by centralized, federated, and twin code paths.

Best-results recipe: AdamW + OneCycle, mixed precision on CUDA, random-crop
training, sliding-window (stride 125) probability averaging at eval.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from ..data.dataset import CROP_LEN
from .metrics import macro_auroc, macro_f1, tune_thresholds

EVAL_STRIDE = 125


def _amp_ctx(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    import contextlib
    return contextlib.nullcontext()


class ModelEma:
    """Exponential moving average of weights; evaluated weights are the EMA.
    A consistently free accuracy gain on long centralized runs."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model):
        st = model.state_dict()
        for k in st:
            st[k] = self.shadow[k].to(st[k].dtype)
        model.load_state_dict(st)


def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]


def train_model(model, loader, device, epochs, lr=1e-3, weight_decay=1e-4,
                prox_mu=0.0, global_params=None, max_steps=None, log_fn=None,
                mixup_alpha=0.0, ema_decay=0.0, ckpt_path=None, ckpt_every=1,
                eval_modules=None):
    """One local/centralized training run. prox_mu>0 adds the FedProx term;
    mixup_alpha>0 enables mixup; ema_decay>0 evaluates the EMA weights.
    Only parameters with requires_grad are optimized (freeze a subset to
    train only the head or only the body). eval_modules stay in eval mode
    (frozen norm statistics, no dropout) while the rest trains."""
    model.to(device).train()
    for m in (eval_modules or []):
        m.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps = max_steps or (epochs * max(1, len(loader)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    crit = nn.BCEWithLogitsLoss()
    ema = ModelEma(model, ema_decay) if ema_decay > 0 else None

    if global_params is not None:
        global_params = [g.to(device) for g in global_params]
        all_params = list(model.parameters())

    step = 0
    for ep in range(epochs):
        for x, y in loader:
            if step >= steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mixup_alpha > 0 and x.size(0) > 1:
                x, y = _mixup(x, y, mixup_alpha)
            opt.zero_grad(set_to_none=True)
            with _amp_ctx(device):
                loss = crit(model(x), y)
            if prox_mu > 0 and global_params is not None:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(all_params, global_params))
                loss = loss + 0.5 * prox_mu * prox
            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # clip in real (unscaled) gradient space
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            step += 1
        if log_fn:
            log_fn(f"  epoch {ep + 1}/{epochs} loss={loss.item():.4f}")
        if ckpt_path and ((ep + 1) % ckpt_every == 0 or ep + 1 == epochs):
            save_model = model
            if ema is not None:  # checkpoint EMA weights without disturbing training
                tmp = {k: v.clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
                torch.save({"epoch": ep + 1, "model": model.state_dict()}, ckpt_path)
                model.load_state_dict(tmp)
            else:
                torch.save({"epoch": ep + 1, "model": save_model.state_dict()}, ckpt_path)
    if ema is not None:
        ema.copy_to(model)
    return model


def train_model_dp(model, loader, device, epochs, lr, noise_multiplier,
                   max_grad_norm, accountant=None, weight_decay=1e-4):
    """Record-level DP-SGD (Opacus) for one local training call.

    Poisson sampling with the loader's batch size as the expected batch,
    per-sample clipping to max_grad_norm and Gaussian noise of std
    noise_multiplier * max_grad_norm. `accountant` (an RDPAccountant kept
    across rounds) is stepped once per sampled batch, so the privacy cost
    accumulates over the whole federated run. FP32 only, no mixup/EMA."""
    from opacus import GradSampleModule
    from opacus.data_loader import DPDataLoader
    from opacus.optimizers import DPOptimizer

    model.to(device).train()
    gsm = GradSampleModule(model, loss_reduction="mean")
    dl = DPDataLoader.from_data_loader(loader)
    q = 1.0 / len(dl)  # Opacus: sample_rate = 1 / steps per epoch
    params = [p for p in model.parameters() if p.requires_grad]
    opt = DPOptimizer(torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay),
                      noise_multiplier=noise_multiplier,
                      max_grad_norm=max_grad_norm,
                      expected_batch_size=loader.batch_size,
                      loss_reduction="mean")
    steps = epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt.original_optimizer,
                                                max_lr=lr, total_steps=steps)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for x, y in dl:
            if accountant is not None:
                accountant.step(noise_multiplier=noise_multiplier, sample_rate=q)
            if x.size(0) == 0:  # empty Poisson batch: nothing to release
                sched.step()
                continue
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(gsm(x), y)
            loss.backward()
            opt.step()
            sched.step()
    gsm._close()  # remove hooks and per-sample grads; model keeps its weights
    return model


@torch.no_grad()
def predict(model, loader, device, crop_len=CROP_LEN, stride=EVAL_STRIDE):
    """Sliding-window prediction over full 10 s records, mean-aggregated."""
    model.to(device).eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if x.shape[-1] > crop_len:
            windows = x.unfold(-1, crop_len, stride)        # (B,12,W,crop)
            B, C, W, L = windows.shape
            flat = windows.permute(0, 2, 1, 3).reshape(B * W, C, L)
            with _amp_ctx(device):
                out = torch.sigmoid(model(flat).float()).reshape(B, W, -1).mean(1)
        else:
            with _amp_ctx(device):
                out = torch.sigmoid(model(x).float())
        probs.append(out.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


@torch.no_grad()
def cache_features(model, loader, device, passes=2):
    """Pooled backbone features of `passes` random training crops per record
    (body frozen, eval mode). Used by FedRep's head phase."""
    model.to(device).eval()
    feats, ys = [], []
    for _ in range(passes):
        for x, y in loader:
            with _amp_ctx(device):
                f = model.forward_features(x.to(device, non_blocking=True))
            feats.append(f.float().cpu())
            ys.append(y)
    return torch.cat(feats), torch.cat(ys)


def train_head_on_features(head, feats, ys, device, epochs, lr,
                           weight_decay=1e-4, batch_size=256):
    """Optimize only the classification head on cached features."""
    head.to(device).train()
    n = len(feats)
    steps_per_epoch = max(1, int(np.ceil(n / batch_size)))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(steps_per_epoch):
            b = perm[i * batch_size:(i + 1) * batch_size]
            if len(b) < 2:  # BatchNorm needs more than one sample
                sched.step()
                continue
            f, y = feats[b].to(device), ys[b].to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(head(f), y)
            loss.backward()
            opt.step()
            sched.step()
    return head


def local_finetune(model, train_loader, val_loader, device, epochs, lr,
                   trainable=None):
    """Local fine-tuning with validation-based epoch selection.

    trainable: None = all parameters, else a set of parameter names. After
    each epoch the model is scored by validation macro-AUROC; the best
    epoch's weights (epoch 0 = no fine-tuning) are restored and returned.
    Returns (model, info) with the selected epoch and the AUROC trace."""
    names = dict(model.named_parameters())
    if trainable is not None:
        for n, p in names.items():
            p.requires_grad_(n in trainable)
    pv, yv = predict(model, val_loader, device)
    best = (macro_auroc(yv, pv), 0, copy.deepcopy(model.state_dict()))
    trace = [best[0]]
    for ep in range(1, epochs + 1):
        train_model(model, train_loader, device, epochs=1, lr=lr)
        pv, yv = predict(model, val_loader, device)
        auc = macro_auroc(yv, pv)
        trace.append(auc)
        if auc > best[0]:
            best = (auc, ep, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    for p in model.parameters():
        p.requires_grad_(True)
    return model, {"best_epoch": best[1], "val_auroc_trace": trace}


def evaluate(model, val_loader, test_loader, device, return_preds=False):
    """Tune thresholds on val, report AUROC/F1 on test."""
    pv, yv = predict(model, val_loader, device)
    th = tune_thresholds(yv, pv)
    pt, yt = predict(model, test_loader, device)
    metrics = {"auroc": macro_auroc(yt, pt), "f1": macro_f1(yt, pt, th)}
    if return_preds:
        return metrics, (pt, yt, th), (pv, yv)
    return metrics, (pt, yt, th)
