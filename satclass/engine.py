"""Training / evaluation loops."""
from __future__ import annotations

import contextlib

import numpy as np
import torch
from tqdm import tqdm


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, desc=""):
    """One pass over ``loader``. Trains if an optimizer is given, otherwise evaluates.

    Returns a dict with mean loss, accuracy, and numpy arrays of targets/preds/probs.
    """
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None and device.type == "cuda"
    total_loss, n = 0.0, 0
    targets, preds, probs = [], [], []

    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
            with amp_ctx:
                logits = model(x)
                loss = criterion(logits, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * y.size(0)
            n += y.size(0)
            p = logits.float().softmax(1)
            probs.append(p.detach().cpu())
            preds.append(p.argmax(1).cpu())
            targets.append(y.cpu())

    targets = torch.cat(targets).numpy()
    preds = torch.cat(preds).numpy()
    return {
        "loss": total_loss / max(n, 1),
        "acc": float((targets == preds).mean()),
        "targets": targets,
        "preds": preds,
        "probs": torch.cat(probs).numpy(),
    }


@torch.no_grad()
def predict_tensor_batch(model, batch: torch.Tensor, device) -> tuple[np.ndarray, np.ndarray]:
    """Already-preprocessed batch -> (pred indices, probabilities)."""
    probs = model(batch.to(device)).float().softmax(1).cpu()
    return probs.argmax(1).numpy(), probs.numpy()
