"""Multi-task masked training loop, device selection, and checkpoint contracts."""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from .data.windows import WindowSet

FIELDS = ("values", "observed_mask", "feature_present_mask", "age", "targets", "target_mask",
          "drug_targets", "drug_mask", "drug_history", "start_targets", "start_mask")


class ArrayDataset(Dataset):
    def __init__(self, windows: WindowSet, split: str) -> None:
        self.indices = np.flatnonzero(windows.split == split); self.windows = windows
        if not len(self.indices): raise ValueError(f"split has no windows: {split}")
    def __len__(self): return len(self.indices)
    def __getitem__(self, index):
        i = self.indices[index]
        return tuple(torch.from_numpy(getattr(self.windows, name)[i]).float() for name in FIELDS)


def resolve_device(requested: str | None) -> torch.device:
    """`auto` (the default) takes CUDA when present, then MPS, then CPU."""
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def wrap_for_devices(model: nn.Module, device: torch.device, verbose: bool = True) -> nn.Module:
    """Spread the batch over every visible CUDA device.

    DataParallel rather than DistributedDataParallel on purpose: this runs inside Kaggle
    and Colab notebooks, where DDP needs a process launcher that a notebook kernel has no
    clean way to provide. For the 2x T4 / 2x GPU shapes those platforms hand out, the
    difference is throughput, not correctness.
    """
    model.to(device)
    count = torch.cuda.device_count() if device.type == "cuda" else 0
    if count > 1:
        if verbose:
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(count))
            print(f"using {count} CUDA devices via DataParallel: {names}", flush=True)
        return nn.DataParallel(model)
    return model


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def drop_columns(present: torch.Tensor, rate: float, generator: torch.Generator | None = None
                 ) -> torch.Tensor:
    """Sample a per-sample column-dropout mask: 1 = keep the column, 0 = hide it.

    This is what trains the impute head. A sample never loses every column, because a model
    given nothing to condition on can only return the training median, and a batch of those
    teaches the head to ignore its input.
    """
    if rate <= 0:
        return present
    keep = (torch.rand(present.shape, device=present.device, generator=generator) >= rate).float() * present
    empty = keep.sum(1) == 0
    if empty.any():  # restore one real column for any sample left with nothing
        fallback = present[empty]
        first = fallback.argmax(1, keepdim=True)
        keep[empty] = torch.zeros_like(fallback).scatter_(1, first, fallback.gather(1, first))
    return keep


def apply_column_dropout(values, observed, present, age, keep):
    """Hide the dropped columns from every input channel, consistently.

    A hidden column must look exactly like a column the source never had: zero in the
    scaled values (the training median), zero observed, zero present, maximum staleness.
    Leaving any one channel intact would leak the answer the impute head is asked for.
    """
    column = keep.unsqueeze(1)
    return values * column, observed * column, keep, age * column + (1.0 - column)


def model_inputs(batch, rate: float, med_rate: float = 0.0, generator: torch.Generator | None = None,
                 meds: bool = True):
    """Positional model inputs for one batch, with column and medication dropout applied.

    Medication dropout hides the whole current-medication input for a fraction of samples,
    so one model serves both "here is what the patient is on" and "we do not know".
    `meds=False` forces the unknown case, for evaluation.
    """
    values, observed, present, age = batch[:4]
    keep = drop_columns(present, rate, generator) if rate > 0 else None
    inputs = apply_column_dropout(values, observed, present, age, keep) if keep is not None else (values, observed, present, age)
    history, known = batch[8], batch[7]
    if not meds:
        history, known = history * 0.0, known * 0.0
    elif med_rate > 0:
        shown = (torch.rand((len(values), 1), device=values.device, generator=generator) >= med_rate).float()
        history, known = history * shown.unsqueeze(1), known * shown
    return (*inputs, history, known), keep


def masked_mse(prediction, target, mask):
    count = mask.sum()
    if count <= 0: raise ValueError("batch has no observed targets")
    return (((prediction - target) ** 2) * mask).sum() / count


def masked_bce(logits, target, mask, pos_weight=None):
    """Multi-label BCE over the drug classes the source actually charts."""
    if mask.sum() <= 0:
        return logits.sum() * 0.0
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum()


def multitask_loss(output, batch, weights, pos_weight=None, keep=None, start_weight=None):
    """Weighted sum of the heads' masked losses, plus the per-head parts."""
    _, observed, _, _, targets, target_mask, drug_targets, drug_mask = batch[:8]
    parts = {"forecast": masked_mse(output["forecast"], targets, target_mask)}
    if "drug_logits" in output and weights.get("drug", 0.0):
        parts["drug"] = masked_bce(output["drug_logits"], drug_targets, drug_mask, pos_weight)
    if "start_logits" in output and weights.get("drug_start", 0.0) and len(batch) > 10:
        parts["drug_start"] = masked_bce(output["start_logits"], batch[9], batch[10], start_weight)
    if keep is not None and weights.get("impute", 0.0):
        # Supervised only where the column was hidden AND the hidden truth was observed.
        hidden = (1.0 - keep) * observed[:, -1, :]
        parts["impute"] = (masked_mse(output["impute"], batch[0][:, -1, :], hidden)
                           if hidden.sum() > 0 else output["impute"].sum() * 0.0)
    total = sum(weights.get(name, 1.0) * value for name, value in parts.items())
    return total, {name: float(value.detach()) for name, value in parts.items()}


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, weights: dict,
                  pos_weight=None, dropout_rate: float = 0.0, seed: int = 0, med_rate: float = 0.0,
                  start_weight=None) -> float:
    """Validation objective: the same weighted multi-task loss, with a fixed dropout draw.

    The column-dropout draw is reseeded per call so the validation number moves with the
    model and not with the random stream.
    """
    model.eval(); totals = 0.0; batches = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for raw in loader:
        batch = tuple(item.to(device, non_blocking=True) for item in raw)
        inputs, keep = model_inputs(batch, dropout_rate, med_rate, generator)
        total, _ = multitask_loss(model(*inputs), batch, weights, pos_weight, keep, start_weight)
        totals += float(total); batches += 1
    if not batches: raise ValueError("evaluation split has no windows")
    return totals / batches


def train_model(model: nn.Module, windows: WindowSet, cfg: dict, checkpoint_path: Path, contract: dict,
                pos_weight: np.ndarray | None = None, verbose: bool = True,
                start_weight: np.ndarray | None = None) -> dict:
    settings = cfg["training"]
    device = resolve_device(cfg["project"].get("device"))
    torch.manual_seed(int(cfg["project"]["seed"]))
    weights = {str(k): float(v) for k, v in cfg["task"].get("loss_weights", {}).items()} or {"forecast": 1.0}
    rate = float(cfg["task"].get("column_dropout_rate", 0.0))
    med_rate = float(cfg["task"].get("medication_input_dropout", 0.0)) if cfg["task"].get("medication_inputs") else 1.0
    model = wrap_for_devices(model, device, verbose)
    pinned = device.type == "cuda"
    workers = int(settings.get("num_workers", 0))
    train_loader = DataLoader(ArrayDataset(windows, "train"), batch_size=int(settings["batch_size"]), shuffle=True,
                              num_workers=workers, pin_memory=pinned, drop_last=False,
                              persistent_workers=bool(workers))
    val_loader = DataLoader(ArrayDataset(windows, "val"), batch_size=int(settings["batch_size"]), shuffle=False,
                            num_workers=workers, pin_memory=pinned, persistent_workers=bool(workers))
    weight_tensor = (torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
                     if pos_weight is not None else None)
    start_tensor = (torch.as_tensor(start_weight, dtype=torch.float32, device=device)
                    if start_weight is not None else None)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    amp = bool(settings.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    # Halve the learning rate when validation stalls; early stopping still has the last word.
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(1, int(settings["patience"]) // 3))
    best = float("inf"); stale = 0; history = []; checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = int(settings["epochs"]); patience = int(settings["patience"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        heads = "+".join(name for name in ("forecast", "drug", "drug_start", "impute") if weights.get(name, 0.0))
        print(f"training on {len(train_loader.dataset):,} windows, validating on {len(val_loader.dataset):,}"
              f"  |  {trainable:,} trainable parameters  |  device {device}", flush=True)
        print(f"heads: {heads}  |  column dropout {rate:.0%}  |  mixed precision {'on' if amp else 'off'}", flush=True)
        print(f"{'epoch':>7}  {'train':>8}  {'val':>8}  {'best':>8}  {'secs':>6}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train(); started = time.perf_counter(); running = 0.0; seen = 0; parts_sum: dict[str, float] = {}
        for raw in train_loader:
            batch = tuple(item.to(device, non_blocking=True) for item in raw)
            values = batch[0]
            inputs, keep = model_inputs(batch, rate, med_rate)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                loss, parts = multitask_loss(model(*inputs), batch, weights, weight_tensor, keep, start_tensor)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += float(loss.detach()) * len(values); seen += len(values)
            for name, value in parts.items(): parts_sum[name] = parts_sum.get(name, 0.0) + value * len(values)
        val = evaluate_loss(model, val_loader, device, weights, weight_tensor, rate, int(cfg["project"]["seed"]),
                            med_rate, start_tensor)
        schedule.step(val)
        history.append({"epoch": epoch, "train_loss": running / seen, "val_loss": val,
                        **{f"train_{k}": v / seen for k, v in parts_sum.items()}})
        improved = val < best
        if improved:
            best = val; stale = 0
            torch.save({"model": unwrap(model).state_dict(), "epoch": epoch, "val_loss": best, "contract": contract}, checkpoint_path)
        else:
            stale += 1
        if verbose:
            note = " saved" if improved else f"  no gain ({stale}/{patience})"
            detail = "  ".join(f"{k}={v / seen:.4f}" for k, v in sorted(parts_sum.items()))
            print(f"{epoch:>3}/{epochs:<3}  {running / seen:>8.4f}  {val:>8.4f}  {best:>8.4f}  {time.perf_counter() - started:>6.1f}{note}   {detail}", flush=True)
        if not improved and stale >= patience:
            if verbose: print(f"early stop: validation has not improved for {patience} epochs", flush=True)
            break
    if verbose:
        print(f"done. best validation loss {best:.4f} at epoch {min(history, key=lambda h: h['val_loss'])['epoch']}"
              f", checkpoint -> {checkpoint_path}", flush=True)
    return {"best_val_loss": best, "epochs_ran": len(history), "device": str(device), "history": history}
