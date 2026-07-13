"""
Main training loop.

Runs identically on CPU (local dev, small smoke-test subsets) and on a
Colab T4 (full runs, mixed precision) — device and AMP are resolved
automatically from config.yaml `training.device` / `training.mixed_precision`,
per the compute-budget section of the README.

Usage:
    python -m src.training.train --config config.yaml
    python -m src.training.train --config config.yaml --smoke-test   # tiny CPU run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import TransitViewDataset, load_splits
from src.models.astronet_lite import AstroNetLite
from src.training.losses import build_loss
from src.utils.config import load_config, override_config
from src.utils.logging_utils import get_logger
from src.utils.seeding import get_device, set_global_seed

logger = get_logger(__name__)


class EarlyStopper:
    """Stops training when a monitored metric stops improving.

    Args:
        patience: number of epochs with no improvement before stopping.
        mode: "max" (higher is better, e.g. PR-AUC) or "min" (lower is
            better, e.g. loss). Matches the direction of
            config.yaml `logging.monitor_metric`.
    """

    def __init__(self, patience: int = 8, mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")
        self.patience = patience
        self.mode = mode
        self.best_score: float | None = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Update with the latest epoch's score. Returns True if this is a
        new best score (caller should checkpoint the model)."""
        is_improvement = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )
        if is_improvement:
            self.best_score = score
            self.epochs_without_improvement = 0
            return True

        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
        return False


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, device) -> dict[str, float]:
    """Run one evaluation pass (no gradient) over a DataLoader, returning
    average loss and predictions needed for downstream metrics.

    Returns:
        dict with "loss" (float) and "n_examples" (int). Full metric
        computation (PR-AUC, calibration, etc.) lives in
        src/evaluation/evaluate.py and is intentionally NOT duplicated
        here — the training loop only needs a lightweight signal for
        early stopping / checkpointing.
    """
    model.eval()
    total_loss = 0.0
    n_examples = 0
    all_probs = []
    all_targets = []

    for global_view, local_view, targets in loader:
        global_view = global_view.to(device)
        local_view = local_view.to(device)
        targets = targets.to(device)

        logits = model(global_view, local_view)
        loss = loss_fn(logits, targets)

        total_loss += loss.item() * targets.size(0)
        n_examples += targets.size(0)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_targets.append(targets.cpu())

    from sklearn.metrics import average_precision_score

    probs = torch.cat(all_probs).numpy()
    targets_np = torch.cat(all_targets).numpy()
    pr_auc = average_precision_score(targets_np, probs) if len(set(targets_np.tolist())) > 1 else float("nan")

    return {
        "loss": total_loss / max(n_examples, 1),
        "pr_auc": pr_auc,
        "n_examples": n_examples,
    }


def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler, grad_accum_steps: int) -> float:
    """Run one training epoch. Returns the average training loss.

    Handles gradient accumulation (config: training.gradient_accumulation_steps)
    and automatic mixed precision (config: training.mixed_precision, only
    active when device is CUDA — a no-op on CPU dev runs).
    """
    model.train()
    total_loss = 0.0
    n_examples = 0
    optimizer.zero_grad()

    for step, (global_view, local_view, targets) in enumerate(loader):
        global_view = global_view.to(device)
        local_view = local_view.to(device)
        targets = targets.to(device)

        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            logits = model(global_view, local_view)
            loss = loss_fn(logits, targets) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * targets.size(0)
        n_examples += targets.size(0)

    return total_loss / max(n_examples, 1)


def run_training(cfg) -> Path:
    """Full training run: build datasets/model/optimizer, train with early
    stopping, checkpoint the best model.

    Returns:
        Path to the best checkpoint saved.
    """
    set_global_seed(cfg.project.seed)
    device = get_device(cfg.training.device)
    logger.info(f"Training on device: {device}")

    train_df, val_df, _test_df = load_splits(cfg)
    train_loader = DataLoader(
        TransitViewDataset(train_df),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        TransitViewDataset(val_df),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    model = AstroNetLite.from_config(cfg).to(device)
    logger.info(f"Model parameter count: {model.count_parameters():,}")

    loss_fn = build_loss(cfg.training.loss, cfg.training.focal_gamma, cfg.training.focal_alpha)

    if cfg.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")

    if cfg.training.scheduler == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    else:
        scheduler = None

    use_amp = bool(cfg.training.mixed_precision) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    logger.info(f"Mixed precision active: {use_amp}")

    checkpoint_dir = Path(cfg.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"

    writer = SummaryWriter(log_dir=cfg.logging.tensorboard_dir)
    stopper = EarlyStopper(patience=cfg.training.early_stopping_patience, mode="max")

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler,
            cfg.training.gradient_accumulation_steps,
        )
        val_metrics = evaluate_epoch(model, val_loader, loss_fn, device)
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:3d}/{cfg.training.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} | {elapsed:.1f}s"
        )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("pr_auc/val", val_metrics["pr_auc"], epoch)

        is_best = stopper.step(val_metrics["pr_auc"])
        if is_best and cfg.logging.save_best_only:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_pr_auc": val_metrics["pr_auc"],
                    "config": dict(cfg),
                },
                best_checkpoint_path,
            )
            logger.info(f"  -> new best model saved (val_pr_auc={val_metrics['pr_auc']:.4f})")

        if stopper.should_stop:
            logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for "
                        f"{cfg.training.early_stopping_patience} epochs).")
            break

    writer.close()
    logger.info(f"Training complete. Best val_pr_auc: {stopper.best_score:.4f}")
    logger.info(f"Best checkpoint: {best_checkpoint_path}")
    return best_checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2 epochs, batch size 4, on CPU — for quickly verifying the "
        "pipeline runs end-to-end before committing to a full Colab T4 run.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke_test:
        cfg = override_config(
            cfg,
            {
                "training.epochs": 2,
                "training.batch_size": 4,
                "training.device": "cpu",
                "training.num_workers": 0,
                "training.early_stopping_patience": 2,
            },
        )
        logger.info("Running in --smoke-test mode: 2 epochs, batch_size=4, CPU.")

    run_training(cfg)


if __name__ == "__main__":
    main()
