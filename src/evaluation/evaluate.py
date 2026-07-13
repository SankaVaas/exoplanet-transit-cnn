"""
Full evaluation pipeline: load a trained checkpoint, run it on the held-out
test set, and produce the complete honest report described in README §3 —
bootstrapped metrics with 95% CIs, temperature-scaling calibration (fit on
val, applied to test), and Grad-CAM saliency figures for a sample of
correct and incorrect predictions.

Usage:
    python -m src.evaluation.evaluate --checkpoint models/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TransitViewDataset, load_splits
from src.evaluation.calibration import TemperatureScaler, expected_calibration_error
from src.evaluation.metrics import full_evaluation_report
from src.evaluation.saliency import LocalViewGradCAM
from src.evaluation.thresholding import find_optimal_threshold, metrics_at_threshold
from src.models.astronet_lite import AstroNetLite
from src.utils.config import ConfigDict, load_config
from src.utils.logging_utils import get_logger
from src.utils.seeding import get_device, set_global_seed

logger = get_logger(__name__)


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a full DataLoader and collect raw logits + labels.

    Returns:
        (logits, labels) as numpy arrays, both shape (N,). Logits are kept
        raw (not sigmoided) so calibration can be fit/applied on them
        directly — sigmoiding before calibration would need to be undone,
        an easy source of a subtle double-transform bug.
    """
    model.eval()
    all_logits, all_labels = [], []
    for global_view, local_view, labels in loader:
        global_view, local_view = global_view.to(device), local_view.to(device)
        logits = model(global_view, local_view)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def plot_reliability_diagram(
    y_true: np.ndarray, y_prob_raw: np.ndarray, y_prob_calibrated: np.ndarray, out_path: Path
) -> None:
    """Save a reliability diagram (predicted confidence vs. observed
    frequency) comparing raw vs. temperature-calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for probs, label, color in [
        (y_prob_raw, "Raw (uncalibrated)", "tab:red"),
        (y_prob_calibrated, "Temperature-scaled", "tab:blue"),
    ]:
        bins = np.linspace(0, 1, 11)
        bin_idx = np.digitize(probs, bins[1:-1])
        bin_conf, bin_acc = [], []
        for b in range(10):
            mask = bin_idx == b
            if mask.sum() > 0:
                bin_conf.append(probs[mask].mean())
                bin_acc.append(y_true[mask].mean())
        ax.plot(bin_conf, bin_acc, marker="o", label=label, color=color)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram: Raw vs. Calibrated")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved reliability diagram -> {out_path}")


def plot_saliency_examples(
    model: AstroNetLite,
    test_df,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
    n_examples: int = 4,
) -> None:
    """Save Grad-CAM overlays for a handful of correct and incorrect
    high-confidence predictions — the "error analysis" figures referenced
    in README §3, letting a reviewer see *what the model attended to* on
    cases it got right vs. wrong."""
    cam_tool = LocalViewGradCAM(model)
    y_pred = (y_prob >= 0.5).astype(int)
    correct_mask = y_pred == y_true

    out_dir.mkdir(parents=True, exist_ok=True)
    for category, mask in [("correct", correct_mask), ("incorrect", ~correct_mask)]:
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            logger.warning(f"No '{category}' examples found in test set to visualize.")
            continue
        # pick the most confident examples in this category for a clean visualization
        confidence = np.abs(y_prob[idxs] - 0.5)
        chosen = idxs[np.argsort(-confidence)[:n_examples]]

        for i, idx in enumerate(chosen):
            row = test_df.iloc[idx]
            npz = np.load(row["view_path"])
            gview = torch.from_numpy(npz["global_view"]).float().view(1, 1, -1)
            lview = torch.from_numpy(npz["local_view"]).float().view(1, 1, -1)
            saliency = cam_tool.compute(gview, lview)

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(npz["local_view"], color="black", linewidth=0.8, label="local view (flux)")
            ax2 = ax.twinx()
            ax2.fill_between(range(len(saliency)), saliency, alpha=0.3, color="tab:red", label="Grad-CAM saliency")
            ax.set_title(
                f"{row['kepoi_name']} | true={row['label']} pred_prob={y_prob[idx]:.3f} ({category})"
            )
            ax.set_xlabel("local view bin")
            fig.tight_layout()
            fig.savefig(out_dir / f"{category}_{i}_{row['kepoi_name']}.png", dpi=150)
            plt.close(fig)

    logger.info(f"Saved saliency example figures -> {out_dir}")


def run_evaluation(checkpoint_path: Path, cfg: ConfigDict) -> dict:
    """Full evaluation: load checkpoint, evaluate on test set, calibrate,
    save figures, write reports/results.md.

    Returns:
        The full metrics report dict (also written to disk as JSON and
        summarized in reports/results.md).
    """
    set_global_seed(cfg.project.seed)
    device = get_device(cfg.training.device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AstroNetLite.from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_pr_auc={checkpoint['val_pr_auc']:.4f})")

    _train_df, val_df, test_df = load_splits(cfg)
    val_loader = DataLoader(TransitViewDataset(val_df), batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(TransitViewDataset(test_df), batch_size=cfg.training.batch_size, shuffle=False)

    val_logits, val_labels = collect_logits(model, val_loader, device)
    test_logits, test_labels = collect_logits(model, test_loader, device)

    # Fit calibration ONLY on validation logits, apply to test — fitting
    # on the test set itself would leak test information into the
    # "calibrated" probabilities we then evaluate on that same test set.
    scaler = TemperatureScaler()
    fitted_temperature = scaler.fit(torch.from_numpy(val_logits), torch.from_numpy(val_labels))
    logger.info(f"Fitted calibration temperature: {fitted_temperature:.4f}")

    test_probs_raw = torch.sigmoid(torch.from_numpy(test_logits)).numpy()
    test_probs_calibrated = scaler.calibrate(torch.from_numpy(test_logits)).numpy()

    ece_raw = expected_calibration_error(test_labels, test_probs_raw)
    ece_calibrated = expected_calibration_error(test_labels, test_probs_calibrated)
    logger.info(f"ECE raw: {ece_raw:.4f} -> ECE calibrated: {ece_calibrated:.4f}")

    # Select the decision threshold on VALIDATION probabilities only (never
    # test) — see src/evaluation/thresholding.py module docstring for why a
    # fixed 0.5 cutoff can silently produce 0 precision/recall despite a
    # well-ranking model under class imbalance, even though ROC-AUC/PR-AUC
    # (threshold-independent) look fine.
    val_probs_calibrated = scaler.calibrate(torch.from_numpy(val_logits)).numpy()
    try:
        optimal_threshold, _val_f1 = find_optimal_threshold(val_labels, val_probs_calibrated, criterion="f1")
    except ValueError:
        logger.warning("Validation set has only one class present — falling back to threshold=0.5.")
        optimal_threshold = 0.5

    report = full_evaluation_report(
        test_labels,
        test_probs_calibrated,
        threshold=cfg.evaluation.decision_threshold,
        n_bootstrap=cfg.evaluation.bootstrap_iterations,
        seed=cfg.project.seed,
    )
    report["calibration"] = {
        "temperature": fitted_temperature,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_calibrated,
    }
    report["threshold_comparison"] = {
        "default_0.5": metrics_at_threshold(test_labels, test_probs_calibrated, 0.5),
        "validation_optimal": metrics_at_threshold(test_labels, test_probs_calibrated, optimal_threshold),
    }
    report["n_test_examples"] = len(test_labels)

    figures_dir = Path(cfg.logging.figures_dir)
    plot_reliability_diagram(test_labels, test_probs_raw, test_probs_calibrated, figures_dir / "reliability_diagram.png")
    plot_saliency_examples(model, test_df, test_labels, test_probs_calibrated, figures_dir / "saliency_examples")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2)

    _write_results_markdown(report, reports_dir / "results.md")
    logger.info(f"Full evaluation report written to {reports_dir / 'results.md'}")
    return report


def _write_results_markdown(report: dict, out_path: Path) -> None:
    """Render the JSON report into the honest, CI-inclusive markdown
    summary referenced throughout the README."""
    lines = [
        "# Results",
        "",
        f"Evaluated on {report['n_test_examples']} held-out test examples "
        "(grouped split — no host star shared with train/val).",
        "",
        "All metrics computed on temperature-calibrated probabilities, "
        f"with bootstrapped 95% confidence intervals ({report.get('n_test_examples', '?')} "
        "test examples, percentile bootstrap).",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
    ]
    for name in ["precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]:
        m = report[name]
        lines.append(f"| {name} | {m['point']:.3f} | [{m['ci_lower']:.3f}, {m['ci_upper']:.3f}] |")

    cal = report["calibration"]
    lines += [
        "",
        "## Calibration",
        f"- Fitted temperature: {cal['temperature']:.3f}",
        f"- Expected Calibration Error, raw: {cal['ece_raw']:.4f}",
        f"- Expected Calibration Error, calibrated: {cal['ece_calibrated']:.4f}",
    ]

    if "threshold_comparison" in report:
        tc = report["threshold_comparison"]
        lines += [
            "",
            "## Decision threshold: default (0.5) vs. validation-optimal",
            "",
            "A fixed 0.5 threshold can be misleading under class imbalance — "
            "ROC-AUC/PR-AUC above are threshold-independent, but precision/"
            "recall/F1 at any single cutoff depend heavily on where that "
            "cutoff sits relative to the model's (calibrated) probability "
            "distribution. The threshold below is selected on the "
            "**validation** set only, never on test.",
            "",
            "| Threshold | Value | Precision | Recall | F1 |",
            "|---|---|---|---|---|",
            f"| Default | {tc['default_0.5']['threshold']:.3f} | "
            f"{tc['default_0.5']['precision']:.3f} | {tc['default_0.5']['recall']:.3f} | "
            f"{tc['default_0.5']['f1']:.3f} |",
            f"| Validation-optimal (F1) | {tc['validation_optimal']['threshold']:.3f} | "
            f"{tc['validation_optimal']['precision']:.3f} | {tc['validation_optimal']['recall']:.3f} | "
            f"{tc['validation_optimal']['f1']:.3f} |",
        ]

    lines += [
        "",
        "See `outputs/figures/reliability_diagram.png` and "
        "`outputs/figures/saliency_examples/` for the corresponding figures.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_evaluation(Path(args.checkpoint), cfg)


if __name__ == "__main__":
    main()
