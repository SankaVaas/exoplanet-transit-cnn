# exoplanet-transit-cnn

**Detecting transiting exoplanet candidates from Kepler/TESS light curves using a
global/local-view convolutional network with attention and calibrated uncertainty.**

---

## 1. Motivation: the physical problem

When a planet passes in front of its host star (a *transit*), it blocks a tiny
fraction of the star's light — typically a dip of 0.01%–1% in brightness,
lasting hours, recurring periodically. The Kepler and TESS missions have
produced photometric time series (*light curves*) for hundreds of thousands
of stars specifically to catch this signal.

The problem is not detecting *a* dip — pipelines like the Kepler/TESS
Science Processing pipeline already do that via box-least-squares search and
flag *Threshold Crossing Events* (TCEs). The real problem is **separating true
planetary transits from astrophysical false positives**: eclipsing binaries,
background eclipsing binaries blended into the same pixel, stellar
variability, and instrumental artifacts all produce transit-*like* dips. This
is a triage problem the community currently spends significant human
vetting time on.

This project reframes that vetting step as supervised classification over
TCEs, using labels from the NASA Exoplanet Archive's cumulative KOI
(Kepler Objects of Interest) disposition table.

## 2. Prior work this builds on

- **Shallue & Vanderburg (2018)**, *"Identifying Exoplanets with Deep Learning"* —
  introduced the global-view (full-orbit, 2001-bin) + local-view (201-bin,
  zoomed on the transit) dual-input CNN design (`AstroNet`). This repo's
  default architecture (`astronet_lite`, see `config.yaml`) is a lighter
  version of that design, sized to train on a single Colab T4 in under
  ~2 hours rather than requiring a TPU pod.
- **Ansdell et al. (2018)**, *"Scientific Domain Knowledge Improves Exoplanet
  Transit Classification with Deep Learning"* — showed that adding
  stellar-parameter side-features (radius, surface gravity) alongside the
  light curve views improves performance. This project includes an optional
  side-input branch for the same reason (see `src/models/astronet_lite.py`).
- **Ansdell et al.** and **Shallue & Vanderburg** both used a single
  fixed CNN; we add a lightweight self-attention layer over the local view
  to let the model weight the transit ingress/egress asymmetrically —
  motivated by the fact that grazing transits and secondary eclipses are
  visually distinguished by exactly this kind of shape asymmetry.

## 3. Why this problem is *hard* — and what "success" honestly means

This README makes no claim of perfect classification, and neither should any
serious project in this domain. A few reasons:

- **The KOI dispositions used as ground truth are themselves the product of a
  mix of automated vetting and human judgment** — they are the best
  available labels, not an infallible oracle. A small fraction of "confirmed"
  and "false positive" labels are later revised as more data comes in.
- **Grazing transits and shallow small-planet signals are close to the noise
  floor** — no model, including the published state of the art, resolves
  these with certainty from photometry alone.
- **Class imbalance is severe**: confirmed/candidate planets are a small
  minority of TCEs vetted, most of which are false positives.

Given this, this project reports **precision, recall, F1, ROC-AUC, PR-AUC,
and calibration (Brier score)** with bootstrapped 95% confidence intervals,
rather than a single accuracy number — and treats PR-AUC as the primary
metric, since accuracy is a misleading metric under class imbalance.
Final results, once training is complete, will be logged in
`reports/results.md` with full honesty about false positive/negative cases,
including example light curves the model got wrong and a discussion of why.

## 4. Repository structure

```
exoplanet-transit-cnn/
├── config.yaml                       # single source of truth for all parameters
├── pytest.ini
├── data/
│   ├── raw/                          # untouched downloads (light curves), gitignored
│   ├── processed/                     # global/local view tensors + metadata.csv, gitignored
│   └── external/                       # KOI disposition table (labels + ephemerides)
├── notebooks/
│   └── 01_eda.ipynb                    # exploratory analysis, class balance, worked example
├── src/
│   ├── data/
│   │   ├── download_koi_catalog.py       # NASA Exoplanet Archive TAP query
│   │   ├── download_light_curves.py       # lightkurve/MAST download, resumable
│   │   ├── preprocess.py                   # detrend -> phase-fold -> global/local views
│   │   └── dataset.py                       # PyTorch Dataset + grouped (leakage-safe) split
│   ├── models/
│   │   └── astronet_lite.py                  # dual-branch CNN + attention
│   ├── training/
│   │   ├── losses.py                           # focal loss
│   │   └── train.py                             # training loop, AMP, early stopping
│   ├── evaluation/
│   │   ├── metrics.py                            # bootstrapped precision/recall/F1/AUCs
│   │   ├── calibration.py                         # temperature scaling + ECE
│   │   ├── thresholding.py                         # validation-set threshold optimization
│   │   ├── saliency.py                              # 1D Grad-CAM for the local view
│   │   └── evaluate.py                               # full evaluation orchestrator
│   └── utils/                                          # config loader, seeding, logging helpers
├── models/checkpoints/                # best model weights (gitignored, released separately)
├── outputs/
│   ├── figures/                        # reliability diagram, Grad-CAM overlays
│   └── logs/                             # TensorBoard logs
├── reports/
│   └── results.md                          # honest write-up: metrics + CIs + calibration + thresholds
├── tests/                                    # 60 pytest tests covering every module above
├── requirements.txt
└── LICENSE
```

## 5. Method summary

| Stage | Detail |
|---|---|
| **Data source** | Kepler light curves via `lightkurve`/MAST; labels from NASA Exoplanet Archive cumulative KOI table |
| **Preprocessing** | Flatten (spline detrending), phase-fold on reported period, extract global view (2001 bins, full phase) and local view (201 bins, ±2× transit duration) |
| **Splitting** | Grouped by host star (`kepid`) — no star appears in both train and test, preventing leakage from multiple TCEs per star |
| **Architecture** | Dual-branch 1D-CNN (global + local views) + optional stellar-parameter side branch, lightweight self-attention over local view, fusion MLP head |
| **Loss** | Focal loss (γ=2.0, α=0.75) to address class imbalance, rather than naive cross-entropy |
| **Training hardware** | Local CPU for data pipeline dev/debugging; Colab T4 (mixed precision) for full training runs |
| **Evaluation** | Precision/recall/F1/ROC-AUC/PR-AUC/Brier score, all with bootstrapped 95% CIs; temperature-scaling calibration; Grad-CAM-style saliency over the local view |

## 6. Compute budget (stated explicitly, not hidden)

- **Local CPU** (dev machine): data download, preprocessing, unit tests, and
  smoke-testing the training loop on a ~200-sample subset (~5–10 min).
- **Colab T4** (full training): full training set (~15k TCEs after
  preprocessing), batch size 64, mixed precision, ~60 epochs with early
  stopping — target wall-clock ≈ 60–90 minutes per full run.
- Dataset footprint after preprocessing (global+local view tensors,
  float32): well under 1 GB — deliberately kept light enough to iterate on
  a free-tier T4 without hitting the 15 GB VRAM ceiling or Colab's disk quota.

## 7. Reproducing this project

```bash
git clone https://github.com/<your-username>/exoplanet-transit-cnn.git
cd exoplanet-transit-cnn
pip install -r requirements.txt

# 0. Run the test suite (60 tests, ~15s on CPU, no data download needed —
#    exercises detrending, phase-folding, model shapes, focal loss,
#    calibration, thresholding, and grouped-split leakage prevention
#    against synthetic data with known ground truth)
pytest

# 1. Download + label TCEs (writes to data/raw, data/external)
python -m src.data.download_koi_catalog
python -m src.data.download_light_curves --mission kepler --limit 50  # small first, drop --limit for full run

# 2. Preprocess into global/local view tensors (writes to data/processed)
python -m src.data.preprocess --config config.yaml

# 3. Train — CPU smoke test first, then full run on Colab T4
python -m src.training.train --smoke-test          # 2 epochs, tiny batch, CPU — verifies the pipeline runs
python -m src.training.train --config config.yaml    # full run (use Colab T4 for this)

# 4. Evaluate with bootstrapped confidence intervals, calibration, and
#    threshold analysis — writes reports/results.md + figures
python -m src.evaluation.evaluate --checkpoint models/checkpoints/best.pt
```

## 8. Status

- [x] Repository scaffolding, config, environment
- [x] Data download scripts (KOI catalog + light curves)
- [x] Preprocessing pipeline (detrending, phase-folding, view extraction) — unit tested against synthetic light curves with known injected transits
- [x] `astronet_lite` model (dual-branch CNN + attention) + training loop (focal loss, AMP, early stopping)
- [x] Grouped train/val/test split with leakage guard (no host star spans splits)
- [x] Calibration (temperature scaling) + bootstrapped evaluation + threshold optimization
- [x] Grad-CAM interpretability for the local-view branch
- [x] Full pytest suite (60 tests) covering every module above
- [ ] Full training run on real Kepler data via Colab T4 (scaffolding is complete and CPU-verified end-to-end on synthetic data; awaiting the multi-hour real download + train run)
- [ ] Final `reports/results.md` from the real trained model (structure and generation code are complete — see `src/evaluation/evaluate.py`)

### A concrete finding from pipeline validation

While building and testing this pipeline (see `tests/`), an instructive
failure mode showed up and is worth documenting rather than hiding: with a
class-imbalanced synthetic test, the model achieved **ROC-AUC 0.98 / PR-AUC
0.92** (excellent ranking) but **0 precision/recall/F1 at the default 0.5
threshold** — because calibrated probabilities for the rare positive class
sat entirely below 0.5. Selecting a threshold on the validation set (F1-
optimal) recovered a usable operating point (F1 0.60 in that run). This is
exactly why `src/evaluation/evaluate.py` reports metrics at *both* 0.5 and
a validation-selected threshold, and why threshold-independent metrics
(ROC-AUC/PR-AUC) are treated as primary — see `src/evaluation/thresholding.py`.

## 9. References

1. Shallue, C. J. & Vanderburg, A. (2018). *Identifying Exoplanets with Deep
   Learning: A Five-Planet Resonant Chain around Kepler-80 and an Eighth
   Planet around Kepler-90.* The Astronomical Journal, 155(2), 94.
2. Ansdell, M., et al. (2018). *Scientific Domain Knowledge Improves
   Exoplanet Transit Classification with Deep Learning.* The Astrophysical
   Journal Letters, 869(1), L7.
3. NASA Exoplanet Archive, Cumulative KOI Table —
   https://exoplanetarchive.ipac.caltech.edu
4. Lightkurve Collaboration (2018). *Lightkurve: Kepler and TESS time series
   analysis in Python.*

---

*This is a research-engineering portfolio project, not a scientific
publication. All performance claims in `reports/results.md` are reported
with confidence intervals and are meant to demonstrate rigorous ML practice
on a genuinely hard, real astrophysical dataset — not to claim novel
scientific discovery.*
