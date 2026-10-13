# QADiag-ML

**Capturing Complex Biomarker Interactions for Early Alzheimer’s Disease Diagnosis Using Quantum-Inspired Machine Learning**

QADiag-ML is a research-oriented, **classical PyTorch** implementation of a multimodal learning architecture for Alzheimer's disease (AD) diagnosis. It combines clinical–cognitive assessments, MRI/PET-derived region-of-interest (ROI) features, and biofluid–genetic biomarkers using structured pairwise/third-order interactions, angle-encoded features, matrix product state (MPS) aggregation, prototype-based kernel refinement, and adaptive gated fusion.

> **Research software only.** This repository is not a medical device or clinical diagnostic system. Synthetic-data results are functionality checks, **not** scientific validation. The paper's reported performance numbers are not embedded in the code or guaranteed.

## Highlights

- Three modality-specific multilayer perceptron (MLP) encoders, each producing a **32-dimensional** embedding.
- Three pairwise and one third-order projected interaction candidate (**4 × 32** dimensions).
- Normalized sine–cosine angle mapping into **16 local vectors**, each of dimension 2.
- Differentiable **16-site, open-boundary MPS** with bond dimension 8, a 32-dimensional output, and **2,320 tensor-core parameters**.
- **32 trainable Gaussian-RBF prototypes** for kernel-similarity refinement.
- Adaptive gated fusion, latent attention, and three-class prediction: **CN / MCI / early AD**.
- Longitudinal time-aware attention, discrete-time conversion hazards for **0–6, 6–12, and 12–24 months**, and a right-censoring-aware training loss.
- CSV/schema-based data loading, training-only preprocessing, patient-disjoint cross-validation, baseline and ablation scripts, and research explainability utilities.

The term *quantum-inspired* describes the mathematical feature representation and tensor-network operations; **no quantum computer or quantum simulator is required**.

## Project structure

```text
QADiag-ML/
├── configs/model.json
├── data/schemas/example_schema.json
├── docs/IMPLEMENTATION.md
├── scripts/
│   ├── check_install.py
│   ├── run_experiment.py
│   ├── run_baselines.py
│   ├── run_neural_baselines.py
│   ├── run_ablations.py
│   ├── run_longitudinal.py
│   └── run_scalability.py
├── src/qadiag_ml/
│   ├── model.py
│   ├── longitudinal.py
│   ├── data.py
│   ├── train.py
│   ├── baselines.py
│   ├── metrics.py
│   ├── explain.py
│   └── report.py
├── tests/test_core.py
├── requirements.txt
├── requirements-optional.txt
└── pyproject.toml
```

## Installation

**Requirements:** Python 3.10+ recommended; PyTorch, NumPy, Pandas, scikit-learn, SciPy, Matplotlib, and pytest. A CUDA-enabled GPU is optional.

```bash
# From the repository root
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
python scripts/check_install.py
python -m pytest -q
```

Some attribution/advanced evaluation functionality uses optional dependencies described in `requirements-optional.txt`.

## Quick start: synthetic validation

Use synthetic data to verify installation and execution without downloading restricted biomedical records.

```bash
# Fast diagnostic cross-validation smoke run
python scripts/run_experiment.py --synthetic --epochs 3 --folds 3 --seeds 42 --output outputs/smoke

# Classical baselines
python scripts/run_baselines.py --synthetic --out outputs/baselines.json

# Neural baseline smoke evaluation
python scripts/run_neural_baselines.py --synthetic --epochs 2 --out outputs/neural_baselines.json

# Component ablations
python scripts/run_ablations.py --epochs 3 --out outputs/ablations.json

# Longitudinal survival-model smoke run
python scripts/run_longitudinal.py --synthetic --epochs 2 --out outputs/longitudinal_smoke.json

# Scaling smoke benchmark
python scripts/run_scalability.py --dimensions 16 32 64 --out outputs/scaling.json
```

**Do not report synthetic smoke-test metrics as Alzheimer's disease diagnostic performance.**

## Research datasets

| Dataset | Intended use | Access |
| --- | --- | --- |
| Alzheimer's Disease Neuroimaging Initiative (ADNI) | Internal diagnostic classification | [Dataset access](https://ida.loni.usc.edu/) |
| TADPOLE (derived from ADNI) | Longitudinal MCI-to-AD progression | [Dataset access](https://tadpole.grand-challenge.org/Data/) |
| OASIS-3 | Independent external evaluation | [Dataset access](https://sites.wustl.edu/oasisbrains/home/oasis-3/) |

The model expects **tabular MRI/PET-derived ROI measurements**, not raw image volumes. Dataset access and redistribution may be governed by data-use agreements. No patient-level source files are included.

### Input CSV and schema

Prepare a visit-level CSV with a subject identifier, diagnosis, visit timing, scanner/site field (where available), and numeric features grouped by modality. Match the *actual* column names to a JSON schema following `data/schemas/example_schema.json`, e.g.:

```json
{
  "subject_id": "RID",
  "diagnosis": "DX",
  "visit_time": "VISIT_MONTH",
  "site": "SITE",
  "label_map": {"CN": 0, "MCI": 1, "AD": 2},
  "clinical": ["ADAS13", "MMSE", "CDRSB"],
  "imaging": ["HIPPOCAMPUS", "VENTRICLES"],
  "bio_genetic": ["ABETA", "TAU", "APOE4"]
}
```

These feature names are **illustrative placeholders**, not a universal schema for all releases. Verify variable definitions, units, missing-value encodings, diagnostic labels, and clinical eligibility before analysis. Encode categorical features using **training-only** vocabularies before providing them to the numeric preprocessor.

### Run on prepared ADNI tables

```bash
python scripts/run_experiment.py \
  --csv /path/to/adni_roi.csv \
  --schema /path/to/adni_schema.json \
  --epochs 40 \
  --folds 5 \
  --seeds 42 43 44 45 46 \
  --device cpu \
  --output outputs/adni
```

For a suitable CUDA-enabled PyTorch installation, change `--device cpu` to `--device cuda`.

Patient-level separation is necessary because repeated visits from the same participant must not span training and held-out test partitions. **TADPOLE and ADNI share participants**; cross-resource ID reconciliation and explicit held-out exclusions are required for manuscript-grade multi-dataset experiments.

## Architecture overview

```text
Clinical–Cognitive ─► MLP (32) ──┐
MRI/PET ROI ────────► MLP (32) ───┼──► Pairwise/third-order candidates (4 × 32)
Biofluid–Genetic ──► MLP (32) ────┘                 │
                                                  ▼
                                      Angle mapping (16 × 2)
                                                  │
                                                  ▼
                                     16-site MPS → 32-D vector
                                                  │
                                                  ▼
                                      RBF prototype refinement
                                                  │
                         96-D modality vector ───┤
                                                  ▼
                                     32-D adaptive gated fusion
                                                  │
                                                  ▼
                                           Latent attention
                                                  │
                                    ┌─────────────┴─────────────┐
                                    ▼                           ▼
                             3-class diagnosis          Longitudinal hazards
                              CN / MCI / AD             6 / 12 / 24 months
```

The longitudinal head is evaluated through a separate time-ordered event/censoring workflow; it is not automatically trained by the basic diagnostic CSV script.

## Experiments and evaluation

The package provides a diagnostic runner and utilities for accuracy, macro-averaged classification metrics, AUROC, calibration/ECE, explanation attribution, and stability assessment. Separate scripts support classical baselines, neural baseline smoke runs, component ablations, and computational scaling checks.

**Important scope boundaries:**

1. The included CV runner uses patient-disjoint outer folds with an internal training-derived validation split; **full nested hyperparameter search and final outer-train refitting are not automated**.
2. Site/scanner adjustment, demographic regression, genetic categorical encoding, label eligibility, and matching feature spaces across independent cohorts require dataset-specific verification and, in some cases, additional code.
3. TADPOLE longitudinal landmark construction and full censoring-aware external metrics (including validated IPCW estimates) are **not a turnkey published evaluation pipeline**.
4. Neural baselines, SHAP/Integrated Gradients utilities, and explanation stability methods are included, but complete comparable, publication-ready runs need explicit configurations and validation.
5. The scalability runner uses Python allocation-memory measurements; **this is not a GPU peak-memory benchmark**.
6. Claims such as the manuscript's **91.8% accuracy, 0.96 AUROC, and OASIS-3 external performance** must be independently reproduced using authenticated records and documented partitions.

For publication-quality evaluation, retain per-subject split manifests (without exposing patient identifiers), software and hardware versions, preprocessing-fit boundaries, training seeds, model checkpoints, fold-wise predictions, uncertainty intervals, and paired comparison logs.

## Reproducibility and testing

```bash
python -m pytest -q
```

The core tests cover model operations, tensor/network properties, preprocessing and longitudinal behavior. Passing unit tests verifies engineering checks, **not clinical generalization**. Keep datasets and patient-linked artifacts outside Git; use the included `.gitignore` and follow the source datasets' use agreements.

## Citation

If using this implementation in research, cite the associated paper **once its final bibliographic details or DOI are available**:

> *QADiag-ML: Capturing Complex Biomarker Interactions for Early Alzheimer’s Disease Diagnosis Using Quantum-Inspired Machine Learning.*

Do not fabricate a publication venue, DOI, or author list before these details are confirmed.

## Responsible use

QADiag-ML is intended for reproducible machine-learning research. Its predictions and model explanations must not be used as standalone clinical diagnoses, treatment recommendations, or evidence of causal biomarker relationships.
