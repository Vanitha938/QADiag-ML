# QADiag-ML — Research Implementation

Classical PyTorch implementation of the architecture in **QADiag-ML: Capturing Complex Biomarker Interactions for Early Alzheimer’s Disease Diagnosis Using Quantum-Inspired Machine Learning**.

## Quick start

```bash
python -m pip install -e .
pytest -q
python scripts/run_experiment.py --synthetic --epochs 3 --folds 3 --seeds 42 --output outputs/smoke
python scripts/run_baselines.py --synthetic
python scripts/run_ablations.py --epochs 3
python scripts/run_scalability.py --dimensions 16 32
```

Synthetic data are **only** for function/integration testing. Synthetic metrics are **not** evidence supporting the paper's results.

## Real diagnostic dataset

1. Obtain authorized ADNI data from https://ida.loni.usc.edu/. For OASIS-3 see https://sites.wustl.edu/oasisbrains/home/oasis-3/; TADPOLE: https://tadpole.grand-challenge.org/Data/.
2. Prepare a numeric ROI-level CSV, one record per visit, with subject ID, diagnostic label and features. Define actual available column names in a JSON schema like `data/schemas/example_schema.json`. The example names are illustrative and MUST be adapted; not all features are released in every cohort.
3. Run:

```bash
python scripts/run_experiment.py --csv /path/to/adni_roi.csv --schema /path/to/adni_schema.json --epochs 40 --output outputs/adni --device cpu
```

Use `--device cuda` on an installed CUDA-enabled PyTorch platform. Output includes seed/fold metrics and learning curves. Always keep raw patient datasets outside version control.

## Main architecture

Three 32-dimensional modality-specific encoders; projected cross-stream CI/CB/IB/CIB products; 4 coordinates per candidate mapped to 16 normalized 2-vectors; open-boundary 16-site MPS (bond=8; 2,320 core parameters); 32 reference-prototype Gaussian-RBF interaction refinement; 32-dimensional adaptive gated fusion/attention; 3-class diagnostic head. Longitudinal time-aware masked attention with conditional 0–6, 6–12 and 12–24 month hazard head and right-censoring loss in `model.py` and `longitudinal.py`.

## Module coverage and limitations

- Provided: core neural model; CSV dataset schema adapters; baseline-index-visit selection; subject-disjoint validation; training-only iterative imputation, standardized continuous features, missingness masks, site location-scale normalization; CV training; basic sklearn baselines and configurable neural transformer/graph/late fusion architectures; architecture ablations; MPS parameter verification; gradient-input, candidate-level IG and SHAP utilities; survival loss and longitudinal collation/training primitives; classification metrics, calibration/ECE, bootstrapping, stability, scalability smoke benchmark.
- Dataset-dependent: cohort eligibility adjudication, matching subject IDs across ADNI/TADPOLE, clinically justified label harmonization, site/scanner feature metadata, longitudinal landmark construction and censoring distribution estimation. These cannot be inferred from the manuscript alone. **The default CSV runner does not claim to reproduce verified clinical cohort counts or complete nested model selection.**
- `cross_validate` uses a patient-disjoint outer fold and a train-derived validation split; the paper's full nested hyperparameter search and training refit are not automatically performed. Use Optuna and fixed inner splits for manuscript-grade claims.
- `run_baselines.py` executes four classical baselines. Neural baselines are instantiated in `baselines.py` but require a matched neural training protocol for publication comparisons.
- `run_scalability.py` reports Python allocation memory, not actual GPU/process peak RAM; do not interpret it as the memory claim in the paper.
- The preprocessor expects continuous, numeric biomarkers. Categorical demographic/genetic variables must be encoded using train-only category vocabularies upstream; regression adjustment and feature-matched external validation require cohort-specific validated implementations.
- SHAP candidate explainer is provided as a research utility, can be expensive and requires care with backgrounds. Default clinical evaluation does not run it.
- No reported paper result (including the stated 91.8% accuracy) is embedded or guaranteed. All metrics must be independently reproduced on authorized, inspected data.

## Ethics and scope

Not a clinical diagnostic tool. No treatment advice. ADNI, TADPOLE, and OASIS-3 may require access agreements; no participant-level data are distributed.

## Additional executable entry points

```bash
python scripts/run_neural_baselines.py --synthetic --epochs 2
python scripts/run_longitudinal.py --synthetic --epochs 2
```

The latter checks real differentiable longitudinal training on synthetic event/censoring labels, but does not claim verified IPCW evaluation. The MPS uses intermediate LayerNorm for stable backpropagation; this is a numerical modeling choice rather than a literal unnormalized tensor-network contraction. All scripts are research software, not a deployable clinical product.
