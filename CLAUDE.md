# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Surrogate model evaluation framework for Bayesian Optimization (BO) of pharmaceutical microemulsion formulations. The goal is to identify which feature schema (baseline vs. enriched with HLB/compatibility/solubility descriptors) yields the best **uncertainty-calibrated** surrogate for BO. Primary metric throughout is NLL (regression) or log-loss (classification), not point accuracy.

## Setup

No requirements.txt. Install dependencies manually:

```bash
pip install numpy pandas scipy scikit-learn
pip install quantile-forest   # optional — enables QuantileRF models
pip install optuna             # optional — hyperparameter tuning
```

## Running

The three utility modules are imported, not run directly. The top-level experiment script (referenced in docs but not in this directory) is typically invoked as:

```bash
python rigorous_surrogate_eval_ablation.py \
    --data_path data/MicroemulsionFormulation_new.csv \
    --feature_schemas base base_hlb base_compat base_hlb_compat base_hlb_compat_solubility \
    --targets Phase_Sep Droplet_Size PDI Zeta_P \
    --n_trials 30 \
    --output results/ablation.json
```

Add `--fast` to reduce CV folds for quick iteration.

## Architecture

### Three core modules (no interdependencies)

**`feature_builder.py` — `build_features(data_path, schema, data_dir)`**  
Returns `(X, y_dict, meta)`. Supports 5 schemas of increasing complexity:

| Schema | Features | Adds |
|--------|----------|------|
| `base` | 21 | One-hot categoricals + min-max scaled continuous |
| `base_hlb` | 24 | HLB descriptors |
| `base_compat` | 25 | Pairwise binary compatibility flags |
| `base_hlb_compat` | 28 | Both HLB + compatibility |
| `base_hlb_compat_solubility` | 33 | + solubility/miscibility indicators |

Continuous features are scaled to fixed ranges defined in `_CONT_RANGES` (not fit-on-train), so scaling is deterministic across splits.

**`model_builders.py` — `build_classifier_candidates(d, seed)` / `build_regressor_candidates(d, seed)`**  
Returns a list of sklearn-compatible estimators, each wrapping uncertainty quantification:
- Classification (9 candidates): GPC (3 kernels), RF variants, GBT variants, SVM
- Regression (11+ candidates): GPR (7 kernel configs), QuantileRF, MLPEnsemble

**`calibration_utils.py` — `regression_calibration_report(y_true, mu, sigma)` / `classification_calibration_report(...)`**  
Computes NLL, RMSE, R², multi-level coverage (50/80/90/95%), calibration slope, and BO suitability metrics. Models are rejected if NLL=NaN, std collapses, or 90% coverage < 30%.

### Target masking

The 6 output targets use different row subsets:
- `Phase_Sep` (binary classification): all 147 rows
- `Droplet_Size`, `PDI`, `Zeta_P` (regression): rows where `Phase_Sep == 0` (~108 rows)
- `Drug_Loading`, `Permeability` (regression): non-NaN rows (~6 rows each — very sparse)

### Data quirks

- **PEG_400 dual role**: appears as both `Surfactant` and `Cosurfactant`; handled with dedicated indicator columns in compatibility features
- **Surfactant–cosurfactant compatibility matrix**: contains NaN entries (not missing data — unknown interactions); NaN gets its own indicator column to distinguish from "incompatible"
- **Drug_Loading / Permeability**: ~6 samples each — treat evaluation results for these targets as illustrative only

### Results layout

- `results/rigorous_phase12_plots/` — per-target calibration and ranking plots
- `results/ablation_all5/` — cross-schema comparison heatmaps and delta plots
- `results/rankings_*.csv` — model rankings per target per schema
- `results/rigorous_surrogate_results_*.json` — full numeric results

## Key design decisions

- **NLL is the primary ranking metric** — not RMSE/accuracy — because downstream BO acquisition functions require well-calibrated uncertainty, not just accurate means
- Structured holdout splits (boundary, sparse region, category combo) are used alongside k-fold to probe extrapolation, not just interpolation
- Missing value indicators are intentional features, not preprocessing artifacts
