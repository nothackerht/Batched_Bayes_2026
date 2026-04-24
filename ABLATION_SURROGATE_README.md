# Feature-Ablation Surrogate Evaluation Framework

## Overview

This framework extends the rigorous surrogate evaluation pipeline to test
**multiple feature schemas** for each target. The original framework asked
*which model family is best*. This ablation framework asks two questions at once:

1. Which **surrogate model** is best for each target?
2. Which **feature representation** is best for each target?

The motivation is to determine whether secondary formulation descriptors
(HLB values, pairwise compatibility, solubility) improve surrogate quality
in a way that matters for **uncertainty-aware Bayesian optimization**, not just
ordinary point prediction.

---

## Why This Was Built

The original `rigorous_surrogate_eval.py` evaluated all models on a single
feature representation. In practice that representation silently used zero
descriptors because the HLB column name in the CSV (`HLB_Value`) did not
match what the loader expected (`HLB`). So the baseline was effectively
one-hot categoricals + scaled continuous variables.

The ablation framework was created to:

1. Fix that representation issue with a clean, schema-aware feature builder
2. Explicitly test five schemas and compare their effect on surrogate quality
3. Answer whether adding HLB, compatibility, and solubility descriptors
   is worth the added complexity in the BO pipeline

The original `rigorous_surrogate_eval.py` was left **unchanged**. All new
functionality is in the companion files described below.

---

## Files

| File | Purpose |
|------|---------|
| `feature_builder.py` | Builds feature matrices for each schema — no side effects, no file writes |
| `rigorous_surrogate_eval_ablation.py` | Ablation runner — loops schemas, evaluates targets, exports JSON |
| `plot_ablation_results.py` | Cross-schema comparison plots (6 plot types) |
| `ablation_results_guide.md` | How to read the JSON output and interpret the plots |

The original framework files are untouched:

| File | Purpose |
|------|---------|
| `rigorous_surrogate_eval.py` | Original single-schema evaluation |
| `calibration_utils.py` | Calibration metrics (shared by both frameworks) |
| `model_builders.py` | Candidate model builders (shared by both frameworks) |
| `sampling_diagnostics.py` | Thompson-BO suitability diagnostics |
| `plot_surrogate_results.py` | Original single-schema visualization |

---

## Feature Schemas

All five schemas share the same target masking, CV logic, candidate families,
and ranking metrics as the original framework. They differ only in which
feature columns are passed to the models.

### Layout by schema

```
base:                      [cat (17) | cont (3)]                           = 20 features
base_hlb:                  [cat (17) | HLB (3) | cont (3)]                = 23 features
base_compat:               [cat (17) | compat (4) | cont (3)]             = 24 features
base_hlb_compat:           [cat (17) | HLB (3) | compat (4) | cont (3)]  = 27 features
base_hlb_compat_solubility:[cat (17) | HLB (3) | compat (4) | sol (5) | cont (3)] = 32 features
```

### Categorical block (17 columns)
One-hot encoded: Oil (5), Surfactant (4), Cosurfactant (5), API_Name (3).

### Continuous block (3 columns)
Min-max scaled using known domain ranges:
- `Oil_V` → [7.5, 22.5]
- `S_Ratio` → [3.0, 9.0]
- `Sonication` → [0.0, 3.0]

### HLB descriptors (3 columns)
`HLB_Oil`, `HLB_Surfactant`, `HLB_Cosurfactant` — scaled min-max across
all entries in `hlb_values.csv`.

**Note:** Lookup is by component name only, ignoring the `Type` column.
`PEG_400` and `Tween_80` appear in dual roles (surfactant and cosurfactant)
across different rows; name-only lookup returns the correct HLB value
regardless of the role in a given row.

### Compatibility descriptors (4 columns)

| Column | Source | Values |
|--------|--------|--------|
| `compat_oil_surf` | `oil_surfactant_compatibility.csv` | 0 / 1 |
| `compat_oil_cosurf` | `oil_cosurfactant_compatibility.csv` | 0 / 1 |
| `compat_surf_cosurf` | `surfactant_cosurfactant_compatibility.csv` | 0 / 1 / **-1** |
| `compat_surf_cosurf_missing` | indicator for NaN entries in the above | 0 / 1 |

The surfactant–cosurfactant matrix has genuine NaN entries where no
compatibility measurement was reported. These are encoded as `-1` (distinct
from `0` = incompatible) plus a separate binary missing-indicator column so
the model can condition on whether data exists.

### Solubility descriptors (5 columns)

| Column | Description |
|--------|-------------|
| `sol_oil` | Scaled solubility of the Oil component |
| `sol_surf` | Scaled solubility of the Surfactant (0 if missing) |
| `sol_surf_missing` | 1 if surfactant solubility not in file |
| `sol_cosurf` | Scaled solubility of the Cosurfactant |
| `sol_diff_oil_surf` | `|sol_oil - sol_surf|` — miscibility proxy (0 when surf is missing) |

**Note:** `PEG_400` appears in `solubility_values.csv` only as a Cosurfactant
entry. When used as a Surfactant in a row it has no solubility entry, giving
`sol_surf = 0`, `sol_surf_missing = 1`. Approximately 13% of rows trigger this.

---

## Target Masking

Identical to `rigorous_surrogate_eval.py`:

| Target | Rows used |
|--------|-----------|
| `Phase_Sep` | All 147 rows |
| `Droplet_Size`, `PDI`, `Zeta_P` | Rows where `Phase_Sep == 0` (~108 rows) |
| `Drug_Loading`, `Permeability` | Non-NaN rows (~6 rows each) |

---

## Ranking Metrics

Identical to the original framework — BO-relevant probabilistic quality, not RMSE:

| Task | Primary metric |
|------|---------------|
| Classification (`Phase_Sep`) | `log_loss` (lower = better) |
| Regression (all other targets) | mean NLL (lower = better) |

RMSE, MAE, R², multi-level coverage (50/80/90/95%), and calibration slope
are reported as secondary diagnostics.

---

## Model Candidates

All candidate families from `model_builders.py` are used unchanged.

### Classification (Phase_Sep)
GPC_Matern25, GPC_Matern15, GPC_RBF, GPC_ARD_Tuned (Optuna-tuned),
RF_Calibrated, RF_Uncalibrated, GBT_Calibrated, GBT_Uncalibrated, SVM_Platt.

### Regression (continuous targets)
GPR_Matern25_White, GPR_Matern15_White, GPR_RBF_White, GPR_RatQuad_White,
GPR_Matern25_Scaled, GPR_Matern15_Scaled, GPR_Composite_M25_RBF,
GPR_ARD_Tuned (Optuna-tuned), GPyTorch_ExactGP, QuantileRF, MLPEnsemble.

---

## Selection Rule

Same as the original framework:

1. **Reject** models with pathological uncertainty:
   - NLL not finite
   - `sharpness < 1e-6` (std collapsed everywhere)
   - Empirical 90% coverage below 30%

2. Among acceptable models, **rank by** log-loss (classification) or NLL (regression).

3. The ablation then compares winners **across schemas** to identify the best
   feature representation per target.

---

## Output JSON Structure

```json
{
  "metadata": {
    "feature_schemas_tested": ["base", "base_hlb", ...],
    "targets_evaluated": ["Phase_Sep", "Droplet_Size", ...],
    "n_optuna_trials": 100,
    "cv_method": "kfold",
    "n_folds": 5
  },
  "schema_results": {
    "base": {
      "n_features": 20,
      "feature_names": [...],
      "targets": {
        "Phase_Sep": { "winner": "...", "winner_score": 0.093, ... },
        "Droplet_Size": { ... }
      },
      "winner_summary": { ... }
    },
    "base_hlb": { ... }
  },
  "ablation_comparison": {
    "Phase_Sep": {
      "metric": "log_loss",
      "schema_scores": { "base": 0.42, "base_hlb": 0.38, ... },
      "schema_winners": { "base": "GPC_RBF", ... },
      "best_schema": "base_hlb_compat",
      "best_score": 0.31
    },
    "Droplet_Size": { ... }
  },
  "best_per_target": {
    "Phase_Sep": {
      "best_schema": "base_hlb_compat",
      "best_score": 0.31,
      "best_winner": "GPC_ARD_Tuned",
      "metric": "log_loss"
    }
  }
}
```

**Start reading from `best_per_target`** — it is the clean summary.
`ablation_comparison` gives the full cross-schema breakdown.
`schema_results` stores the complete candidate-level reports per schema.

---

## Plots Generated

| File | Shows |
|------|-------|
| `ablation_heatmap_scores.png` | Score matrix: schemas (rows) × targets (cols) |
| `ablation_heatmap_delta.png` | Δ vs `base` schema — negative = improvement |
| `ablation_<target>_bars.png` | Candidate model scores under each schema (per target) |
| `ablation_<target>_calibration.png` | Calibration curves for best model per schema (regression only) |
| `ablation_Phase_Sep_reliability.png` | Reliability diagram for best classifier per schema |
| `ablation_best_schema_summary.png` | Best score + schema win counts across all targets |

See `ablation_results_guide.md` for a full interpretation guide including
how to use each plot to answer specific BO-relevant questions.

---

## Usage

Run from the `surrogate_models/` directory.

```bash
# Fast test — 2 schemas, 2 targets, 30 trials
python rigorous_surrogate_eval_ablation.py \
    --data_path ../data/MicroemulsionFormulation.csv \
    --feature_schemas base base_hlb \
    --targets Phase_Sep Droplet_Size \
    --n_trials 30 --fast \
    --output results/ablation_fast.json

# Full run — all 5 schemas, all 6 targets
python rigorous_surrogate_eval_ablation.py \
    --data_path ../data/MicroemulsionFormulation.csv \
    --n_trials 100 \
    --output results/ablation_results.json

# Specific schemas only
python rigorous_surrogate_eval_ablation.py \
    --data_path ../data/MicroemulsionFormulation.csv \
    --feature_schemas base base_hlb_compat base_hlb_compat_solubility \
    --n_trials 100 \
    --output results/ablation_results.json

# Generate comparison plots
python plot_ablation_results.py \
    --results_json results/ablation_results.json \
    --output_dir results/plots/ablation/

# Original framework — still works unchanged
python rigorous_surrogate_eval.py \
    --data_path ../data/MicroemulsionFormulation.csv \
    --n_trials 100 \
    --output results/rigorous_surrogate_results.json
```

Optional flags for the ablation runner:

| Flag | Default | Effect |
|------|---------|--------|
| `--fast` | off | Skip structured splits; cap Optuna at 30 trials |
| `--no_gpytorch` | off | Skip GPyTorch ExactGP evaluation |
| `--cv_method` | `kfold` | `kfold` or `loo` |
| `--n_folds` | `5` | Number of folds |
| `--seed` | `42` | Random seed |
| `--feature_schemas` | all five | Space-separated schema names |
| `--targets` | all six | Space-separated target names |

---

## Questions This Ablation Is Designed to Answer

1. Do secondary formulation descriptors improve `Phase_Sep` classification?
2. Do they help continuous regression targets (droplet size, PDI, zeta potential)?
3. Which feature schema is best for each target?
4. Do compatibility descriptors especially help generalization (combo holdout splits)?
5. Does adding these descriptors improve **uncertainty calibration** or only point prediction?

If a descriptor-rich schema reduces NLL and improves calibration shape without
simply overfitting the mean, that is the strongest evidence the added features
are genuinely BO-useful.

---

## Known Limitations

### Drug_Loading and Permeability
Only ~6 rows available. LOO-CV is used automatically. Results are directional
only — do not treat schema differences for these targets as statistically
meaningful without further validation.

### Surfactant–cosurfactant NaN entries
The compatibility matrix has structural gaps (e.g., Labrasol–Tween_20 missing).
These are encoded conservatively as `-1` + missing indicator. If experimental
data for those pairs becomes available, update `surfactant_cosurfactant_compatibility.csv`
and re-run.

### PEG_400 solubility as surfactant
Not present in `solubility_values.csv` in the surfactant role. Approximately
13% of rows receive `sol_surf = 0`, `sol_surf_missing = 1`. This is a known
data gap, not a code error.

### QuantileRF and MLPEnsemble
Neither provides a true Gaussian posterior and both are consistently rejected
by the calibration filter on this dataset. They are included as diagnostic
baselines only.
