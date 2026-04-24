"""Calibration and uncertainty-quality utilities for surrogate evaluation.

Provides:
  - Regression calibration: multi-level coverage, sharpness, NLL
  - Classification calibration: ECE, reliability curve, Brier score
  - Sampling-suitability diagnostics for Thompson-style BO
  - Structured split generators for extrapolation stress tests
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Regression calibration                                             #
# ------------------------------------------------------------------ #

COVERAGE_LEVELS: list[float] = [0.50, 0.80, 0.90, 0.95]


def coverage_at_level(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    level: float,
) -> float:
    """Empirical coverage at a nominal confidence level.

    Args:
        y_true: True values, shape (n,).
        mu: Predicted means, shape (n,).
        sigma: Predicted std devs, shape (n,).
        level: Nominal coverage probability (e.g., 0.90).

    Returns:
        Fraction of points inside the interval.
    """
    z = norm.ppf(0.5 + level / 2.0)
    lo = mu - z * sigma
    hi = mu + z * sigma
    inside = (y_true >= lo) & (y_true <= hi)
    return float(inside.mean())


def multi_level_coverage(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    levels: list[float] = COVERAGE_LEVELS,
) -> dict[float, float]:
    """Compute empirical coverage at multiple nominal levels.

    Args:
        y_true: True values.
        mu: Predicted means.
        sigma: Predicted stds.
        levels: Nominal probability levels to check.

    Returns:
        Dict mapping nominal → empirical coverage.
    """
    return {
        lv: coverage_at_level(y_true, mu, sigma, lv)
        for lv in levels
    }


def calibration_slope(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute calibration slope and intercept from reliability curve.

    Bins predicted confidence levels; fits a linear regression of
    empirical coverage on nominal coverage. Slope=1, intercept=0
    indicates perfect calibration.

    Args:
        y_true: True values.
        mu: Predicted means.
        sigma: Predicted stds.
        n_bins: Number of quantile bins.

    Returns:
        Dict with 'slope', 'intercept', 'nominal', 'empirical' arrays.
    """
    levels = np.linspace(0.05, 0.95, n_bins)
    empirical = np.array([
        coverage_at_level(y_true, mu, sigma, lv) for lv in levels
    ])
    # Linear fit: empirical ~ slope * nominal + intercept
    A = np.column_stack([levels, np.ones_like(levels)])
    result = np.linalg.lstsq(A, empirical, rcond=None)
    slope, intercept = result[0]
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "nominal": levels.tolist(),
        "empirical": empirical.tolist(),
    }


def sharpness(sigma: np.ndarray) -> float:
    """Average predicted std (lower = sharper).

    Args:
        sigma: Predicted stds, shape (n,).

    Returns:
        Mean std value.
    """
    return float(np.mean(sigma))


def nll_gaussian(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Mean Gaussian NLL.

    Args:
        y_true: True values.
        mu: Predicted means.
        sigma: Predicted stds.

    Returns:
        Mean NLL.
    """
    sigma = np.maximum(sigma, 1e-8)
    return float(np.mean(-norm.logpdf(y_true, loc=mu, scale=sigma)))


def regression_calibration_report(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, Any]:
    """Full calibration report for a regression surrogate.

    Args:
        y_true: True values, shape (n,).
        mu: Predicted means, shape (n,).
        sigma: Predicted stds, shape (n,).

    Returns:
        Dict with NLL, coverage at multiple levels, sharpness,
        calibration slope.
    """
    cov = multi_level_coverage(y_true, mu, sigma)
    cal = calibration_slope(y_true, mu, sigma)
    return {
        "nll": nll_gaussian(y_true, mu, sigma),
        "sharpness": sharpness(sigma),
        "coverage": {f"cov_{int(100*k)}": v for k, v in cov.items()},
        "calibration_slope": cal["slope"],
        "calibration_intercept": cal["intercept"],
        "calibration_nominal": cal["nominal"],
        "calibration_empirical": cal["empirical"],
    }


# ------------------------------------------------------------------ #
#  Classification calibration                                         #
# ------------------------------------------------------------------ #

def expected_calibration_error(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Args:
        y_true: Binary labels, shape (n,).
        p_pred: Predicted probabilities (class 1), shape (n,).
        n_bins: Number of equal-width bins.

    Returns:
        ECE scalar.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(p_pred[mask].mean())
        ece += mask.sum() / n * abs(acc - conf)
    return ece


def reliability_curve(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Compute reliability curve data.

    Args:
        y_true: Binary labels.
        p_pred: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        Dict with 'mean_predicted_prob' and 'fraction_of_positives'.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred = []
    frac_pos = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        mean_pred.append(float(p_pred[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
    return {
        "mean_predicted_prob": mean_pred,
        "fraction_of_positives": frac_pos,
    }


def classification_calibration_report(
    y_true: np.ndarray,
    p_pred: np.ndarray,
) -> dict[str, Any]:
    """Full calibration report for a Phase_Sep classifier.

    Args:
        y_true: Binary labels.
        p_pred: Predicted P(phase_sep=1), shape (n,).

    Returns:
        Dict with log_loss, brier_score, ECE, AUC, reliability curve.
    """
    p_pred = np.clip(p_pred, 1e-8, 1 - 1e-8)
    labels = y_true.astype(int)

    try:
        auc = roc_auc_score(labels, p_pred)
    except Exception:
        auc = float("nan")

    return {
        "log_loss": log_loss(labels, p_pred),
        "brier_score": brier_score_loss(labels, p_pred),
        "ece": expected_calibration_error(labels, p_pred),
        "auc": auc,
        "reliability_curve": reliability_curve(labels, p_pred),
    }


# ------------------------------------------------------------------ #
#  Structured split generators                                        #
# ------------------------------------------------------------------ #

def boundary_holdout_splits(
    X: np.ndarray,
    cont_cols: list[int],
    quantile: float = 0.15,
    n_folds: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate splits where the test set lives at the boundary of
    continuous features (extrapolation stress test).

    Creates alternating high-boundary and low-boundary test sets.

    Args:
        X: Input array, shape (n, d).
        cont_cols: Indices of continuous (scaled) columns.
        quantile: Fraction of range treated as boundary.
        n_folds: Number of splits (alternating high/low).

    Returns:
        List of (train_idx, test_idx) tuples.
    """
    n = len(X)
    splits = []
    for fold in range(n_folds):
        # Alternate between high and low extremes of each cont dim
        col = cont_cols[fold % len(cont_cols)]
        vals = X[:, col]
        if fold % 2 == 0:
            # Test on high-extreme points
            threshold = np.quantile(vals, 1.0 - quantile)
            test_mask = vals >= threshold
        else:
            # Test on low-extreme points
            threshold = np.quantile(vals, quantile)
            test_mask = vals <= threshold

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]

        if len(test_idx) == 0 or len(train_idx) < 5:
            continue
        splits.append((train_idx, test_idx))

    return splits


def sparse_region_holdout_splits(
    X: np.ndarray,
    n_folds: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out points in the sparse (low-density) regions of the space.

    Uses distance to nearest neighbor as a density proxy.
    Points far from their neighbors are "sparse."

    Args:
        X: Input array, shape (n, d).
        n_folds: Number of splits.

    Returns:
        List of (train_idx, test_idx) tuples.
    """
    from sklearn.metrics import pairwise_distances

    n = len(X)
    dists = pairwise_distances(X)
    np.fill_diagonal(dists, np.inf)
    nn_dist = dists.min(axis=1)  # nearest-neighbor distance
    # Rank by sparsity descending
    order = np.argsort(nn_dist)[::-1]
    splits = []
    fold_size = max(1, n // (n_folds * 3))
    for fold in range(n_folds):
        test_idx = order[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.array(
            [i for i in range(n) if i not in set(test_idx)],
        )
        if len(test_idx) == 0 or len(train_idx) < 5:
            continue
        splits.append((train_idx, test_idx))
    return splits


def category_combo_holdout_splits(
    X: np.ndarray,
    cat_cols: list[int],
    n_combos: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out specific category combinations (leave-combo-out).

    Args:
        X: Input array (one-hot or integer encoded).
        cat_cols: Columns that represent categories.
        n_combos: Number of combos to hold out.

    Returns:
        List of (train_idx, test_idx) tuples.
    """
    from collections import Counter

    n = len(X)
    # Build combo ID for each row
    combos = [tuple(X[i, cat_cols].astype(int).tolist()) for i in range(n)]
    counts = Counter(combos)
    # Pick combos with enough support
    sorted_combos = sorted(
        counts.keys(), key=lambda c: counts[c], reverse=True,
    )
    splits = []
    for combo in sorted_combos[:n_combos]:
        test_idx = np.array(
            [i for i, c in enumerate(combos) if c == combo],
        )
        train_idx = np.array(
            [i for i, c in enumerate(combos) if c != combo],
        )
        if len(test_idx) < 2 or len(train_idx) < 5:
            continue
        splits.append((train_idx, test_idx))
    return splits


# ------------------------------------------------------------------ #
#  Sampling-suitability diagnostics                                   #
# ------------------------------------------------------------------ #

def sampling_suitability_check(
    model: Any,
    X_train: np.ndarray,
    X_sparse: np.ndarray,
    X_dense: np.ndarray,
    n_samples: int = 200,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Lightweight sampling suitability diagnostic for Thompson BO.

    Checks whether:
    1. Uncertainty grows in sparse vs dense regions.
    2. Posterior samples span reasonable ranges.
    3. Std does not collapse to near-zero everywhere.

    Args:
        model: Fitted surrogate with predict() and predict_std().
        X_train: Training inputs for density reference.
        X_sparse: Candidate points in sparse/edge regions.
        X_dense: Candidate points in dense training regions.
        n_samples: Number of posterior sample draws.
        rng: NumPy random generator.

    Returns:
        Dict with diagnostics per region.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    results = {}

    for region_name, X_region in [
        ("sparse", X_sparse), ("dense", X_dense),
    ]:
        if len(X_region) == 0:
            continue
        mu = model.predict(X_region)
        sigma = model.predict_std(X_region)

        # Sample from predictive distribution
        z = rng.standard_normal((n_samples, len(X_region), mu.shape[1]))
        samples = mu[None] + z * sigma[None]  # (n_samples, n, 6)

        mean_std = float(np.nanmean(sigma))
        min_std = float(np.nanmin(sigma))
        max_std = float(np.nanmax(sigma))
        sample_range = float(
            np.nanmean(samples.max(axis=0) - samples.min(axis=0))
        )
        near_zero_std_frac = float(
            np.mean(sigma < 1e-4)
        )

        results[region_name] = {
            "mean_std": mean_std,
            "min_std": min_std,
            "max_std": max_std,
            "sample_range_mean": sample_range,
            "near_zero_std_fraction": near_zero_std_frac,
        }

    # Sparsity check: sparse std > dense std?
    if "sparse" in results and "dense" in results:
        results["std_ratio_sparse_over_dense"] = (
            results["sparse"]["mean_std"]
            / max(results["dense"]["mean_std"], 1e-12)
        )
        results["uncertainty_grows_in_sparse"] = (
            results["std_ratio_sparse_over_dense"] > 1.05
        )

    return results
