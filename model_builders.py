"""Candidate model builders for surrogate selection shootout.

Each builder returns either:
  - A classification candidate dict (for Phase_Sep)
  - A regression candidate dict (for continuous targets)

Regression models must provide predict(X) and predict_std(X) or
return_std=True via sklearn GPR convention.

Non-GP uncertainty is explicitly flagged as approximate:
  - QuantileRF: interval-based pseudo-std
  - MLPEnsemble: disagreement-based pseudo-std
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Phase 1 classification candidates                                  #
# ------------------------------------------------------------------ #

def build_classifier_candidates(
    d: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return list of classification candidate configs for Phase_Sep.

    Args:
        d: Input dimensionality.
        seed: Random seed.

    Returns:
        List of dicts with 'name', 'build_fn', 'uncertainty_type'.
    """
    candidates = []

    # GPC Matern 2.5
    candidates.append({
        "name": "GPC_Matern25",
        "uncertainty_type": "gp_posterior",
        "build_fn": lambda: GaussianProcessClassifier(
            kernel=Matern(nu=2.5),
            random_state=seed,
            max_iter_predict=200,
        ),
    })

    # GPC Matern 1.5
    candidates.append({
        "name": "GPC_Matern15",
        "uncertainty_type": "gp_posterior",
        "build_fn": lambda: GaussianProcessClassifier(
            kernel=Matern(nu=1.5),
            random_state=seed,
            max_iter_predict=200,
        ),
    })

    # GPC RBF
    candidates.append({
        "name": "GPC_RBF",
        "uncertainty_type": "gp_posterior",
        "build_fn": lambda: GaussianProcessClassifier(
            kernel=RBF(),
            random_state=seed,
            max_iter_predict=200,
        ),
    })

    # Random Forest with Platt calibration
    candidates.append({
        "name": "RF_Calibrated",
        "uncertainty_type": "calibrated_ensemble",
        "build_fn": lambda: CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=seed,
            ),
            cv=3,
            method="isotonic",
        ),
    })

    # Random Forest uncalibrated (for comparison)
    candidates.append({
        "name": "RF_Uncalibrated",
        "uncertainty_type": "ensemble",
        "build_fn": lambda: RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
        ),
    })

    # Gradient Boosting with Platt calibration
    candidates.append({
        "name": "GBT_Calibrated",
        "uncertainty_type": "calibrated_ensemble",
        "build_fn": lambda: CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=seed,
            ),
            cv=3,
            method="isotonic",
        ),
    })

    # Gradient Boosting uncalibrated
    candidates.append({
        "name": "GBT_Uncalibrated",
        "uncertainty_type": "ensemble",
        "build_fn": lambda: GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=seed,
        ),
    })

    # SVM with Platt scaling (probability quality explicitly evaluated)
    candidates.append({
        "name": "SVM_Platt",
        "uncertainty_type": "platt_scaling",
        "build_fn": lambda: SVC(
            kernel="rbf",
            probability=True,
            random_state=seed,
            C=1.0,
            gamma="scale",
        ),
    })

    return candidates


# ------------------------------------------------------------------ #
#  Phase 2 regression candidates                                      #
# ------------------------------------------------------------------ #

class QuantileRFRegressor:
    """Quantile Random Forest wrapper providing pseudo-std from IQR.

    Uncertainty is interval-based, not a true Gaussian posterior.
    Pseudo-std = (Q95 - Q5) / (2 * 1.645).

    This is an approximation — document this clearly.
    """

    UNCERTAINTY_TYPE = "interval_approx"

    def __init__(self, n_estimators: int = 300, seed: int = 42) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantileRFRegressor:
        try:
            from quantile_forest import RandomForestQuantileRegressor
            self._model = RandomForestQuantileRegressor(
                n_estimators=self.n_estimators,
                random_state=self.seed,
            )
        except ImportError:
            logger.warning(
                "quantile-forest not installed; falling back to sklearn RF."
            )
            self._model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.seed,
            )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self._model, 'predict'):
            return self._model.predict(X)
        return self._model.predict(X, quantiles=0.5)

    def predict_with_std(
        self, X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, pseudo_std) from quantile intervals."""
        if hasattr(self._model, 'predict') and not hasattr(
            self._model, '_estimators',
        ):
            # Fallback sklearn RF: use std over trees
            preds = np.array([
                t.predict(X)
                for t in self._model.estimators_
            ])
            mu = preds.mean(axis=0)
            std = preds.std(axis=0)
            return mu, std

        # True Quantile RF: use Q5/Q95 for pseudo-std
        q5 = self._model.predict(X, quantiles=0.05)
        q50 = self._model.predict(X, quantiles=0.50)
        q95 = self._model.predict(X, quantiles=0.95)
        pseudo_std = (q95 - q5) / (2.0 * 1.645)
        pseudo_std = np.maximum(pseudo_std, 1e-8)
        return q50, pseudo_std


class MLPEnsembleRegressor:
    """Ensemble of MLPs; uncertainty from member disagreement.

    Uncertainty is ensemble-disagreement-based, not a Bayesian posterior.
    std = std of member predictions (inter-member spread).
    """

    UNCERTAINTY_TYPE = "ensemble_disagreement"

    def __init__(
        self,
        n_members: int = 10,
        hidden_layer_sizes: tuple = (64, 64),
        max_iter: int = 500,
        seed: int = 42,
    ) -> None:
        self.n_members = n_members
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.seed = seed
        self._members: list[MLPRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPEnsembleRegressor:
        self._members = []
        for i in range(self.n_members):
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=self.seed + i,
                early_stopping=True,
                validation_fraction=0.15,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlp.fit(X, y)
            self._members.append(mlp)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([m.predict(X) for m in self._members])
        return preds.mean(axis=0)

    def predict_with_std(
        self, X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        preds = np.array([m.predict(X) for m in self._members])
        return preds.mean(axis=0), np.maximum(preds.std(axis=0), 1e-8)


def build_regressor_candidates(
    d: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return list of regression candidate configs.

    Each candidate has:
      - 'name': str
      - 'uncertainty_type': str flag
      - 'build_fn': callable → model
      - 'predict_std_method': 'return_std' | 'predict_with_std'

    Args:
        d: Input dimensionality.
        seed: Random seed.

    Returns:
        List of candidate dicts.
    """
    candidates = []

    # GPR kernels with WhiteKernel
    kernel_configs = [
        ("GPR_Matern25_White", Matern(nu=2.5) + WhiteKernel(1e-3)),
        ("GPR_Matern15_White", Matern(nu=1.5) + WhiteKernel(1e-3)),
        ("GPR_RBF_White", RBF() + WhiteKernel(1e-3)),
        ("GPR_RatQuad_White", RationalQuadratic() + WhiteKernel(1e-3)),
        (
            "GPR_Matern25_Scaled",
            ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-3),
        ),
        (
            "GPR_Matern15_Scaled",
            ConstantKernel(1.0) * Matern(nu=1.5) + WhiteKernel(1e-3),
        ),
        (
            "GPR_Composite_M25_RBF",
            Matern(nu=2.5) + RBF() + WhiteKernel(1e-3),
        ),
    ]

    for name, kernel in kernel_configs:
        k = kernel  # capture for closure
        candidates.append({
            "name": name,
            "uncertainty_type": "gp_posterior",
            "predict_std_method": "return_std",
            "build_fn": lambda k=k: GaussianProcessRegressor(
                kernel=k,
                normalize_y=True,
                random_state=seed,
                n_restarts_optimizer=3,
            ),
        })

    # Quantile RF
    candidates.append({
        "name": "QuantileRF",
        "uncertainty_type": "interval_approx",
        "predict_std_method": "predict_with_std",
        "build_fn": lambda: QuantileRFRegressor(
            n_estimators=300, seed=seed,
        ),
    })

    # MLP Ensemble
    candidates.append({
        "name": "MLPEnsemble",
        "uncertainty_type": "ensemble_disagreement",
        "predict_std_method": "predict_with_std",
        "build_fn": lambda: MLPEnsembleRegressor(
            n_members=10, seed=seed,
        ),
    })

    return candidates


def build_tuned_regressor(
    kernel_name: str,
    lengthscales: np.ndarray,
    alpha: float,
    nu: float | None,
    seed: int,
) -> GaussianProcessRegressor:
    """Build a GPR with pre-tuned fixed hyperparameters.

    Args:
        kernel_name: One of 'Matern25', 'Matern15', 'RBF', 'RatQuad'.
        lengthscales: Per-dimension lengthscale array.
        alpha: Noise parameter.
        nu: Matern smoothness (only used if kernel_name is Matern).
        seed: Random seed.

    Returns:
        Configured GaussianProcessRegressor with fixed kernel.
    """
    if kernel_name.startswith("Matern"):
        _nu = nu if nu is not None else (
            2.5 if "25" in kernel_name else 1.5
        )
        kernel = Matern(
            nu=_nu,
            length_scale=lengthscales,
            length_scale_bounds="fixed",
        )
    elif kernel_name == "RBF":
        kernel = RBF(
            length_scale=lengthscales,
            length_scale_bounds="fixed",
        )
    elif kernel_name == "RatQuad":
        kernel = RationalQuadratic(
            length_scale=np.mean(lengthscales),
            length_scale_bounds="fixed",
        )
    else:
        kernel = Matern(
            nu=2.5,
            length_scale=lengthscales,
            length_scale_bounds="fixed",
        )

    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        random_state=seed,
        optimizer=None,
    )
