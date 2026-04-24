"""Modular feature construction for microemulsion surrogate ablation study.

Five schemas supported:
  base                       — one-hot categoricals + scaled continuous
  base_hlb                   — base + HLB descriptors (oil, surf, cosurf)
  base_compat                — base + pairwise compatibility descriptors
  base_hlb_compat            — base + HLB + compatibility
  base_hlb_compat_solubility — base + HLB + compatibility + solubility

Continuous inputs (updated dataset format)
------------------------------------------
  Oil_V, Surfactant_V, Cosurfactant_V, Sonication

  Note: the legacy S_Ratio column is no longer present. Surfactant_V and
  Cosurfactant_V are the independent volume representations that replaced it.

Scaling ranges for continuous inputs
-------------------------------------
  Oil_V          : (6.6666667, 20.625)   — observed data range; verify against
                                            the intended BO search-space bounds
                                            if different from training data
  Surfactant_V   : (10.0, 30.0)
  Cosurfactant_V : (10.0, 30.0)
  Sonication     : (0.0, 3.0)

Missing-value strategy
----------------------
  surfactant–cosurfactant NaN entries:
    → raw value set to -1  (unknown, distinguishable from 0=incompatible)
    → plus a binary missing-indicator column  compat_surf_cosurf_missing

  surfactant solubility missing (PEG_400 as surfactant not in file):
    → raw value set to 0
    → plus a binary missing-indicator column  sol_surf_missing

Schema feature counts (with 4 continuous inputs)
-------------------------------------------------
  base                       : 17 cat + 4 cont           = 21
  base_hlb                   : 17 cat + 3 hlb + 4 cont   = 24
  base_compat                : 17 cat + 4 compat + 4 cont = 25
  base_hlb_compat            : 17 cat + 3 hlb + 4 compat + 4 cont = 28
  base_hlb_compat_solubility : 17 cat + 3 hlb + 4 compat + 5 sol + 4 cont = 33
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

logger = logging.getLogger(__name__)

SCHEMAS: list[str] = [
    "base",
    "base_hlb",
    "base_compat",
    "base_hlb_compat",
    "base_hlb_compat_solubility",
]

OUTPUT_HEADERS: list[str] = [
    "Droplet_Size", "PDI", "Zeta_P",
    "Phase_Sep", "Drug_Loading", "Permeability",
]

_CAT_HEADERS: list[str] = ["Oil", "Surfactant", "Cosurfactant", "API_Name"]

# Updated: S_Ratio replaced by Surfactant_V and Cosurfactant_V
_CONT_HEADERS: list[str] = ["Oil_V", "Surfactant_V", "Cosurfactant_V", "Sonication"]

# Scaling ranges for continuous inputs.
# Oil_V range reflects the observed data extent in the updated dataset.
# Surfactant_V and Cosurfactant_V use their clean design-space bounds (10–30).
# Override these constants if your BO search space differs from the training data.
_CONT_RANGES: dict[str, tuple[float, float]] = {
    "Oil_V":          (6.6666667, 20.625),
    "Surfactant_V":   (10.0, 30.0),
    "Cosurfactant_V": (10.0, 30.0),
    "Sonication":     (0.0, 3.0),
}


# ------------------------------------------------------------------ #
#  Internal block builders                                            #
# ------------------------------------------------------------------ #

def _scale_cont(arr: np.ndarray) -> np.ndarray:
    """Domain min-max scale for continuous columns."""
    result = np.array(arr, dtype=float)
    for j, col in enumerate(_CONT_HEADERS):
        lo, hi = _CONT_RANGES[col]
        result[:, j] = (result[:, j] - lo) / max(hi - lo, 1e-8)
    return result


def _build_base_blocks(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[str]]:
    """One-hot categoricals + scaled continuous.

    Returns:
        cat_block   shape (n, n_cat)
        cont_block  shape (n, 4)
        cat_col_indices   in the final X layout
        cont_col_indices  in the final X layout  (after cat block)
        feature_names
    """
    input_headers = _CAT_HEADERS + _CONT_HEADERS
    ct = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                _CAT_HEADERS,
            ),
            (
                "cont",
                FunctionTransformer(_scale_cont, validate=False),
                _CONT_HEADERS,
            ),
        ],
        remainder="drop",
    )
    processed = ct.fit_transform(df[input_headers])
    if issparse(processed):
        processed = processed.toarray()

    n_cat = processed.shape[1] - len(_CONT_HEADERS)
    cat_block  = processed[:, :n_cat]
    cont_block = processed[:, n_cat:]

    enc = ct.named_transformers_["cat"]
    cat_names  = enc.get_feature_names_out(_CAT_HEADERS).tolist()
    cont_names = list(_CONT_HEADERS)

    cat_cols  = list(range(n_cat))
    cont_cols = list(range(n_cat, n_cat + len(_CONT_HEADERS)))

    return cat_block, cont_block, cat_cols, cont_cols, cat_names + cont_names


def _build_hlb_block(
    df: pd.DataFrame,
    data_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Three HLB columns: oil, surfactant, cosurfactant.

    ASSUMPTION: lookup is by Component name only, ignoring the Type column.
    Rationale: PEG_400 is listed as Cosurfactant in hlb_values.csv but appears
    as Surfactant in some formulation rows; Tween_80 is listed as Surfactant
    but also used as Cosurfactant.  Using name-only lookup returns the correct
    HLB value regardless of which role the component plays in a given row.

    If a component name is not found, its HLB is imputed as the global mean
    and a WARNING is emitted.
    """
    hlb_df  = pd.read_csv(data_dir / "hlb_values.csv")
    hlb_map = dict(zip(hlb_df["Component"], hlb_df["HLB_Value"]))

    hlb_min   = float(min(hlb_map.values()))
    hlb_max   = float(max(hlb_map.values()))
    hlb_range = max(hlb_max - hlb_min, 1e-8)
    hlb_mean  = (hlb_min + hlb_max) / 2.0

    def _lookup(name: str) -> float:
        v = hlb_map.get(name)
        if v is None:
            logger.warning("HLB missing for '%s'; imputing mean HLB.", name)
            return (hlb_mean - hlb_min) / hlb_range
        return (float(v) - hlb_min) / hlb_range

    oil_hlb    = np.array([_lookup(o) for o in df["Oil"]],          dtype=float)
    surf_hlb   = np.array([_lookup(s) for s in df["Surfactant"]],   dtype=float)
    cosurf_hlb = np.array([_lookup(c) for c in df["Cosurfactant"]], dtype=float)

    block = np.column_stack([oil_hlb, surf_hlb, cosurf_hlb])
    names = ["HLB_Oil", "HLB_Surfactant", "HLB_Cosurfactant"]
    return block, names


def _build_compat_block(
    df: pd.DataFrame,
    data_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Four compatibility columns (3 scores + 1 missing indicator).

    Columns:
      compat_oil_surf           0/1   from oil_surfactant_compatibility.csv
      compat_oil_cosurf         0/1   from oil_cosurfactant_compatibility.csv
      compat_surf_cosurf        0/1/-1   NaN entries encoded as -1
      compat_surf_cosurf_missing  1 if NaN in source matrix, else 0

    ASSUMPTION: The surfactant–cosurfactant matrix has genuine missing entries
    (NaN) where no compatibility experiment was reported.  These are encoded as
    -1 (distinct from 0=incompatible) plus a separate missing-indicator column
    so the model can learn both that the pairing is uncertain AND condition on
    whether data exists.
    """
    os_df = pd.read_csv(
        data_dir / "oil_surfactant_compatibility.csv", index_col=0,
    )
    oc_df = pd.read_csv(
        data_dir / "oil_cosurfactant_compatibility.csv", index_col=0,
    )
    sc_df = pd.read_csv(
        data_dir / "surfactant_cosurfactant_compatibility.csv", index_col=0,
    )

    n = len(df)
    oil_surf    = np.zeros(n, dtype=float)
    oil_cosurf  = np.zeros(n, dtype=float)
    surf_cosurf = np.zeros(n, dtype=float)
    sc_missing  = np.zeros(n, dtype=float)

    for i, (oil, surf, cosurf) in enumerate(
        zip(df["Oil"], df["Surfactant"], df["Cosurfactant"])
    ):
        # oil–surfactant (matrix is fully populated, no NaN)
        try:
            oil_surf[i] = float(os_df.loc[oil, surf])
        except KeyError:
            logger.debug("oil_surf key missing (%s, %s); using -1", oil, surf)
            oil_surf[i] = -1.0

        # oil–cosurfactant (matrix is fully populated, no NaN)
        try:
            oil_cosurf[i] = float(oc_df.loc[oil, cosurf])
        except KeyError:
            logger.debug("oil_cosurf key missing (%s, %s); using -1", oil, cosurf)
            oil_cosurf[i] = -1.0

        # surfactant–cosurfactant (NaN entries present)
        try:
            val = sc_df.loc[surf, cosurf]
            if pd.isna(val):
                surf_cosurf[i] = -1.0
                sc_missing[i]  = 1.0
            else:
                surf_cosurf[i] = float(val)
                sc_missing[i]  = 0.0
        except KeyError:
            surf_cosurf[i] = -1.0
            sc_missing[i]  = 1.0

    block = np.column_stack([oil_surf, oil_cosurf, surf_cosurf, sc_missing])
    names = [
        "compat_oil_surf",
        "compat_oil_cosurf",
        "compat_surf_cosurf",
        "compat_surf_cosurf_missing",
    ]
    return block, names


def _build_solubility_block(
    df: pd.DataFrame,
    data_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Five solubility columns (3 raw + 1 missing indicator + 1 interaction).

    Columns:
      sol_oil             scaled solubility for Oil
      sol_surf            scaled solubility for Surfactant (0 if missing)
      sol_surf_missing    1 if surfactant solubility not in file, else 0
      sol_cosurf          scaled solubility for Cosurfactant
      sol_diff_oil_surf   |sol_oil - sol_surf|, proxy for miscibility
                          (set to 0 when surf solubility is missing)

    ASSUMPTION: PEG_400 appears in solubility_values.csv only as Cosurfactant.
    When used as Surfactant in a row it has no solubility entry → encoded as
    (0, missing=1).  The sol_diff feature is zeroed for those rows to avoid
    spurious signal from the imputed 0.

    All values are min-max scaled across all entries in solubility_values.csv.
    """
    sol_df  = pd.read_csv(data_dir / "solubility_values.csv")
    sol_map = dict(zip(sol_df["Component"], sol_df["Solubility_Value"]))

    sol_min   = float(min(sol_map.values()))
    sol_max   = float(max(sol_map.values()))
    sol_range = max(sol_max - sol_min, 1e-8)

    def _lookup(name: str) -> tuple[float, float]:
        """Return (scaled_value, missing_flag)."""
        v = sol_map.get(name)
        if v is None:
            return 0.0, 1.0
        return (float(v) - sol_min) / sol_range, 0.0

    n = len(df)
    sol_oil      = np.zeros(n, dtype=float)
    sol_surf     = np.zeros(n, dtype=float)
    sol_surf_mis = np.zeros(n, dtype=float)
    sol_cosurf   = np.zeros(n, dtype=float)
    sol_diff     = np.zeros(n, dtype=float)

    for i, (oil, surf, cosurf) in enumerate(
        zip(df["Oil"], df["Surfactant"], df["Cosurfactant"])
    ):
        so, _   = _lookup(oil)
        ss, sm  = _lookup(surf)
        sc, _   = _lookup(cosurf)

        sol_oil[i]      = so
        sol_surf[i]     = ss
        sol_surf_mis[i] = sm
        sol_cosurf[i]   = sc
        sol_diff[i]     = abs(so - ss) if sm == 0.0 else 0.0

    block = np.column_stack([sol_oil, sol_surf, sol_surf_mis, sol_cosurf, sol_diff])
    names = [
        "sol_oil",
        "sol_surf",
        "sol_surf_missing",
        "sol_cosurf",
        "sol_diff_oil_surf",
    ]
    return block, names


# ------------------------------------------------------------------ #
#  Public interface                                                   #
# ------------------------------------------------------------------ #

def build_features(
    data_path: str | Path,
    schema: str = "base",
    data_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build feature matrix X and output matrix y for the given schema.

    Args:
        data_path: Path to MicroemulsionFormulation.csv.
        schema:    One of SCHEMAS.
        data_dir:  Directory containing descriptor CSV files.
                   Defaults to the same directory as data_path.

    Returns:
        X     — shape (n, n_features), fully numeric, no NaN
        y     — shape (n, 6), raw output values (NaN preserved for targets
                that are not always observed)
        meta  — dict with keys:
                  schema, n_features, feature_names,
                  cat_cols, cont_cols, desc_cols

    Feature layout (by schema):
      base:                         [cat (17) | cont (4)]
      base_hlb:                     [cat (17) | HLB (3) | cont (4)]
      base_compat:                  [cat (17) | compat (4) | cont (4)]
      base_hlb_compat:              [cat (17) | HLB (3) | compat (4) | cont (4)]
      base_hlb_compat_solubility:   [cat (17) | HLB (3) | compat (4) | sol (5) | cont (4)]

    Expected feature counts (17 cat + 4 cont = 21 base):
      base                       21
      base_hlb                   24
      base_compat                25
      base_hlb_compat            28
      base_hlb_compat_solubility 33
    """
    if schema not in SCHEMAS:
        raise ValueError(
            f"Unknown schema '{schema}'. Choose from: {SCHEMAS}"
        )

    data_path = Path(data_path)
    data_dir  = Path(data_dir) if data_dir else data_path.parent

    df = pd.read_csv(data_path)

    # Verify the updated column format is present
    missing_cols = [c for c in _CONT_HEADERS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Expected continuous columns not found in CSV: {missing_cols}. "
            f"CSV has columns: {df.columns.tolist()}"
        )

    logger.info("build_features: schema=%s, n_rows=%d", schema, len(df))

    cat_block, cont_block, _, _, base_names = _build_base_blocks(df)
    n_cat = cat_block.shape[1]
    cat_names  = base_names[:n_cat]
    cont_names = base_names[n_cat:]

    # Accumulate descriptor blocks in insertion order
    desc_blocks: list[np.ndarray] = []
    desc_names:  list[str]        = []

    if schema in ("base_hlb", "base_hlb_compat", "base_hlb_compat_solubility"):
        blk, nms = _build_hlb_block(df, data_dir)
        desc_blocks.append(blk)
        desc_names.extend(nms)

    if schema in ("base_compat", "base_hlb_compat", "base_hlb_compat_solubility"):
        blk, nms = _build_compat_block(df, data_dir)
        desc_blocks.append(blk)
        desc_names.extend(nms)

    if schema == "base_hlb_compat_solubility":
        blk, nms = _build_solubility_block(df, data_dir)
        desc_blocks.append(blk)
        desc_names.extend(nms)

    n_desc = sum(b.shape[1] for b in desc_blocks)

    all_blocks = [cat_block] + desc_blocks + [cont_block]
    X = np.hstack(all_blocks)
    y = df[OUTPUT_HEADERS].values.astype(float)

    # Index sets in the final layout
    cat_cols  = list(range(n_cat))
    desc_cols = list(range(n_cat, n_cat + n_desc))
    cont_cols = list(range(n_cat + n_desc, n_cat + n_desc + len(_CONT_HEADERS)))

    feature_names = cat_names + desc_names + cont_names

    assert X.shape[1] == len(feature_names), (
        f"Shape mismatch: X has {X.shape[1]} cols but {len(feature_names)} names"
    )
    assert not np.any(np.isnan(X)), "NaN found in feature matrix X"

    meta: dict[str, Any] = {
        "schema":        schema,
        "n_features":    X.shape[1],
        "feature_names": feature_names,
        "cat_cols":      cat_cols,
        "cont_cols":     cont_cols,
        "desc_cols":     desc_cols,
    }

    logger.info(
        "  X=%s  cat=%d  desc=%d  cont=%d",
        X.shape, len(cat_cols), len(desc_cols), len(cont_cols),
    )
    return X, y, meta
