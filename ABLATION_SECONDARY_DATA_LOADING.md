# How the Ablation Code Reads and Uses Secondary Data

This document explains exactly how `feature_builder.py` loads and uses each
descriptor file. It covers what the file contains, what the code does with it,
how missing values are handled, and which columns end up in the feature matrix.

---

## Where This Happens in the Code

Secondary data is loaded inside `feature_builder.py` by three internal functions:

| Function | File(s) it reads |
|---|---|
| `_build_hlb_block()` | `hlb_values.csv` |
| `_build_compat_block()` | `oil_surfactant_compatibility.csv`, `oil_cosurfactant_compatibility.csv`, `surfactant_cosurfactant_compatibility.csv` |
| `_build_solubility_block()` | `solubility_values.csv` |

These are called by `build_features()` depending on which schema was requested.
The base schema never calls any of them. Each function returns a numpy array
block and a list of column names, which are horizontally stacked into the final
feature matrix between the categorical block and the continuous block.

---

## 1. `hlb_values.csv`

### What the file contains

17 rows, one per component. Three columns:

```
Component,Type,HLB_Value
Capmul_MCM,Oil,5.5
Capryol_90,Oil,5.0
Maisine_Oil,Oil,1.0
Soybean_Oil,Oil,7.8
Oleic_Acid,Oil,1.0
Labrasol,Surfactant,12.0
Tween_80,Surfactant,15.0
Tween_20,Surfactant,16.7
PEG_400,Cosurfactant,11.3
Transcutol_HP,Surfactant,4.0
... (17 total)
```

HLB values range from 1.0 to 29.0.

### How the code loads it

```python
hlb_df  = pd.read_csv(data_dir / "hlb_values.csv")
hlb_map = dict(zip(hlb_df["Component"], hlb_df["HLB_Value"]))
```

It builds a flat dictionary keyed by component name. The `Type` column is
**ignored entirely** at this step.

### Why Type is ignored

Two components appear in dual roles across formulation rows:

- `PEG_400` is labelled `Cosurfactant` in this file, but is used as a
  `Surfactant` in approximately 12 rows of `MicroemulsionFormulation.csv`
- `Tween_80` is labelled `Surfactant` in this file, but is used as a
  `Cosurfactant` in some formulation rows

Ignoring `Type` means the lookup always finds the correct HLB value regardless
of which role the component plays in a given formulation row.

### How the values are scaled

All 17 HLB values are min-max scaled using the global min and max across the
whole file:

```
min = 1.0    (Maisine_Oil, Oleic_Acid)
max = 29.0   (Pluronic_F-68)
range = 28.0

scaled = (raw_value - 1.0) / 28.0
```

So `Oleic_Acid` → 0.0, `Pluronic_F-68` → 1.0, `Tween_80` → 0.5.

### What goes into the feature matrix

Three columns, one per component role in a formulation row:

| Column | Lookup key per row |
|---|---|
| `HLB_Oil` | `df["Oil"]` |
| `HLB_Surfactant` | `df["Surfactant"]` |
| `HLB_Cosurfactant` | `df["Cosurfactant"]` |

Each of the 147 formulation rows gets its three HLB values looked up
individually and placed in its corresponding row of the feature matrix.

### Missing value handling

If a component name is not found in the lookup dict, the code logs a warning
and imputes the scaled global mean. In the current dataset this never triggers
— all 5 oils, 4 surfactants, and 5 cosurfactants present in the formulation
data have entries in `hlb_values.csv`.

---

## 2. `oil_surfactant_compatibility.csv`

### What the file contains

A 6×7 binary matrix. Rows are the 6 oils. Columns are 7 surfactants. All
entries are 0 or 1 — there are no NaN values.

```
               Labrasol  Tween_80  Tween_20  Transcutol_HP  ...  PEG_400
Capmul_MCM           1         1         0              1           1
Capryol_90           1         1         1              1           1
Maisine_Oil          1         1         1              1           1
Soybean_Oil          1         1         1              1           1
Safflower_Oil        1         1         1              1           0
Oleic_Acid           1         1         0              1           1
```

Note: the matrix contains 7 surfactant columns including some (Kolliphor_RH_40,
Pluronic_F-68) that do not appear in the current formulation dataset.

### How the code loads it

```python
os_df = pd.read_csv(
    data_dir / "oil_surfactant_compatibility.csv", index_col=0,
)
```

Loaded with `index_col=0` so the oil names become the row index and the
surfactant names become column headers. Lookup is then:

```python
float(os_df.loc[row.Oil, row.Surfactant])
```

### What goes into the feature matrix

One column: `compat_oil_surf`. A 0 or 1 for each formulation row. No scaling
is applied because the values are already binary.

### Missing value handling

If the (Oil, Surfactant) key pair is not found in the matrix, the value falls
back to `-1` and a debug message is logged. This does not occur for any row in
the current dataset.

---

## 3. `oil_cosurfactant_compatibility.csv`

### What the file contains

A 6×7 binary matrix structured identically to the surfactant version above.
Rows are 6 oils. Columns are 7 cosurfactants. All entries are 0 or 1, no NaN.

```
               PEG_400  Propylene_Glycol  Ethanol  Cremophor_EL  ...  Tween_80
Capmul_MCM           1                 1        1             1           1
Capryol_90           1                 1        1             1           1
Maisine_Oil          1                 0        1             1           1
Soybean_Oil          1                 0        1             0           1
Safflower_Oil        0                 0        0             1           1
Oleic_Acid           1                 0        1             1           1
```

### How the code loads and uses it

Identical pattern to the surfactant version:

```python
oc_df = pd.read_csv(
    data_dir / "oil_cosurfactant_compatibility.csv", index_col=0,
)
float(oc_df.loc[row.Oil, row.Cosurfactant])
```

### What goes into the feature matrix

One column: `compat_oil_cosurf`. Binary 0/1, no scaling. Fallback to `-1` if
key missing (does not occur in the current dataset).

---

## 4. `surfactant_cosurfactant_compatibility.csv`

### What the file contains

A 9×9 matrix where both rows and columns are the same 9 components:
Cremophor_EL, Ethanol, Glycerin, Labrasol, PEG_400, Propylene_Glycol,
Transcutol_HP, Tween_20, Tween_80.

**This matrix has genuine NaN entries.** Not every surfactant–cosurfactant
pair has been measured.

```
                  Cremophor_EL  Ethanol  Glycerin  Labrasol  PEG_400  ...  Tween_80
Cremophor_EL               1.0      NaN       NaN       1.0      NaN           1.0
Ethanol                    NaN      1.0       NaN       1.0      NaN           1.0
Glycerin                   NaN      NaN       1.0       0.0      NaN           0.0
Labrasol                   1.0      1.0       0.0       1.0      1.0           NaN
PEG_400                    NaN      NaN       NaN       1.0      1.0           0.0
Propylene_Glycol           NaN      NaN       NaN       1.0      NaN           1.0
Transcutol_HP              1.0      1.0       1.0       1.0      1.0           1.0
Tween_20                   1.0      1.0       0.0       NaN      1.0           0.0
Tween_80                   1.0      1.0       0.0       NaN      0.0           1.0
```

Example NaN pairs: Labrasol↔Tween_20, Labrasol↔Tween_80, PEG_400↔Cremophor_EL.

### Why NaN is different from 0

A `0` in this matrix means the pairing is **known to be incompatible**.
A `NaN` means **no measurement exists** — the pairing may be compatible,
incompatible, or untested. These are meaningfully different states and must not
be conflated.

### How the code handles it

This is the most nuanced loading step. The code checks for NaN explicitly and
produces two columns instead of one:

```python
val = sc_df.loc[surf, cosurf]

if pd.isna(val):
    surf_cosurf[i] = -1.0      # unknown — not incompatible
    sc_missing[i]  = 1.0       # flag: no data for this pair
else:
    surf_cosurf[i] = float(val) # 0 = incompatible, 1 = compatible
    sc_missing[i]  = 0.0        # flag: data exists
```

If the (Surfactant, Cosurfactant) key pair is completely absent from the matrix
(not found as a row or column at all), it is also treated as missing:
`surf_cosurf = -1`, `sc_missing = 1`.

### What goes into the feature matrix

Two columns:

| Column | Values | Meaning |
|---|---|---|
| `compat_surf_cosurf` | `1` | Known compatible |
| | `0` | Known incompatible |
| | `-1` | No measurement — unknown |
| `compat_surf_cosurf_missing` | `0` | Data exists for this pair |
| | `1` | No data exists for this pair |

The missing indicator column exists so the model can learn separately that the
pairing is uncertain. Without it, `-1` is just another numeric value and the
model has no way to distinguish "measured as unknown" from a deliberate encoding
choice.

### Which formulation rows get missing=1

Any row where the `(Surfactant, Cosurfactant)` combination falls on a NaN cell.
For example: rows with `Surfactant=Labrasol` and `Cosurfactant=Tween_80` get
`compat_surf_cosurf = -1`, `compat_surf_cosurf_missing = 1`.

---

## 5. `solubility_values.csv`

### What the file contains

15 rows, one per component. Three columns: `Component`, `Type`,
`Solubility_Value`.

```
Component,Type,Solubility_Value
Capmul_MCM,Oils,3.106
Capryol_90,Oils,12.755
Maisine_Oil,Oils,5.463
Soybean_Oil,Oils,0.622
Safflower_Oil,Oils,0.663
Oleic_Acid,Oils,4.298
Labrasol,Surfactant,6.76
Tween_80,Surfactant,7.609
Tween_20,Surfactant,2.316
PEG_400,Cosurfactant,11.405
Propylene_Glycol,Cosurfactant,5.300
Ethanol,Cosurfactant,5.560
Cremophor_EL,Cosurfactant,6.219
Glycerin,Cosurfactant,0.033
Transcutol_HP,Cosurfactant,10.095
```

Values range from 0.033 (Glycerin) to 12.755 (Capryol_90).

**Notable gap:** `PEG_400` appears here only as a `Cosurfactant` entry with
value 11.405. It does **not** appear as a `Surfactant` entry. In the formulation
data, `PEG_400` is used as a `Surfactant` in approximately 19 rows (~13% of the
dataset). Those rows have no solubility entry for their surfactant role.

### How the code loads it

```python
sol_df  = pd.read_csv(data_dir / "solubility_values.csv")
sol_map = dict(zip(sol_df["Component"], sol_df["Solubility_Value"]))
```

Same name-only lookup pattern as HLB. The `Type` column is not used.

### How the values are scaled

Min-max scaled across all 15 entries in the file:

```
min = 0.033    (Glycerin)
max = 12.755   (Capryol_90)
range = 12.722

scaled = (raw_value - 0.033) / 12.722
```

The lookup function returns `(scaled_value, missing_flag)`:

```python
def _lookup(name):
    v = sol_map.get(name)
    if v is None:
        return 0.0, 1.0       # missing: raw=0, flag=1
    return (v - sol_min) / sol_range, 0.0   # found: scaled, flag=0
```

### What goes into the feature matrix

Five columns:

| Column | What it is | Missing case |
|---|---|---|
| `sol_oil` | Scaled solubility of `df["Oil"]` | Never missing in current dataset |
| `sol_surf` | Scaled solubility of `df["Surfactant"]` | `0.0` when not in file |
| `sol_surf_missing` | `1` if surfactant solubility absent, else `0` | `1` for ~19 PEG_400-as-surfactant rows |
| `sol_cosurf` | Scaled solubility of `df["Cosurfactant"]` | Never missing in current dataset |
| `sol_diff_oil_surf` | `abs(sol_oil - sol_surf)` — miscibility proxy | `0.0` when surf is missing |

### Why `sol_diff_oil_surf` is zeroed when surf is missing

Without this guard, rows where `PEG_400` is the surfactant would produce
`abs(sol_oil - 0.0)`, which would equal the oil's scaled solubility value and
be treated by the model as a real miscibility measurement. That would be
spurious signal. Zeroing the diff when the surfactant data is absent prevents
this. The `sol_surf_missing` column separately informs the model that the diff
is not meaningful for those rows.

---

## Summary: All Secondary Columns Produced

| Schema | File(s) used | Columns added | Total desc cols |
|---|---|---|---|
| `base` | None | None | 0 |
| `base_hlb` | `hlb_values.csv` | `HLB_Oil`, `HLB_Surfactant`, `HLB_Cosurfactant` | 3 |
| `base_compat` | Oil-surf, oil-cosurf, surf-cosurf CSVs | `compat_oil_surf`, `compat_oil_cosurf`, `compat_surf_cosurf`, `compat_surf_cosurf_missing` | 4 |
| `base_hlb_compat` | All four above | HLB (3) + compat (4) | 7 |
| `base_hlb_compat_solubility` | All five files | HLB (3) + compat (4) + sol (5) | 12 |

### Full descriptor column list for the largest schema

```
[17] HLB_Oil
[18] HLB_Surfactant
[19] HLB_Cosurfactant
[20] compat_oil_surf
[21] compat_oil_cosurf
[22] compat_surf_cosurf            ← 0/1/-1
[23] compat_surf_cosurf_missing    ← 0/1 indicator
[24] sol_oil
[25] sol_surf                      ← 0 when PEG_400 is surfactant
[26] sol_surf_missing              ← 1 when PEG_400 is surfactant
[27] sol_cosurf
[28] sol_diff_oil_surf             ← 0 when surf missing
```

Indices 0–16 are one-hot categorical columns. Indices 29–31 are the scaled
continuous process variables (Oil_V, S_Ratio, Sonication).

---

## Data Gaps That Produce Non-Trivial Encodings

Two situations produce something other than a clean measured value in the
feature matrix. Both are handled with explicit missing indicators rather than
silent imputation:

| Gap | Affected rows | Columns affected |
|---|---|---|
| NaN in surf–cosurf compatibility matrix | Any row where `(Surfactant, Cosurfactant)` falls on a NaN cell | `compat_surf_cosurf = -1`, `compat_surf_cosurf_missing = 1` |
| PEG_400 solubility absent as surfactant | ~19 rows where `Surfactant = PEG_400` | `sol_surf = 0`, `sol_surf_missing = 1`, `sol_diff_oil_surf = 0` |

If experimental data for either gap becomes available, update the relevant CSV
and re-run `build_features()`. No code changes are needed.
