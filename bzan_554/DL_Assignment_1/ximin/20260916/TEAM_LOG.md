# Team Log — Shared Reference

**Project:** Pricing quantity regression (deep learning)
**Owner of this doc:** xzeng3@utk.edu
**Last updated:** 2026-09-17

Single source of truth for *where things live*, *what to install*, and *what happened on each run*.

---

## 1. Repository map

Current state: all `.py` files are flat in one directory, but `config.py` assumes a
`code/` + `data/` + `results/` layout (issue I-09).

### Expected layout (what `config.py` implies)

```
project_root/
├── code/
│   ├── config.py
│   ├── data_handler.py
│   ├── model.py
│   ├── evaluate.py
│   ├── main.py
│   ├── test_integration.py
│   ├── utils.py
│   └── run_sanity_checks.py          <- new
├── data/
│   ├── pricing.csv                                  (19 MB, 500,000 rows)
│   ├── pricing_TEST_SAMPLE_first1000.csv            <- new
│   └── pricing_TEST_SAMPLE_random1000.csv           <- new
├── results/                          (auto-created by config.py)
│   ├── trained_model.h5
│   └── metrics.csv
├── TESTING_CHECKLIST.md              <- new
├── PROJECT_NOTES.md                  <- new
├── TEAM_LOG.md                       <- new (this file)
├── requirements.txt                  <- new
└── README.md
```

`config.py` resolves `DATA_FILE` as `<parent of config.py>/data/pricing.csv`. **If the files stay
flat, `main.py` will raise `FileNotFoundError` on line 17 of `data_handler.py`.** Either create the
folders above or change `config.DATA_FILE`.

---

## 2. File-by-file reference

| File | Lines | Purpose | Key contents | Notes |
|---|---|---|---|---|
| `config.py` | 33 | All settings | `DATA_FILE`, `TEST_SPLIT=0.2`, `RANDOM_SEED=42`, layer sizes, `BATCH_SIZE=32`, `LEARNING_RATE=0.001`, paths | Creates `results/` on import as a side effect. `EPOCHS` and `VERBOSE` are **never read** (I-06) |
| `data_handler.py` | 63 | Load, split, scale, batch | `load_and_split_data()`, `create_mini_batches()` | Scaler is **not returned or saved** (I-03) |
| `model.py` | 37 | Network definition | `build_model()` → 3,041 params | 3× sigmoid hidden + linear output (D-01) |
| `evaluate.py` | 52 | Metrics | `calculate_r2()`, `calculate_mse()`, `calculate_moving_average()` | Thin sklearn wrappers. Self-test: R² 0.9880, MSE 0.0220 |
| `main.py` | 96 | Training entry point | `train_incremental()` | Overwrites `metrics.csv` each run (I-01) |
| `test_integration.py` | 95 | Smoke test | `test_incremental_learning_small()` | No assertions (I-10); loads full CSV (I-08) |
| `utils.py` | 1 | — | *empty* | One comment, zero code. Imported by nothing (I-07) |
| `run_sanity_checks.py` | ~320 | **New.** Automated D/F/P/B checks | 24 checks, no TensorFlow needed | Exit 0 = pass |

### Import graph

```
config.py  ──> imported by all 6 other modules (via `from config import *`)
                    │
data_handler.py ────┤
model.py ───────────┤
evaluate.py ────────┘ (no config dependency)
                    │
main.py ────────────> model, data_handler, evaluate, config
test_integration.py ─> model, data_handler, evaluate, config
run_sanity_checks.py > standalone (pandas/numpy/sklearn only)
```

All modules use `from config import *`. This works but pollutes the namespace and makes it
impossible to tell where a name came from — `BATCH_SIZE` in `data_handler.py` has no visible
origin. Prefer `import config` and `config.BATCH_SIZE` (I-13).

---

## 3. Dependencies

### `requirements.txt`

```
tensorflow>=2.16.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
```

### Install

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Where each package is used

| Package | Used by | For |
|---|---|---|
| `tensorflow` / `keras` | `model.py`, `main.py`, `test_integration.py` | Network, Adam, `train_on_batch` |
| `pandas` | `data_handler.py`, `main.py`, `test_integration.py` | CSV I/O, DataFrames, `.iloc` batching |
| `numpy` | `evaluate.py`, `main.py` | `np.convolve` for the moving average |
| `scikit-learn` | `data_handler.py`, `evaluate.py` | `train_test_split`, `StandardScaler`, `r2_score`, `mean_squared_error` |
| `joblib` | *not yet used* | Needed to persist the scaler (fix I-03) |
| `matplotlib` | *not yet used* | Needed to plot the learning curve (fix I-05) |

### Version warnings

- **Keras 3** (bundled with TF ≥ 2.16) changed the saving API. `config.MODEL_SAVE_PATH` ends in
  `.h5`, which is legacy; Keras 3 prefers `.keras`. It still works but warns. (I-11)
- **Keras 3** also changed `train_on_batch` return types. `main.py:44` already handles both with
  `metrics[0] if isinstance(metrics, list) else metrics` — leave that in.
- `tf.keras.utils.set_random_seed()` requires TF ≥ 2.9 (used in the I-02 fix).

### Verify the environment

```bash
python -c "import tensorflow as tf, sklearn, pandas, numpy; \
print('tf', tf.__version__, '| sk', sklearn.__version__, \
'| pd', pandas.__version__, '| np', numpy.__version__)"
```

**Record the output in the run log below for every run.** Version drift is a common cause of
"the same code gives a different number."

> **Environment note (2026-09-17):** TensorFlow is **not installed** in the sandbox used to
> produce these documents, so tiers S, M and R have **not** been executed here. Tiers D, F, P
> and B were fully executed against the real 500k dataset and the numbers in the checklist are
> measured, not estimated. The sigmoid-vs-relu comparison (D-01) was measured with
> `sklearn.neural_network.MLPRegressor` configured to match `model.py` (layers 64/32/16, Adam,
> lr 0.001, batch 32, single pass) — it is a faithful proxy, but re-confirm it in Keras.

---

## 4. Test data subsets

Both files have the **same 6-column schema** as `pricing.csv` and can be dropped into
`config.DATA_FILE` with no code changes.

| File | Rows | Purpose | Coverage |
|---|---|---|---|
| `pricing_TEST_SAMPLE_first1000.csv` | 1,000 | **Literal first 1,000 rows.** Code-path smoke testing. | sku 0–165; **17 of 32** categories; mean qty 26.75 |
| `pricing_TEST_SAMPLE_random1000.csv` | 1,000 | Random sample, `seed=42`. Anything distribution-sensitive. | **21 of 32** categories; mean qty 22.88 (full data: 22.82) |

### ⚠ Read this before using the first-1000 file

**`pricing.csv` is sorted by `sku` ascending.** The first 1,000 rows are therefore not a random
sample — they cover sku 0–165 out of 74,998, and their `category` mean is 19.3 against the
full-data 12.6.

| Use case | Which file |
|---|---|
| "Does the pipeline run without crashing?" | `first1000` — matches the brief, and bias is irrelevant here |
| Shape/dtype/schema checks | `first1000` |
| Anything reading a distribution or a metric | `random1000` |
| Any accuracy claim | **Neither — use the full dataset** |

Both are for **testing only**. Never report a metric computed on 1,000 rows: with 800 training
rows, even linear regression scores R² = −0.18. See `TESTING_CHECKLIST.md` §5.2.

### Usage

```bash
# Fast checks, no TensorFlow required
python run_sanity_checks.py --data ../data/pricing_TEST_SAMPLE_first1000.csv

# Full dataset with baselines (~2 min)
python run_sanity_checks.py --data ../data/pricing.csv --full --baselines
```

To point the pipeline at a sample, temporarily edit `config.py`:

```python
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'pricing_TEST_SAMPLE_first1000.csv')
```

Better: make it overridable so nobody accidentally commits a config pointing at the sample —

```python
DATA_FILE = os.environ.get('PRICING_DATA',
                           os.path.join(PROJECT_DIR, 'data', 'pricing.csv'))
```

---

## 5. Dataset reference card

```
pricing.csv — 500,000 rows × 6 columns — 19 MB — 0 nulls — 0 duplicates — sorted by sku
```

| Column | Type | Role | Range | Corr w/ target |
|---|---|---|---|---|
| `sku` | int64 | **ID — should not be a feature** | 0–74,998 (74,999 unique) | +0.00008 |
| `price` | float64 | Feature | 0.00–274.95 | −0.107 |
| `quantity` | int64 | **TARGET** | 1–4,165 | — |
| `order` | int64 | Feature | 0–446 | +0.041 |
| `duration` | float64 | Feature — **strongest** | 1.00–31.93 | **+0.598** |
| `category` | int64 | Feature — nominal, 32 levels | 0–32 | +0.154 |

**Target:** mean 22.82 · median 7 · mode 1 (15.6%) · **var 2,528** · skew 10.1 · kurtosis 283
**Split (seed 42):** train 400,000 / test 100,000 · `var(y_test) = 2,478.8`
**Conversion:** `MSE_test ≈ 2478.8 × (1 − R²_test)`

---

## 6. Run log

Add one row per training run. **Never overwrite** — and note that `metrics.csv` currently
overwrites itself (I-01), so **this table is the only durable record until that is fixed.**

| Run | Date | Who | Config | R² train | R² test | MSE test | Time | Notes |
|---|---|---|---|---|---|---|---|---|
| 001 | 2026-09-17 | team member | 3×sigmoid 64/32/16, Adam 1e-3, batch 32, 1 pass, 5 feats | not recorded | **0.3235** | not recorded | not recorded | Pipeline verified end to end. **Below the 0.3744 linear baseline** (I-04). Not reproducible (I-02) |
| 002 | — | — | *after fixes 1–4 in `PROJECT_NOTES.md` §3* | — | *expect ~0.54* | *expect ~1,150* | — | Planned |

### Reference points (not training runs)

| Model | R² test | MSE test | Source |
|---|---|---|---|
| Mean predictor | 0.0000 | 2,478.8 | `run_sanity_checks.py --full --baselines` |
| **Linear regression** | **0.3744** | 1,550.7 | same — **the number to beat** |
| Gradient boosting | 0.5885 | 1,020.0 | same — realistic ceiling |
| 3×sigmoid, 1 pass | ~0.35 | ~1,610 | MLPRegressor proxy — matches run 001 |
| 3×relu, 1 pass | ~0.54 | ~1,150 | MLPRegressor proxy — **one word changed** |

### What to record every run

Date/time · git commit · who · **full config** (layers, activation, LR, batch, epochs, features)
· R² train **and** test · MSE train **and** test · training time · package versions · anything unusual.

Run 001 recorded only test R². That is not enough to reproduce or debug it — record the rest going forward.

---

## 7. Quick start for a new team member

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Verify the data contract (5 s, no TensorFlow)
python run_sanity_checks.py --data ../data/pricing_TEST_SAMPLE_first1000.csv

# 3. Verify the pipeline runs (< 60 s)
python test_integration.py

# 4. Only then, the full run (10-40 min)
python main.py
```

Then read, in order: `TESTING_CHECKLIST.md` §1 (gate sequence) → `PROJECT_NOTES.md` Part 3
(fix order) → this file §6 (run log).

---

## 8. Status summary

**Works:** data loads cleanly (0 nulls, 0 duplicates, no schema surprises) · train/test split is
correct with no leakage · scaler is fit on train only (D-07) · the metric functions are correct ·
the pipeline runs end to end and produces a genuine, non-trivial score.

**Needs work before run 002:** seeding (I-02) · `metrics.csv` append (I-01) · scaler persistence
(I-03) — about 12 minutes of work total, and all three are prerequisites for any result being
believable.

**Biggest opportunity:** `sigmoid` → `relu`, one word, measured at **+0.19 R²** (D-01).

**Main caveat on the current result:** 0.3235 is a working pipeline, not yet a good model — it
sits below the 0.3744 linear-regression baseline. The near-zero train/test gap reflects
underfitting rather than good generalization (checklist M6). The path from 0.32 to ~0.54 is
known, cheap, and listed in `PROJECT_NOTES.md` §3.
