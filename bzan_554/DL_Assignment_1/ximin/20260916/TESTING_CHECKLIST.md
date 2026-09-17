# Testing Checklist — Pricing Quantity Regression

**Owner:** xzeng3@utk.edu
**Created:** 2026-09-17
**Status:** Active — v1.0
**Applies to:** `config.py`, `data_handler.py`, `model.py`, `main.py`, `evaluate.py`, `test_integration.py`, `utils.py`

---

## 0. How to use this document

Checks are grouped by **when** they run. Each has an ID you can quote in commit messages,
issue tickets, and the run log in `TEAM_LOG.md` (e.g. *"blocked by D4"*).

| Tier | When it runs | Runtime | Gate |
|---|---|---|---|
| **D** Data integrity | Before any training, and whenever `pricing.csv` changes | < 5 s | **Blocking** |
| **F** Feature/target | Once per dataset version | < 5 s | Advisory |
| **P** Pipeline | Every code change to `data_handler.py` / `config.py` | < 10 s | **Blocking** |
| **S** Smoke test | Every commit, before any full run | < 60 s | **Blocking** |
| **B** Baselines | Once per dataset; re-run when features change | ~2 min | **Blocking for claims** |
| **M** Model/training | Every full training run | 10–40 min | **Blocking** |
| **R** Reproducibility | Before reporting any result externally | 2× full run | **Blocking for claims** |

Tiers D, F, P and B are automated in **`run_sanity_checks.py`** and need no TensorFlow:

```bash
python run_sanity_checks.py                                        # fast, on the 1k sample
python run_sanity_checks.py --data ../data/pricing.csv --full --baselines   # full dataset
```

Exit code `0` = pass, `1` = at least one FAIL.

---

## 1. Testing order (the gate sequence)

Do not skip ahead. Each gate is cheap relative to the one after it, and every gate has
caught a real problem in this project at least once.

```
D. Data integrity        (5 s)     ──┐
F. Feature sanity        (5 s)       │ run_sanity_checks.py
P. Pipeline correctness  (10 s)      │
B. Baselines             (2 min)   ──┘
        │
        ▼
S. Smoke test on 1,000-row sample    (< 60 s)   test_integration.py
        │
        ▼
M. Full training run                 (10–40 min) main.py
        │
        ▼
R. Reproducibility + reporting       (2× full run)
```

**Rule:** no metric leaves this repo — no email, no slide, no README — until it has passed
**R**. Run 001's `R² = 0.3235` has not passed **B** or **R**. See §7.

---

## 2. Tier D — Data integrity (blocking, automated)

Reference profile of `pricing.csv`, measured 2026-09-17: **500,000 rows × 6 columns, 0 nulls, 0 duplicates.**

| ID | Check | Expected | On failure |
|---|---|---|---|
| D1 | Column names and order | `sku, price, quantity, order, duration, category` | STOP — schema drift breaks `INPUT_FEATURES` |
| D2 | Row count | 500,000 full / 1,000 sample | STOP — wrong file loaded |
| D3 | All columns numeric | `int64` ×4, `float64` ×2 | STOP — `StandardScaler` will throw |
| D4 | Null count | **0** | STOP — there is no imputation step; one NaN → NaN loss, silently |
| D5 | Infinity count | **0** | STOP — same as D4 |
| D6 | Exact duplicate rows | **0** | WARN — duplicates straddle the split and inflate test scores |
| D7 | Value ranges | see table below | STOP if outside hard bounds |
| D8 | Target positivity | all `quantity` ≥ 1 | STOP — it is a count; a 0 or negative means corruption |
| D9 | Distribution drift vs full data | mean within ±0.5σ | WARN — sample is unrepresentative |

### D7 — Expected ranges (full dataset, verified)

| Column | Min | 25% | Median | 75% | 99% | Max | Hard bounds | Notes |
|---|---|---|---|---|---|---|---|---|
| `sku` | 0 | — | — | 55,040 | — | 74,998 | `[0, 74998]` | 74,999 unique; ~6.7 rows per sku |
| `price` | 0.000 | 0.132 | 0.264 | 0.788 | 8.80 | 274.95 | `[0, 300]` | Heavy right tail |
| `quantity` **(target)** | 1 | 2 | 7 | 22 | 228 | 4,165 | `[1, 5000]` | Count; skew **10.1**, kurtosis **283** |
| `order` | 0 | 1 | 5 | 13 | 151 | 446 | `[0, 500]` | Integer count |
| `duration` | 1.000 | 1.191 | 1.496 | 2.031 | 5.70 | 31.93 | `[1, 35]` | Strongest predictor |
| `category` | 0 | — | — | 17 | — | 32 | `[0, 32]` | 32 nominal levels |

### Key target statistics (drive every sanity range below)

```
mean(quantity)      = 22.82
var(quantity)       = 2,528      <- full data
var(quantity_test)  = 2,478.8    <- the 100k test split, seed 42
skew                = 10.10
mode                = 1  (78,154 rows, 15.6% of the dataset)
```

> **The single most useful number in this document is `var(y_test) = 2478.8`.**
> Because `R² = 1 − MSE/var(y)`, every MSE you see has a predictable partner:
> `MSE_test ≈ 2478.8 × (1 − R²_test)`.

---

## 3. Tier F — Feature & target sanity (advisory, automated)

| ID | Finding | Expected | Implication |
|---|---|---|---|
| F1 | `corr(sku, quantity) = +0.00008` | ≈ 0 | `sku` is a **row identifier, not a feature**. It contributes pure noise to 1/5 of the input layer. Drop it. |
| F2 | File is **sorted by `sku` ascending** | — | `head(n)` is **not** a random sample. See §6. |
| F3 | Target skew = 10.1 | — | MSE is dominated by a few thousand huge orders. Test `log1p(quantity)` as the training target. |
| F4 | `category` is fed as a raw integer | — | The network is told category 30 > category 3. Needs one-hot or an embedding. |

### Correlation with `quantity` (full dataset)

| Feature | Corr | Verdict |
|---|---|---|
| `duration` | **+0.598** | Dominant signal — carries most of the achievable R² |
| `category` | +0.154 | Real but weak; encoding is wrong (F4) |
| `price` | −0.107 | Weak negative — economically sensible |
| `order` | +0.041 | Very weak |
| `sku` | +0.00008 | **Noise** |

---

## 4. Tier P — Pipeline correctness (blocking, automated)

| ID | Check | Expected | Why it matters |
|---|---|---|---|
| P1 | Feature count after dropping target | **5** = `config.INPUT_FEATURES` | Mismatch → cryptic Keras shape error |
| P2 | Split sizes, seed 42 | train 400,000 / test 100,000; **zero index overlap** | Overlap = leakage = meaningless score |
| P3 | Scaler fit on **train only** | train mean ≈ 0, std ≈ 1; test mean **near but ≠ 0** (max \|mean\| ≈ 0.0046) | Test mean **exactly** 0 proves the scaler was refit on test data |
| P4 | Batch shapes from `create_mini_batches` | `X (32, 5)`, `y (32,)`; final batch may be short | Last batch is 400,000 mod 32 = **0**, so all 12,500 batches are full |
| P5 | Batch count, full train set | **12,500** batches | `print` every 100 → 125 progress lines |
| P6 | `X`/`y` alignment after scaling | Row *i* of `X_train` must match row *i* of `y_train` | `X_train` is rebuilt with a fresh `RangeIndex` while `y_train` keeps the original shuffled index. Both are accessed with `.iloc`, so this is **currently correct — but fragile.** Any switch to `.loc` or a merge silently scrambles the labels. |

> **P6 is the highest-risk latent bug in the repo.** It is correct today by coincidence of
> using `.iloc` in both places. Add an explicit `y_train = y_train.reset_index(drop=True)`
> in `load_and_split_data()` to make the guarantee structural rather than accidental.

---

## 5. Tier S — Smoke test (blocking, every commit)

Run on the 1,000-row sample only. **Purpose: prove the code runs end to end. Not to judge accuracy.**

| ID | Check | Expected | On failure |
|---|---|---|---|
| S1 | `python config.py` | No error; `RESULTS_DIR` created | Path/permission issue |
| S2 | `python evaluate.py` | `R² = 0.9880`, `MSE = 0.0220` (built-in self-test) | Metric functions broken |
| S3 | `python model.py` | Summary prints **3,041** total params | Architecture drift — see §5.1 |
| S4 | `python data_handler.py` | `Training set size: 800`, `Test set size: 200` | Split broken |
| S5 | `python test_integration.py` | Completes in < 60 s, prints `[SUCCESS]` | Pipeline broken |
| S6 | Loss is finite | No `nan` / `inf` in any batch loss | LR too high, or a D4/D5 escape |
| S7 | Loss decreases | Last-10-batch mean < first-10-batch mean | Model not learning at all |
| S8 | Prediction shape | `(n, 1)` from Keras; flatten before metrics | Silent broadcasting → garbage R² |
| S9 | Moving average length | `len(mse_history) − window + 1` = 25 − 10 + 1 = **16** | `np.convolve` misuse |

### 5.1 — S3 parameter count (verify by hand)

```
Dense(64)  : 5 × 64 + 64  =   384
Dense(32)  : 64 × 32 + 32 = 2,080
Dense(16)  : 32 × 16 + 16 =   528
Dense(1)   : 16 ×  1 +  1 =    17
                            -------
Total trainable parameters  = 3,041
```

If `model.summary()` prints anything other than **3,041**, `config.py` was edited.

### 5.2 — Expected smoke-test *metrics* (read this before panicking)

On 800 training rows, `test_integration.py` will report R² roughly in **−0.5 to +0.3**, and
**negative R² is a PASS at this tier.** With 25 gradient steps against a target of skew 5,
the model has barely moved off its initialisation. On the 1,000-row sample, even a linear
regression scores **R² = −0.18**.

> **S-tier asserts "it ran", not "it learned." Do not tune hyperparameters on smoke-test numbers.**

---

## 6. Tier B — Baselines (blocking for any performance claim)

Measured on the full dataset, seed 42, 80/20 split. **Reproduce these before trusting any NN score.**

| ID | Model | R² train | R² test | MSE test | Role |
|---|---|---|---|---|---|
| B1 | Mean predictor | 0.0000 | **0.0000** | 2,478.8 | Absolute floor. R² ≈ 0 by construction — if not, the metric code is broken. |
| B2 | Linear regression | 0.3784 | **0.3744** | 1,550.7 | **The number a neural network must beat to justify existing.** |
| B3 | Gradient boosting | 0.6378 | **0.5885** | 1,020.0 | Realistic target for this feature set. |
| B4 | 3×sigmoid net, 1 pass | ~0.34 | **~0.33** | ~1,660 | Current architecture. **Below B2.** |
| B5 | 3×relu net, 1 pass | ~0.54 | **~0.54** | ~1,150 | Identical net, one word changed. |

**Acceptance criteria for a training run:**

- R²_test < 0.00 → **FAIL**, worse than predicting the mean. Pipeline bug.
- 0.00–0.37 → **FAIL**, worse than linear regression. Do not report as a success.
- 0.37–0.54 → **MARGINAL**, beats linear but loses to a one-word activation fix.
- 0.54–0.59 → **PASS**, competitive with the tree ensemble.
- \> 0.65 → **SUSPICIOUS**, above the GBM ceiling. Check for leakage before celebrating.
- \> 0.95 → **ALARM**. Near-certainly the target leaked into the features.

---

## 7. Tier M — Full training run (blocking, every run)

| ID | Check | Expected | Interpretation |
|---|---|---|---|
| M1 | Batches processed | 12,500 | Fewer = data truncated |
| M2 | Training time | 8–40 min on CPU | 10× faster ⇒ it silently ran on a subset |
| M3 | First-batch loss | 500–3,000 | Starts near `var(y)` ≈ 2,528 |
| M4 | Final-batch loss | 800–1,800 | Noisy; use the 100-batch moving average |
| M5 | Loss curve trend | Monotone-ish decrease over the moving average | Flat ⇒ vanishing gradients (see `PROJECT_NOTES.md` D-01) |
| M6 | R²_train vs R²_test gap | \|gap\| < 0.05 | See M6 note below |
| M7 | MSE/R² consistency | `MSE_test ≈ 2478.8 × (1 − R²_test)` | Mismatch ⇒ metric bug or shape error |
| M8 | Artifacts written | `results/trained_model.h5`, `results/metrics.csv` | Missing ⇒ save failed |
| M9 | Predictions not constant | `std(y_pred) > 1.0` | Constant output = collapsed to the mean |
| M10 | Prediction range | roughly `[0, 300]` | The net cannot reach 4,165; expect the tail to be crushed |

### M6 — "Good generalization" needs a caveat

Run 001 reported train ≈ test, and this was described as *"good generalization."* That reading
is not safe. **Test R² slightly above train R², at a score below the linear baseline, is the
signature of underfitting, not of a well-regularised model.** A model that has barely learned
anything generalises perfectly — it is equally mediocre everywhere.

A small train/test gap is only evidence of good generalization **once the score clears B2 (0.3744).**
Until then, M6 passing means "not overfitting yet," nothing more.

---

## 8. Tier R — Reproducibility (blocking before reporting)

| ID | Check | Expected | Status |
|---|---|---|---|
| R1 | Two runs, identical config → identical R² | Δ < 0.001 | **Not achievable today** — see below |
| R2 | Global seeds set | `random`, `numpy`, `tensorflow` all seeded | **MISSING** (issue I-02) |
| R3 | `metrics.csv` records the config used | LR, batch size, layers, activation | **MISSING** — only records metrics |
| R4 | `metrics.csv` is appended, not overwritten | One row per run | **FAILS** — run 002 will erase run 001 (issue I-01) |
| R5 | Scaler persisted alongside the model | `results/scaler.pkl` | **MISSING** — saved model cannot score new data (issue I-03) |
| R6 | Package versions pinned | `requirements.txt` | Added — see file |

> `config.RANDOM_SEED = 42` is passed **only** to `train_test_split`. Keras weight
> initialisation is **unseeded**, so run 002 will not reproduce run 001's 0.3235 even with
> zero code changes. Expect run-to-run drift of roughly ±0.02 R² from initialisation alone.
> **This means 0.3235 is currently an unverifiable number.**

---

## 9. Regression checklist for specific code changes

| If you change… | Re-run |
|---|---|
| `pricing.csv` | D, F, B, then full M |
| `config.py` layer sizes | S3 (param count), then S, M |
| Activation function | S, B, M — and log it in `PROJECT_NOTES.md` |
| `data_handler.py` | P (all), S4, S5, then B |
| `evaluate.py` | S2, plus M7 consistency |
| `BATCH_SIZE` | P4, P5, M1 |
| Adding/removing a feature | D1, P1, **B (re-baseline)**, M |

---

## 10. Known gaps in current test coverage

Documented so nobody assumes these are covered:

1. **No unit tests.** `test_integration.py` is a script with prints, not assertions — it
   cannot fail. It prints `[SUCCESS]` unconditionally as long as no exception is raised.
   A model returning all-zeros would still "pass."
2. **No validation set.** There is train and test only. Any tuning across runs selects on the
   test set, which inflates it into an optimistic estimate.
3. **`utils.py` is empty** (one comment line, zero functions) and is imported by nothing.
4. **No CI.** Nothing runs these checks automatically on push.
5. **No test for `EPOCHS` / `VERBOSE`** — both are defined in `config.py` and never read
   by any module. Dead configuration that implies capability the code does not have.
