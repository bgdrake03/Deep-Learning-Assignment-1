# Project Notes — Decisions, Issues, and Fixes

**Owner:** xzeng3@utk.edu
**Started:** 2026-09-17
**Purpose:** Running record of *why* the code looks the way it does, what is broken, and what was
changed. Append to the bottom of each section; never rewrite history — strike through and date instead.

---

## Part 1 — Architecture decisions and their justification status

Each decision is rated:

- **Justified** — a reason exists and the evidence supports it
- **Inherited** — carried over from a template/tutorial; no evidence either way
- **Questionable** — evidence points the other way

---

### D-01 · Why sigmoid activation? — **QUESTIONABLE**

**Current code:** `model.py` uses `activation='sigmoid'` on all three hidden layers.
**Stated reason:** none recorded.

**Assessment — this is the single highest-impact problem in the repo.**

Sigmoid squashes every input into (0, 1) and has a maximum derivative of **0.25**, at x = 0.
Stacking three of them means gradients reaching the first layer are multiplied by at most
`0.25³ ≈ 0.016`. In practice, with saturated units, it is far worse. The first hidden layer
— the one that has to learn what `duration` means — trains roughly **60× slower** than the
output layer.

This is compounded by the training setup: **one single pass** over the data (§D-05). The
network is given 12,500 gradient steps total and a vanishing-gradient architecture that needs
far more than that to converge.

**Measured impact.** Same architecture, same optimiser, same learning rate, same single pass,
same seed — only the activation string changed:

| Activation | R² train | R² test |
|---|---|---|
| `sigmoid` | 0.338 | **0.350** |
| `relu` | 0.537 | **0.537** |

**A one-word change is worth +0.19 R².** For context, the sigmoid version (0.35) loses to a
plain linear regression (0.374), while the ReLU version (0.537) beats it comfortably.

**Recommendation:** switch to `relu` before any further hyperparameter work. Everything else
on the tuning list is worth less than this one edit. Keep sigmoid only if there is a hard
requirement nobody has written down — and if so, record it here.

**Historical note:** sigmoid was the standard hidden activation until ~2012. Essentially every
modern reference net uses ReLU or a variant for hidden layers. Sigmoid's remaining correct use
is a *binary classification output* layer — not hidden layers in a regression net.

---

### D-02 · Why 3 hidden layers (64 → 32 → 16)? — **INHERITED**

**Current code:** `config.py`, `HIDDEN_LAYER_1/2/3`. Total **3,041** trainable parameters.

The halving funnel 64→32→16 is a conventional tutorial default, not a derived choice. No
recorded search over depth or width.

**Assessment:** depth is not the binding constraint, so this is low priority.

- 3,041 parameters against 400,000 training rows is a ratio of ~130 rows per parameter. The
  model is **not** capacity-limited and is in no danger of overfitting — consistent with the
  observed near-zero train/test gap.
- The evidence says the bottleneck is **trainability** (D-01) and **passes over the data**
  (D-05), not capacity. A gradient-boosting model on the same 5 features reaches 0.589, so
  roughly 0.59 of signal is extractable; the net is leaving ~0.25 on the table for reasons
  that have nothing to do with layer count.
- With only 5 input features and one dominant predictor, **2 hidden layers would very likely
  match 3.** Depth mainly costs more sigmoid multiplications to vanish through.

**Recommendation:** leave as-is for now. Re-test depth *after* fixing D-01 and D-05, otherwise
any depth comparison is measuring vanishing gradients rather than capacity.

---

### D-03 · Why `sku` as an input feature? — **QUESTIONABLE**

`sku` is a row identifier spanning 0–74,998 across 74,999 unique values (~6.7 rows each).

```
corr(sku, quantity) = +0.00008
```

That is zero to four decimal places. It is passed through `StandardScaler` and fed to the
network as 1 of 5 inputs, so **20% of the input layer is pure noise.** The net must spend
capacity learning to ignore it.

Worse, it invites a subtle error: treated as a *continuous* variable, the model is told that
sku 74,998 is "large" and sku 3 is "small," which is meaningless — these are labels.

**Recommendation:** drop `sku`, set `INPUT_FEATURES = 4`. Verified harmless: linear regression
scores 0.3744 with sku and 0.3737 without — statistically indistinguishable, minus the noise.
(If per-SKU effects are genuinely wanted, that needs an embedding layer, not a scaled integer.)

---

### D-04 · Why is `category` scaled as a number? — **QUESTIONABLE**

`category` has 32 nominal levels (0–32). `StandardScaler` maps them to a continuous axis, which
tells the network that category 30 is "ten times" category 3 and that category 16 sits
"between" 15 and 17. None of that is true for nominal labels.

Its correlation with the target (+0.154) is the second-strongest in the dataset, so this is
**real signal being fed in a form the model cannot use properly.**

**Recommendation:** one-hot encode (32 columns → `INPUT_FEATURES` becomes 36 with sku dropped),
or use an embedding. One-hot is the simpler first test. Note this is likely a meaningful part
of the gap between the net and the 0.589 gradient-boosting score, since trees handle nominal
splits natively.

---

### D-05 · Why `EPOCHS = 1` / single pass? — **INHERITED, and partly dead code**

`config.py` sets `EPOCHS = 1`, commented *"For incremental learning, typically 1 pass through data."*

Two separate problems:

1. **`EPOCHS` is never read by any module.** `main.py` loops once over `create_mini_batches`
   directly. Changing `EPOCHS` to 50 in the config would do **nothing**. This is dead
   configuration that implies a capability the code does not have. (Same for `VERBOSE`.)
2. The premise is shaky. One pass is normal for *streaming* learning, where data arrives
   continuously and cannot be revisited. Here the entire 500k dataset is a static CSV on disk,
   loaded fully into memory. There is no reason not to make multiple passes — the constraint
   being honoured does not exist.

Combined with sigmoid (D-01), one pass is why the model underfits: 12,500 steps through a
vanishing-gradient stack is not enough to converge.

**Recommendation:** make `EPOCHS` functional by wrapping the batch loop, and **reshuffle each
epoch** (currently batches are carved in a fixed order every time). Expect a substantial gain
from 5–10 epochs, independent of the activation fix.

---

### D-06 · Why raw `quantity` as the target? — **QUESTIONABLE**

`quantity` is right-skewed count data:

```
mean 22.8 | median 7 | mode 1 (15.6% of all rows) | max 4,165 | skew 10.1 | kurtosis 283
```

Under MSE, a single row with quantity 4,165 predicted as 23 contributes an error of
~17 million — as much as ~170,000 typical rows combined. **The loss is effectively being
optimised for a handful of outliers**, while 50% of the data (quantity ≤ 7) is nearly ignored.

This also caps the achievable R², since R² on a skewed target is dominated by the tail.

**Recommendation:** test `log1p(quantity)` as the training target, inverting with `expm1` for
reporting. **Important:** if you do this, R² must still be reported on the *original* scale, or
it is not comparable to any number in this repo — R² in log-space is a different, and usually
flattering, quantity. Log the comparison here when tested.

---

### D-07 · Why StandardScaler, fit on train only? — **JUSTIFIED** ✔

`data_handler.py` fits on `X_train` and only transforms `X_test`. This is correct and is the
one clearly right modelling decision in the pipeline — it prevents test statistics leaking into
training. Verified by check P3: test-set means are near zero but not exactly zero, which is the
fingerprint of a properly one-sided fit.

Caveat: `price` (max 275) and `quantity` have heavy tails, so standardisation leaves extreme
values at ~100σ. A `RobustScaler` or a log transform on `price` may behave better. Minor next
to D-01.

---

### D-08 · Why `train_on_batch` instead of `model.fit`? — **INHERITED**

A manual batch loop is the right call *if* the goal is to demonstrate incremental learning
mechanics or to log per-batch loss (which `main.py` does, into `mse_history`).

The cost: `model.fit` would provide validation-split monitoring, callbacks, early stopping, and
shuffling per epoch for free. The manual loop reimplements a slower version of a subset of that.

**Keep it** if incremental learning is an explicit project requirement — the per-batch loss
history is genuinely useful for the learning curve. **Note:** `mse_history` is currently
computed and then thrown away at the end of `main.py`; it is never saved or plotted. Persisting
it is a cheap win (see I-05).

---

## Part 2 — Issues log

Severity: **P0** blocks correct results · **P1** affects credibility of results · **P2** quality

| ID | Sev | Issue | File | Status |
|---|---|---|---|---|
| I-01 | **P0** | `metrics.csv` is **overwritten** every run | `main.py:86` | Open |
| I-02 | **P0** | No global seeding → results not reproducible | `main.py`, `config.py` | Open |
| I-03 | **P0** | Scaler never saved → model unusable on new data | `data_handler.py` | Open |
| I-04 | **P1** | Result reported below the linear baseline | run 001 | Open |
| I-05 | P2 | `mse_history` computed, never saved | `main.py:39` | Open |
| I-06 | P2 | `EPOCHS`, `VERBOSE` defined but never read | `config.py:26-27` | Open |
| I-07 | P2 | `utils.py` is empty but claimed as documented | `utils.py` | Open |
| I-08 | P2 | `test_integration.py` loads all 500k rows to test on 1,000 | `test_integration.py:22` | **Fixed** |
| I-09 | P2 | `DATA_FILE` expects `../data/pricing.csv`; files are flat | `config.py:12` | Open |
| I-10 | P2 | Integration test has no assertions — cannot fail | `test_integration.py` | Open |
| I-11 | P2 | `.h5` save format is legacy in Keras 3 | `config.py:32` | Open |
| I-12 | P2 | Latent index-alignment fragility between `X` and `y` | `data_handler.py:36` | Open |
| I-13 | P2 | `from config import *` wildcard imports | 5 files | Open |

---

### I-01 · `metrics.csv` is overwritten every run — **P0**

```python
results_df.to_csv(METRICS_SAVE_PATH, index=False)   # main.py:86 — mode='w'
```

The timestamp logging added this week records *when* a run happened, but the file is written in
truncate mode. **Run 002 will silently erase run 001's 0.3235 result**, which is the only record
of it. The timestamp gives a false sense of history: it timestamps a file of size one.

The same applies to `trained_model.h5` — the next run overwrites the trained model with no backup.

**Fix:** append, and include the config in each row.

```python
results_df.to_csv(METRICS_SAVE_PATH, mode='a',
                  header=not os.path.exists(METRICS_SAVE_PATH), index=False)
```

Better: write one **row per run** (wide format) rather than the current long metric/value
layout, so runs can be compared with a single `pd.read_csv`. And version the model path with
the run timestamp: `trained_model_20260917_1432.h5`.

**Do this before run 002.** It is ~3 lines and it is the difference between having an
experiment log and not.

---

### I-02 · No global seeding — **P0**

`RANDOM_SEED = 42` reaches `train_test_split` and nothing else. Keras weight initialisation,
dropout, and shuffling are unseeded.

**Consequence:** re-running `main.py` with zero code changes will not reproduce 0.3235. Drift of
roughly ±0.02 R² from initialisation alone. So when run 002 returns 0.34, **you cannot tell
whether your change helped or the dice landed differently.** Every comparison from here on is
uninterpretable until this is fixed.

**Fix** — at the very top of `main.py`, before importing TensorFlow:

```python
import os, random
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
import numpy as np;        np.random.seed(42)
import tensorflow as tf;   tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)    # Keras 3: seeds all three at once
```

For bit-exact runs also set `tf.config.experimental.enable_op_determinism()` — slower, but
worth it while debugging.

---

### I-03 · Scaler is never persisted — **P0**

`load_and_split_data()` creates a `StandardScaler`, fits it, and lets it go out of scope. The
saved `trained_model.h5` therefore **cannot score a single new row**: the network expects
standardised inputs and the transform that produced them no longer exists.

The model was trained on `(x − μ)/σ` and any new data is raw. Feeding raw values to it produces
confident nonsense — no error, just wrong numbers.

**Fix:** return the scaler and save it next to the model.

```python
import joblib
joblib.dump(scaler, os.path.join(RESULTS_DIR, 'scaler.pkl'))
```

Note the μ and σ are computed from the training split, so they cannot be recovered later from
the full CSV — they depend on the seed. If the seed changes before this is fixed, run 001's
model is permanently unusable.

---

### I-04 · The reported result is below the linear baseline — **P1**

Run 001: **R² = 0.3235**. Linear regression on the same split: **R² = 0.3744**.

The neural network — 3,041 parameters, three hidden layers, Adam — performs **worse than a
straight line** through the same five features. Gradient boosting reaches 0.5885.

This does not mean the work is wrong; the pipeline is sound and the result is genuine. It means
**0.3235 should not be described as a success.** The correct framing for the team is:
*"end-to-end pipeline verified, baseline established at 0.32, currently below the 0.37 linear
reference — three known fixes identified."*

Note also that the *diagnosis* is already in hand, which is the valuable part: D-01 (activation)
and D-05 (single pass) account for most of the gap, and D-01 alone is measured at +0.19.

**Action:** re-frame the result in the README, and re-run after D-01. Expected R² ≈ 0.54.

---

### I-08 · Integration test loads the full dataset — **FIXED**

`test_integration.py:22` calls `load_and_split_data()`, reading all 500,000 rows (19 MB, ~10 s)
and then slicing 1,000 off the top. A smoke test that pays the full data cost defeats its own
purpose — and it means the test cannot run at all if the full CSV is unavailable.

**Fix applied:** created `pricing_TEST_SAMPLE_first1000.csv` (see `TEAM_LOG.md` §4). Point
`config.DATA_FILE` at it, or add a `data_path` argument to `load_and_split_data()`. Runtime
drops from ~10 s to ~0.05 s for the load.

**Important caveat discovered while building it:** `pricing.csv` is **sorted by `sku`
ascending**, so the first 1,000 rows cover only sku 0–165 and **17 of 32 categories**. It is
fine for "does the code run," and must **not** be used to judge accuracy. A random-sample
alternative (`pricing_TEST_SAMPLE_random1000.csv`, 21 categories, mean quantity 22.88 vs the
full-data 22.82) is provided for anything distribution-sensitive.

---

### I-10 · Integration test cannot fail

`test_integration.py` prints `[SUCCESS] INTEGRATION TEST PASSED` at the end of the function.
There is no assertion anywhere. It prints SUCCESS as long as Python raises no exception — a
model outputting all zeros, a NaN loss, or an R² of −50 all "pass."

**Fix:** add assertions matching tier S in `TESTING_CHECKLIST.md`:

```python
assert np.isfinite(mse_history).all(),        "NaN/inf loss detected"
assert y_test_pred.std() > 0.01,              "model collapsed to constant output"
assert batch_count == 25,                     f"expected 25 batches, got {batch_count}"
assert np.mean(mse_history[-10:]) < np.mean(mse_history[:10]), "loss did not decrease"
```

Do **not** assert a minimum R² at this tier — see checklist §5.2 for why negative R² is expected
and correct on 800 training rows.

---

### I-12 · Latent index-alignment fragility

```python
X_train = pd.DataFrame(X_train, columns=X.columns)   # fresh RangeIndex 0..N-1
# y_train keeps the original shuffled index from train_test_split
```

`X_train` is rebuilt from a NumPy array and gets a fresh index; `y_train` retains the shuffled
original. They no longer share labels. **This is currently harmless** because
`create_mini_batches` uses `.iloc` (positional) on both.

It is a trap, not a bug — correct by coincidence. Any future `.loc`, `join`, or `merge` will
silently pair features with the wrong targets and produce an R² near zero with no error raised.

**Fix:** `y_train = y_train.reset_index(drop=True)` (same for `y_test`) so the guarantee is
structural.

---

## Part 3 — Recommended fix order

Cheapest-first, highest-value-first. Do not reorder — each step makes the next measurable.

| # | Action | Effort | Expected effect |
|---|---|---|---|
| 1 | **I-02** global seeding | 5 min | Makes every later comparison meaningful. **Do this first.** |
| 2 | **I-01** append to `metrics.csv` | 5 min | Stops destroying run history |
| 3 | **I-03** save the scaler | 2 min | Makes the model usable |
| 4 | **D-01** sigmoid → relu | 1 min | **+0.19 R² (measured)** → ~0.54 |
| 5 | **D-05** make `EPOCHS` real + reshuffle | 20 min | Further gain from proper convergence |
| 6 | **D-03** drop `sku` | 2 min | Removes 20% input noise |
| 7 | **D-04** one-hot `category` | 30 min | Unlocks the +0.154 correlation |
| 8 | **D-06** try `log1p(quantity)` | 30 min | Stops outliers dominating the loss |

Steps 1–4 total under 15 minutes and should move test R² from 0.32 to roughly 0.54.

---

## Part 4 — Open questions for the team

1. Is **incremental/streaming learning** a hard project requirement, or an implementation choice?
   This determines whether D-05 (multi-epoch) and D-08 (`model.fit`) are open for change. Most
   of the improvement path depends on the answer.
2. Is **sigmoid** required by an assignment spec? If yes, record the source here — it changes the
   ceiling and D-01 becomes a documented constraint rather than a bug.
3. What is the **target metric**? R² is being optimised, but for a skewed count target, MAE or
   Poisson deviance may match the actual business question better.
4. Should we hold out a **validation set**? Currently all tuning decisions read the test set,
   which inflates it. A 70/15/15 split would fix it at the cost of comparability with run 001.

---

## Part 5 — Change log

| Date | Who | Change |
|---|---|---|
| 2026-09-17 | (team member) | Code skeletons, timestamp logging, docs, README; run 001 → R²_test 0.3235 |
| 2026-09-17 | xzeng3 | Full data profile of `pricing.csv`; established baselines B1–B5 |
| 2026-09-17 | xzeng3 | Created `TESTING_CHECKLIST.md`, `PROJECT_NOTES.md`, `TEAM_LOG.md`, `requirements.txt` |
| 2026-09-17 | xzeng3 | Created 1,000-row test samples (I-08 fixed); found the sku sort-order caveat |
| 2026-09-17 | xzeng3 | Created `run_sanity_checks.py`; 24 checks pass on full data, 5 warnings |
| 2026-09-17 | xzeng3 | Measured sigmoid vs relu (+0.19 R²) and NN vs linear baseline (−0.05 R²) |
