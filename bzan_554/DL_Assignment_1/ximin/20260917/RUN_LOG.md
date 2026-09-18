# RUN LOG & MONITORING NOTES

**Maintainer:** Ximin · **Repo:** bgdrake03/Deep-Learning-Assignment-1
**Purpose:** single source of truth for what was run, when, how long it took, and what broke.

---

## 1. Run history

### RUN-001 — baseline
| Field | Value |
|---|---|
| Run ID | RUN-001 |
| Date/time | Not recorded at wall-clock level — see note |
| Operator | Rebecca |
| Commit | unknown — **record this from now on** (`git rev-parse --short HEAD`) |
| Config | 64/32/16 sigmoid, Adam lr=1e-3, batch 32, epochs 1 |
| Samples | 400,000 train / 100,000 test |
| Batches | 12,500 |
| **Wall time** | **~48 s** |
| Train R² | 0.3117 |
| **Test R²** | **0.3235** |
| Train MSE | 1768.13 |
| Test MSE | 1676.84 |
| Test RMSE | 40.95 |
| Peak RAM | **not captured** (no instrumentation) |
| Artefacts | `results/trained_model.h5`, `results/metrics.csv` |
| Status | Completed, no errors |

**Notes on RUN-001**
- Ran clean. Pipeline is sound: correct split, no leakage, scaler fit on train only.
- Test R² > train R² is expected here and is *not* evidence of excellence — the model sits below the 0.3744 linear baseline, which indicates underfitting.
- `metrics.csv` holds this run **only**, and the next run will overwrite it. Copy it aside before running anything (see §4).
- Unseeded — RUN-001 is **not reproducible**. Treat 0.3235 as approximate.

### RUN-002 — template, copy for each new run
| Field | Value |
|---|---|
| Run ID | RUN-002 |
| Date/time start | `YYYY-MM-DD HH:MM` |
| Date/time end | |
| Operator | |
| Commit hash | |
| What changed vs. previous | |
| Wall time | |
| Train R² / Test R² | |
| Train MSE / Test MSE | |
| Peak RAM (MB) | |
| Errors / warnings | |
| Artefacts | |
| Verdict | better / worse / neutral vs. RUN-001 |

> One change per run. Two changes at once and you cannot attribute the result.

---

## 2. Check-in schedule (monitoring without coding)

Full training is ~48 s, so "check in periodically" is really about **longer tuning runs** (more epochs). Scale accordingly.

| Run length | Check in | What to look at |
|---|---|---|
| < 2 min | start + end | Did it finish; final R² |
| 2–30 min | every 5 min | Batch counter advancing; RAM flat; no NaN |
| > 30 min | every 15 min | Same + is R² actually improving |

**At each check-in, record three things:** timestamp, batch counter, anything unusual on screen.

### Red flags — escalate immediately
| Symptom | Means | Tell |
|---|---|---|
| `nan` / `inf` in MSE output | Diverged — run is dead, stop it | Rebecca |
| MSE rising steadily | Learning rate too high | Rebecca |
| RAM climbing without plateau | Memory leak / full-file load | Ansley |
| Batch counter stops advancing | Hung | Ansley |
| Final R² ≤ 0 | Pipeline bug, not a tuning issue | Both |
| Final R² > 0.95 | Too good — suspect leakage | Both |
| Test R² far above train R² | Check split integrity | Both |

**Healthy run:** batch counter climbing steadily, RAM flat after the first few hundred batches, MSE noisy but trending down, final test R² between 0.30 and 0.60.

---

## 3. Issues log

| # | Date | Issue | Severity | Owner | Status |
|---|---|---|---|---|---|
| I-01 | 09-17 | `metrics.csv` truncated each run; run 001 will be erased | High | Ansley | Open |
| I-02 | 09-17 | No memory tracking (`psutil` absent from all files) | High | Ansley | Open |
| I-03 | 09-17 | Per-batch `mse_history` computed then discarded — no learning curve | High | Rebecca | Open |
| I-04 | 09-17 | Data loaded whole, not chunked — ingestion not incremental | High | Ansley | Open |
| I-05 | 09-17 | Scaler not saved; `.h5` model cannot score new data | Med | Rebecca | Open |
| I-06 | 09-17 | No global seed — runs not reproducible, comparisons invalid | Med | Rebecca | Open |
| I-07 | 09-17 | README frames 0.3235 as "excellent generalization"; below 0.3744 linear baseline | Med | Rebecca | Open |
| I-08 | 09-17 | Ximin's docs + sample CSV not committed to repo | Med | Ximin | Open |
| I-09 | 09-17 | `.h5` legacy format may fail on Keras 3 | Low | Rebecca | Open |
| I-10 | 09-17 | `utils.py` is a one-line stub but is listed as a component | Low | — | Open |
| I-11 | 09-17 | `calculate_moving_average` imported in `main.py` but never called | Low | Rebecca | Open |

---

## 4. Results organization conventions

*(Ximin executes manually — proposed standard so files stay sortable.)*

```
results/
├── run_001_20260917_1430/
│   ├── metrics.csv
│   ├── mse_history.csv
│   ├── memory_log.csv
│   ├── trained_model.h5
│   ├── scaler.pkl
│   └── notes.md
├── run_002_20260918_0915/
└── archive/
```

**Naming:** `run_<NNN>_<YYYYMMDD>_<HHMM>_<descriptor>`
Examples: `run_002_20260918_0915_relu`, `run_003_20260918_1030_relu_10epochs`

Rules:
- Zero-pad the run number; `YYYYMMDD` sorts chronologically, `MM-DD-YY` does not
- One folder per run — never overwrite a previous run's folder
- Descriptor names *the one thing that changed*
- Immediately after a run: `cp results/metrics.csv results/run_XXX_<date>/` before anyone runs again (works around I-01)
- Never commit `.h5` files if they get large — check `.gitignore`

---

## 5. Code locations

| File | Purpose | Key lines |
|---|---|---|
| `code/config.py` | All hyperparameters and paths | `BATCH_SIZE=32`, `EPOCHS=1`, `LEARNING_RATE=0.001`, `RANDOM_SEED=42` |
| `code/model.py` | Network definition | `build_model()` — 64/32/16 sigmoid + linear |
| `code/data_handler.py` | Load, split, scale, batch | `:17` full-file read (I-04); `create_mini_batches()` |
| `code/evaluate.py` | R², MSE, moving average | `calculate_r2`, `calculate_mse`, `calculate_moving_average` |
| `code/main.py` | Training loop + logging | `:43` `train_on_batch`; `:45` mse_history (I-03); `:71` save; `:84` to_csv (I-01) |
| `code/test_integration.py` | Small-sample smoke test | 1,000 samples, ~3 s |
| `code/utils.py` | Stub — empty (I-10) | — |
| `rebecca/model-documentation.md` | Model guide | — |
| `rebecca/network-architecture.md` | Architecture spec | — |

**Change a hyperparameter in `config.py` only.** Never edit values inline in `main.py` — that is how runs become untraceable.

---

## 6. Dependencies

Python ≥ 3.9

| Package | Version | Used for |
|---|---|---|
| tensorflow | ≥ 2.13.0 | Keras model, `train_on_batch` |
| numpy | ≥ 1.24.0 | Arrays, moving average |
| pandas | ≥ 2.0.0 | CSV I/O, batching |
| scikit-learn | ≥ 1.3.0 | split, StandardScaler, metrics |
| **psutil** | **≥ 5.9.0** | **memory tracking — NOT YET ADDED (I-02)** |
| joblib | ≥ 1.3.0 | saving the scaler (I-05) |
| matplotlib | ≥ 3.7.0 | learning-curve plot for slide 9 |

```bash
pip install -e .
# or
pip install "tensorflow>=2.13" "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.3" "psutil>=5.9" "joblib>=1.3" "matplotlib>=3.7"
```

**Record the exact TF version used for each run** — `.h5` behaviour differs across Keras 2/3 (I-09).

---

## 7. Quick reference — expected values

| Check | Expected | Action if violated |
|---|---|---|
| Rows loaded | 500,000 | Wrong file |
| Train / test | 400,000 / 100,000 | Check `TEST_SPLIT` |
| Batch count | 12,500 | Check `BATCH_SIZE` |
| Parameters | 3,009 | Architecture changed |
| Wall time | 45–60 s (1 epoch) | 10× longer → investigate |
| Test R² | 0.30 – 0.60 | < 0.30 underfit; > 0.95 suspect leakage |
| Test MSE | 1,000 – 2,500 | > 2,478.8 = worse than predicting the mean |
| Peak RAM | < 2 GB | Climbing → leak |
| **Bar to beat** | **R² > 0.3744** | Below = worse than linear regression |
