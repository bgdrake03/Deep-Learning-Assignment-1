# MASTER CHECKLIST — BZAN 554 Group Assignment 1

**Repo:** https://github.com/bgdrake03/Deep-Learning-Assignment-1 (20 commits, main)
**Owner of this document:** Ximin
**Last audited:** 2026-09-17 15:30 UTC
**Audit method:** README reviewed on GitHub + all 7 `.py` source files read line-by-line + `grep` verification of specific claims.

---

## How to read the status column

| Mark | Meaning |
|---|---|
| `[x]` | Verified complete — I confirmed it in the code or repo |
| `[~]` | Partially complete — works, but a stated sub-requirement is missing |
| `[ ]` | Not done — confirmed absent from the codebase |
| `[?]` | Cannot verify remotely — owner must confirm |

> **Important:** a few items below are marked complete in the README but are **not present in the source**. Those are flagged `[ ]` with evidence. This is not criticism of anyone's work — it is the whole point of having someone audit independently.

---

## TUESDAY — Foundation & Understanding

### Ansley — Data Exploration & Framework

- `[x]` Load `pricing.csv` and perform initial exploration
  - `[x]` Check file size and row count — **verified: 500,000 × 6, 19.0 MB**
  - `[x]` Display first/last rows and basic statistics
  - `[x]` Check for missing values and data types — **verified: zero missing**
  - `[x]` Verify categoricals are integer-encoded and numerics scaled — `StandardScaler` in `data_handler.py`
- `[~]` Create initial data handling code
  - `[ ]` **Write function to load data in chunks/batches** — `data_handler.py:17` is a plain `pd.read_csv(DATA_FILE)`. The entire 500k-row file is loaded into RAM, *then* sliced into batches. There is no `chunksize` anywhere. See **BLOCKER-1**.
  - `[x]` Prepare code structure for mini-batch reading — `create_mini_batches()` generator exists and works
- `[?]` Document data findings in a summary — no `ansley/` folder in the repo. Ask where this lives.

### Rebecca — Neural Network Architecture & Concepts

- `[x]` Research incremental learning approaches
  - `[x]` Document why incremental learning is needed — in `rebecca/model-documentation.md`
  - `[~]` Memory constraints and trade-offs — documented in prose, but **never measured** (see W-05)
- `[x]` Design and implement neural network skeleton
  - `[x]` 3 hidden layers, sigmoid — verified `model.py`: 64 → 32 → 16, all sigmoid
  - `[x]` Input 5 features — verified `INPUT_FEATURES = 5`
  - `[x]` Output 1 (quantity) — verified, linear activation
  - `[x]` Framework — TensorFlow / Keras Sequential
- `[x]` Set up repo/structure for collaboration
  - `[x]` Main training script template — `main.py`
  - `[x]` Separate modules for model, data, evaluation — `model.py`, `data_handler.py`, `evaluate.py`

### End of Tuesday

- `[x]` Data loaded and understood by all
- `[x]` Team agrees on architecture
- `[x]` Code structure/repo set up

---

## WEDNESDAY — Implementation

### Ansley — Training Loop & Integration

- `[x]` Implement incremental training function
  - `[x]` Read data in mini-batches (32) — verified
  - `[x]` Call `train_on_batch` per batch — verified `main.py:43`
  - `[~]` Loop: load → train → **save updated model** → repeat — the model is saved **once at the end** (`main.py:71`), not during the loop
- `[ ]` **Create memory tracking code** — **NOT IMPLEMENTED**
  - `[ ]` Use `psutil` — `grep -n "psutil" *.py` returns **nothing across all 7 files**
  - `[ ]` Log memory at each batch/epoch — absent
  - → See **BLOCKER-2**
- `[~]` Implement basic logging/checkpointing
  - `[ ]` Save intermediate model states — no checkpointing of any kind
  - `[x]` Log progress — batch counter every 100 batches, elapsed time captured
- `[x]` Create train/test split
  - `[x]` 80/20 — verified `TEST_SPLIT = 0.2`
  - `[x]` Test set held completely separate — **verified no leakage**: scaler is `fit_transform` on train, `transform` on test. This is correct and worth stating in the presentation.

### Rebecca — Model Training & Evaluation Metrics

- `[x]` Finalize neural network implementation
  - `[x]` Incremental training works (not whole-dataset `.fit()`) — confirmed, uses `train_on_batch`
  - `[x]` Tested on small sample — `test_integration.py`
- `[x]` Implement R² calculation — `evaluate.py`, wraps `sklearn.r2_score`, formula matches spec
- `[~]` Implement MSE tracking
  - `[~]` Calculate MSE after each batch — it **is** computed and appended to `mse_history`, but **never written to disk**; the list is discarded when `train_incremental()` returns. See **BLOCKER-3**.
  - `[x]` Moving average function exists — `calculate_moving_average()`, window=100
  - `[ ]` ...but it is **never called in `main.py`**. It is imported (`main.py:6`) and used only in `test_integration.py`.
- `[x]` Set up early integration test
  - `[x]` Runs on 1,000-sample subset
  - `[x]` Verifies pieces work together

### Ximin — Documentation & Testing Support

- `[x]` Create testing checklist (with expected ranges) — delivered
- `[x]` Document decisions made (why sigmoid, why 3 layers) — delivered, with measured evidence
- `[x]` Record issues/fixes — delivered
- `[x]` Prepare test data subset (first 1000 rows, clearly labelled) — delivered
- `[x]` Create shared log/notes document — delivered
- `[x]` Document code locations and file names — delivered
- `[x]` List dependencies/packages — delivered
- `[ ]` **All of the above are still only local — none appear in the repo.** There is no `ximin/` folder and no sample CSV in `data/`. See **BLOCKER-6**.

### End of Wednesday

- `[x]` Full pipeline works on small test data
- `[x]` Training loop runs without errors
- `[x]` R² and MSE calculations work correctly
- `[~]` Team identifies integration issues — identified, **not yet resolved**

---

## THURSDAY — Full Training & Data Collection

### Ansley — Execution & Monitoring

- `[x]` Run full training pipeline — run 001 completed
  - `[x]` Complete loop on entire dataset — 12,500 batches
  - `[~]` Monitor for errors and memory issues — errors yes, **memory no instrumentation**
  - `[x]` Run to completion — ~48 s
- `[~]` Collect all raw metrics during training
  - `[ ]` **Save MSE values at each batch** — computed, not saved (BLOCKER-3)
  - `[x]` Final R² on train and test — 0.3117 / 0.3235
  - `[ ]` **Save memory usage log** — no instrumentation exists (BLOCKER-2)
  - `[x]` Record total training time — ~48 s
- `[~]` Create results log
  - `[~]` Organize metrics into one file — `results/metrics.csv` exists, but is opened in **truncate mode**, so run 002 silently erases run 001 (BLOCKER-4)
  - `[x]` Document final model performance — in README

### Rebecca — Validation & Troubleshooting

- `[x]` Monitor training in parallel
- `[~]` Validate results make sense
  - `[x]` R² within (−∞, 1] — yes
  - `[ ]` **Verify MSE decreases over time** — cannot be verified, per-batch history is not persisted
  - `[ ]` **Spot-check learning curve** — same reason; no curve can be plotted
- `[?]` Handle runtime issues — none reported
- `[x]` Save final trained model — `results/trained_model.h5`
  - `[ ]` ...but the **scaler is not saved**, so the `.h5` file cannot score a new row (BLOCKER-5)

### Ximin — Organization & Progress Tracking

- `[x]` Monitor progress without coding — this audit
- `[x]` Keep log of run times and issues — `RUN_LOG.md`
- `[ ]` Organize results / create folders / date-time labels — **Ximin, manual** (conventions in `RUN_LOG.md` §4)
- `[x]` Update master checklist — this document
- `[x]` Note blockers for team — below
- `[x]` Prepare presentation template with group ID — `.pptx`, 15 slides
- `[x]` Slide placeholders for each required element — done

### End of Thursday

- `[x]` Full training completed successfully
- `[ ]` **All metrics collected and saved** — per-batch MSE and memory log are missing
- `[x]` No runtime errors or data issues
- `[x]` Presentation skeleton created

---

## BLOCKERS — ranked by what stops the deliverable

### BLOCKER-1 · Data is not loaded incrementally · Owner: Ansley · HIGH
`data_handler.py:17` reads the full CSV into memory before batching. Training is incremental; *ingestion is not*. If the assignment requires memory-constrained ingestion, this is a scope miss, and it is also the most likely question from the grader.
**Fix:** `pd.read_csv(DATA_FILE, chunksize=...)`, or agree on a defensible answer.
**Team decision needed before the presentation.**

### BLOCKER-2 · No memory tracking anywhere · Owner: Ansley · HIGH
Confirmed by grep: `psutil` appears in **zero** of the 7 source files. This was a Wednesday deliverable and a Thursday deliverable. Slide 11 cannot be filled and the central claim of incremental learning is unevidenced.
**Fix:** ~10 lines — `psutil.Process().memory_info().rss` sampled every N batches, appended to a list, written to CSV.

### BLOCKER-3 · Per-batch MSE is discarded · Owner: Rebecca/Ansley · HIGH
`mse_history` is populated (`main.py:45`) and then dropped when the function returns. No learning curve can be plotted — slide 9 is blocked, and "verify MSE decreases" cannot be validated.
**Fix:** ~3 lines — `pd.DataFrame({'batch':..., 'mse':mse_history}).to_csv(...)`.

### BLOCKER-4 · `metrics.csv` overwritten every run · Owner: Ansley · HIGH
`results_df.to_csv(METRICS_SAVE_PATH)` truncates. The README states results are "timestamped for tracking progress across multiple runs" — the timestamp is recorded, but the **previous row is destroyed**. Run 002 will erase run 001.
**Fix:** append mode — `mode='a', header=not os.path.exists(path)`.
**Do this before anyone runs training again.**

### BLOCKER-5 · Scaler never persisted · Owner: Rebecca · MEDIUM
`trained_model.h5` expects scaled inputs, but the fitted `StandardScaler` is thrown away. The saved model cannot make a prediction on new data.
**Fix:** `joblib.dump(scaler, 'results/scaler.pkl')`.

### BLOCKER-6 · Ximin's Wednesday deliverables are not in the repo · Owner: Ximin · MEDIUM
No `ximin/` folder, no sample CSV in `data/`. If the grader checks the repo, that work is invisible.
**Fix:** commit `ximin/` + `data/pricing_sample_1000_TEST.csv`.

### BLOCKER-7 · No global seed · Owner: Rebecca · MEDIUM
`RANDOM_SEED = 42` reaches only `train_test_split`. Keras weight init is unseeded, so run 002 will not reproduce 0.3235. **Every future A/B comparison is uninterpretable until this is fixed** — you won't know if a change helped or the dice landed differently.
**Fix:** `tf.keras.utils.set_random_seed(RANDOM_SEED)` at the top of `main.py`.

### BLOCKER-8 · README overstates the result · Owner: Rebecca · MEDIUM
README: *"Test R² > Training R²: Excellent generalization (no overfitting)"* and *"Simpler networks often better."* Measured on the identical split, plain linear regression scores **0.3744** — higher than the network's 0.3235. A tiny train/test gap *below the linear baseline* is the signature of **underfitting**. If this framing reaches the presentation it is an easy target.
**Fix:** reword to "the model currently underperforms a linear baseline; we treat this as an underfitting diagnosis."

### BLOCKER-9 · `.h5` save format · Owner: Rebecca · LOW
`model.save()` to `.h5` is legacy in Keras 3 and errors on some versions. Note the working TF version in `RUN_LOG.md`; consider `.keras`.

---

## Reference numbers for sanity checks

| Quantity | Value |
|---|---|
| Rows × columns | 500,000 × 6 |
| Train / test | 400,000 / 100,000 |
| Batches per epoch @ 32 | 12,500 |
| Trainable parameters | 3,009 |
| Target: mean / median / max / sd | 22.8 / 7 / 4,165 / 50.5 |
| MSE of mean-predictor | ≈ 2,478.8 |
| R² of mean-predictor | 0.0000 |
| **Linear regression R² (bar to beat)** | **0.3744** |
| Gradient boosting R² (headroom) | 0.5885 |
| Run 001 test R² | 0.3235 |

---

## Suggested order of operations

1. BLOCKER-4 (append mode) — **before any further runs**, or run 001 is lost
2. BLOCKER-7 (seed) — before any comparison run
3. BLOCKER-3 (save MSE history) + BLOCKER-2 (memory log) — unblocks slides 9 and 11
4. Re-run training → produces every artefact the deck needs
5. BLOCKER-1 team decision
6. BLOCKER-8 README rewording
7. BLOCKER-6 commit Ximin's folder

Items 1–3 total roughly 20 minutes of editing and are prerequisites for the presentation being fillable.
