# Group Assignment 1 — Incremental Learning on Large Data

A feed-forward neural network that predicts `quantity` sold, trained **one
mini-batch at a time straight off disk**, so the data is never held in RAM.

```bash
uv run python code/main.py
```

One command runs everything and writes every deliverable into `results/`.
Takes about 40 seconds.

---

## The code, in the order the course covered it

| File | What it does | Where we learned it |
|---|---|---|
| `config.py` | every setting in one place | — |
| `data.py` | reads the CSV in chunks, scales it, shuffles it through a bounded buffer | assignment 0 — learning from a stream |
| `model.py` | 5 inputs → three sigmoid hidden layers → linear output | assignment 2 (by hand), assignment 4 (in Keras) |
| `metrics.py` | R², MSE, moving average, permutation importance, partial dependence | assignment 4 (losses), assignment 1 (interpretation) |
| `plots.py` | the four figures | — |
| `main.py` | runs steps 1–8 in order | — |

`metrics.py` knows nothing about TensorFlow — every function that needs
predictions takes a `predict` function as an argument, so the same code would
work on any model.

---

## How the data never lands in RAM

`pricing.csv` is 18 MB, which fits easily — but the assignment is about the
*technique*, so nothing here assumes it fits. Three passes over the file, each
holding only a bounded amount:

**Pass 1 — `compute_statistics()`**
Standardising needs the mean and standard deviation of the training set, which
you cannot know without seeing every row. So we accumulate three running
totals — count, sum, sum of squares — and derive the statistics at the end.
Only a handful of numbers are kept.

**Pass 2 — `stream_train_batches()`**
Reads `CHUNK_SIZE` rows, scales them, and yields mini-batches to the model.

**Pass 3 — `load_sample()`**
Keeps a capped random sample in memory for scoring, importance and partial
dependence. Capped, so this cost is fixed rather than growing with the file.

### The train/test split without shuffling the file

`_is_test()` hashes the row number instead of drawing at random. The same row
therefore lands in the same set on every pass and every run — which lets us
split the data without ever holding it or reordering it on disk.

### The shuffle buffer, and why it matters

`pricing.csv` is **sorted by `sku`** (correlation 0.9995 with row number). Fed
to the model in file order, every mini-batch would contain nearly identical
rows — and stochastic gradient descent assumes each batch is a fair sample.

We cannot shuffle a file we cannot hold, so `data.py` fills a fixed-size
buffer, shuffles that, and empties it before filling the next one.

| shuffle buffer | test R² |
|---|---|
| none (file order) | 0.4445 |
| 50,000 | 0.4478 |
| 100,000 | 0.5000 |
| **200,000 (chosen)** | **0.5312** |
| 400,000 | 0.5406 — a full shuffle of this file, for reference |

200,000 rows costs about 5 MB and recovers essentially all the accuracy. It is
a *fixed* number of rows, so it would stay 5 MB on a file a hundred times
larger.

---

## Two other decisions worth knowing

**The target is standardised.** Raw `quantity` has mean 23 and reaches 4165.
Asking sigmoid layers to produce numbers that large saturates them and stalls
learning. `unscale()` converts predictions back before anything is scored.

**The output layer is linear, not sigmoid.** Sigmoid can only produce values
between 0 and 1; `quantity` runs to 4165.

---

## Deliverables

| Required by the PDF | Produced by |
|---|---|
| R² on train and test | `results/metrics.csv` |
| Learning curve (records seen vs moving-average MSE) | `results/plots/learning_curve.png` |
| Variable importance | `results/plots/variable_importance.png` |
| Multiple partial dependence plots | `results/plots/partial_dependence.png` |
| RAM usage | `results/plots/memory_usage.png`, `results/memory_log.csv` |
| Training time | `results/metrics.csv` |
| Who did what | the presentation |

---

## Current results

```
R²   train 0.5260    test 0.5312
MSE  train 1,107     test 1,123

training time         32.5 s  (about 12,300 records/second)
RAM growth while training   +49 MB over 400,000 records
peak RAM                    407 MB  on a 14,171 MB machine
```

Permutation importance, by how far test R² falls when the feature is shuffled:

```
duration   0.9007      <- dominant
price      0.3418
category   0.0224
order      0.0096
sku        0.0013      <- essentially unused
```

`MSE_smoothed_start` and `MSE_smoothed_end` in `metrics.csv` are moving
averages, not single batches. A single batch is 32 records and far too noisy
to quote — the first and last individual batches can easily suggest the model
got worse when it did not.

---

## Questions to be ready for

The assignment says the test may ask you to reason through this code, so have
an answer for:

- **`sku` and `category` are treated as continuous numbers.** The PDF says
  they are integer-encoded *categoricals*. Standardising 75,000 SKU codes
  tells the network that SKU 5000 is "bigger" than SKU 2500, which is
  meaningless. Embeddings are the proper answer. Importance shows the model
  barely uses `sku` anyway.
- **Adam, not plain SGD.** Know what Adam changes about
  `w ← w − η · gradient`.
- **Layer sizes 64 / 32 / 16.** A tapering funnel is the conventional choice;
  we did not tune them.
- **One pass over the data.** True incremental learning sees each record once.
  More passes would raise R² but stop being a demonstration of streaming.

---

## Note

`rebecca/model-documentation.md` and `rebecca/network-architecture.md` still
describe the previous file layout (`data_handler.py`, `evaluate.py`,
`utils.py`, `test_integration.py`). Those files no longer exist — worth
coordinating before the presentation.
