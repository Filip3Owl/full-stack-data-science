# Module 02 — Exercises

Notebooks 01–08 each end with five exercises. This page collects all 40 in one
place so you can work through them without scrolling, and points at the worked
solutions.

> [!WARNING]
> **Attempt each exercise before opening the solution.** Reading a worked answer
> produces a convincing feeling of understanding that vanishes the moment you
> face a blank cell. Ten minutes of being stuck teaches more than an hour of
> reading.

---

## Where the solutions are

All solutions live in a single executed notebook:

📓 **[`../notebooks/10_exercise_solutions.ipynb`](../notebooks/10_exercise_solutions.ipynb)**

Each solution includes the reasoning, and — where more than one answer is
defensible — the alternatives with their trade-offs. Several go beyond the
question asked, because the interesting part is usually *why* the obvious
approach is slow or wrong.

---

## The exercise list

### Notebook 01 — Why NumPy Exists

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 1.1 | Measure it yourself | Honest timing methodology |
| 1.2 | The honest memory question | `nbytes` vs `sys.getsizeof` |
| 1.3 | Find the break-even point | Fixed per-call cost of a ufunc |
| 1.4 | Spot the de-vectorisation | Recognising a hidden Python loop |
| 1.5 | `out=` in a loop | Reusing buffers, allocation cost |

### Notebook 02 — ndarray Fundamentals

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 2.1 | Predict the shapes | Shape arithmetic before running code |
| 2.2 | View or copy? | `.base`, slicing vs fancy indexing |
| 2.3 | Fix the mutation bug | Defensive copying at API boundaries |
| 2.4 | Build a checkerboard | Strided slice assignment |
| 2.5 | Reshape for scikit-learn | `reshape(-1, 1)`, `ravel`, the 2-D contract |

### Notebook 03 — Dtypes & Numerical Precision

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 3.1 | Predict the dtype | NumPy 2.0 promotion rules |
| 3.2 | Trigger and then prevent an overflow | Integer wraparound, deliberate upcasting |
| 3.3 | When is `float32` not enough? | Precision budgeting |
| 3.4 | Build a memory report | Safe downcasting with `iinfo`/`finfo` |
| 3.5 | Catastrophic cancellation in the wild | `np.spacing`, numerically stable formulas |

### Notebook 04 — Broadcasting

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 4.1 | Compatible or not? | Applying the two rules by hand |
| 4.2 | Fix the silent bug | `(n,)` vs `(n, 1)` |
| 4.3 | Normalise rows instead of columns | `axis=` + `keepdims=True` together |
| 4.4 | Distance to centroids | `np.newaxis`, pairwise distances |
| 4.5 | Avoid the intermediate | Algebraic expansion instead of an `(n, m)` tensor |

### Notebook 05 — Aggregations & Boolean Masking

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 5.1 | Axis drill | Which dimension a reduction collapses |
| 5.2 | The keepdims trap | Broadcasting a result back onto its source |
| 5.3 | Masking practice | Combining conditions with `&`, `|`, `~` |
| 5.4 | Group-by from scratch | `unique(return_inverse=True)` + `bincount` |
| 5.5 | Top-k per row | `argpartition` vs `argsort` |

### Notebook 06 — Linear Algebra

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 6.1 | Shapes | `@` vs `*`, conformability |
| 6.2 | Cosine similarity matrix | Norms, row normalisation, one matmul |
| 6.3 | Implement least squares | Normal equations vs `lstsq` |
| 6.4 | Condition number and collinearity | `np.linalg.cond`, numerical hopelessness |
| 6.5 | Image compression with SVD | Truncated SVD, rank vs storage |

### Notebook 07 — Randomness & Simulation

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 7.1 | Break and fix reproducibility | `Generator` vs global seeding |
| 7.2 | Choose the distribution | Matching a generator to a process |
| 7.3 | The birthday problem | Monte Carlo against a closed form |
| 7.4 | Bootstrap a difference | Resampled confidence intervals |
| 7.5 | Sample size by simulation | Statistical power without a formula |

### Notebook 08 — Performance & Memory

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 8.1 | Predict the strides | Strides, `itemsize`, why `.T` is free |
| 8.2 | Row-major versus column-major | Memory order and cache locality |
| 8.3 | Eliminate the temporaries | In-place operators, `out=` |
| 8.4 | Chunked statistics | One-pass accumulation in bounded memory |
| 8.5 | Know when to stop | Recognising what NumPy can still do in C |

### Notebook 09 — Module Project

The project ends with six open-ended extensions (⭐ to ⭐⭐⭐). They have no single
right answer — they are the closest thing in this module to real work.

| # | Extension | Difficulty |
| :-- | :--- | :-- |
| 1 | Promote both estimators into `dstoolkit` | ⭐ |
| 2 | Weighted least squares without an `(n, n)` matrix | ⭐⭐ |
| 3 | Mini-batch k-means | ⭐⭐ |
| 4 | Silhouette score, vectorised | ⭐⭐⭐ |
| 5 | Polynomial features and the condition number | ⭐⭐ |
| 6 | Elkan's algorithm | ⭐⭐⭐ |

---

## How to check your own work

Beyond comparing with the solutions, three habits generalise — the third is
specific to NumPy and will catch most of your bugs:

1. **Write a test.** If you cannot state what the correct output is, you do not
   yet understand the problem.
2. **Restart & Run All.** If your solution only works when cells run in the order
   you happened to click them, it does not work.
3. **Print the shape and the dtype, every time.** Most NumPy bugs are a shape
   that broadcast when you did not intend it to, or a dtype that promoted when
   you were not looking. Neither raises an exception.
