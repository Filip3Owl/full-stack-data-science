# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3.12 (Data Science Course)
#     language: python
#     name: dsc-py312
# ---

# %% [markdown]
# # 08 · Performance & Memory
#
# **Module 02 — NumPy & Numerical Computing**
#
# ---
#
# ## What you will learn
#
# 1. **Strides** — how one flat buffer pretends to be an N-dimensional array.
# 2. C-order vs Fortran-order, and why iterating the wrong way is slower.
# 3. In-place operations and how to stop allocating temporaries.
# 4. Reading memory usage honestly, including hidden intermediates.
# 5. `np.memmap` for arrays bigger than RAM.
# 6. Knowing when NumPy has run out of road — and what comes next.
#
# **Estimated time:** 75 minutes

# %%
import time

import matplotlib.pyplot as plt
import numpy as np

from dstoolkit import format_bytes, setup_plot_style
from dstoolkit.plotting import COLORS

setup_plot_style()
rng = np.random.default_rng(seed=42)


def best_of(fn, *args, repeats: int = 5) -> float:
    """Return the fastest of `repeats` runs, in seconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - start)
    return best

# %% [markdown]
# ## 1. Strides — the trick behind everything
#
# An ndarray is a **flat buffer** plus a rule for interpreting it. That rule is
# the `strides` tuple: how many **bytes** to step to advance one position along
# each axis.

# %%
arr = np.arange(12, dtype=np.int64).reshape(3, 4)

print(f"arr:\n{arr}")
print(f"\nshape   : {arr.shape}")
print(f"strides : {arr.strides} bytes")
print(f"itemsize: {arr.itemsize} bytes")
print(f"\n→ advance one ROW    : skip {arr.strides[0]} bytes = 4 elements")
print(f"→ advance one COLUMN : skip {arr.strides[1]} bytes = 1 element")

# %% [markdown]
# This is why reshaping and transposing are free — they only rewrite the strides,
# never the data.

# %%
print(f"arr.strides       : {arr.strides}")
print(f"arr.T.strides     : {arr.T.strides}   ← swapped, no data moved")
print(f"same buffer       : {np.shares_memory(arr, arr.T)}")

print(f"\narr[::2].strides  : {arr[::2].strides}   ← doubled row stride, skips rows")
print(f"arr[:, ::2].strides : {arr[:, ::2].strides}   ← doubled column stride")

# %% [markdown]
# ### C-order versus Fortran-order

# %%
c_order = np.arange(12, dtype=np.int64).reshape(3, 4)          # row-major (default)
f_order = np.asfortranarray(c_order)                            # column-major

print(f"C order  : strides {c_order.strides}   "
      f"C_CONTIGUOUS={c_order.flags['C_CONTIGUOUS']}")
print(f"F order  : strides {f_order.strides}   "
      f"F_CONTIGUOUS={f_order.flags['F_CONTIGUOUS']}")
print(f"\nsame values : {np.array_equal(c_order, f_order)}")
print(f"different memory order:")
print(f"  C, flat : {c_order.ravel(order='K')}")
print(f"  F, flat : {f_order.ravel(order='K')}")

# %% [markdown]
# ### Why the order matters: cache locality
#
# CPUs fetch memory in **cache lines** of 64 bytes. When you read data that sits
# contiguously, every fetch delivers several useful values. When you stride
# across memory, most of each fetch is wasted.

# %%
size = 4000
c_matrix = rng.random((size, size))
f_matrix = np.asfortranarray(c_matrix)

print(f"matrix: {size}×{size}  ({format_bytes(c_matrix.nbytes)})\n")

# Summing along axis 0 walks DOWN columns; axis 1 walks ACROSS rows.
t_c_rows = best_of(lambda: c_matrix.sum(axis=1))
t_c_cols = best_of(lambda: c_matrix.sum(axis=0))
t_f_rows = best_of(lambda: f_matrix.sum(axis=1))
t_f_cols = best_of(lambda: f_matrix.sum(axis=0))

print(f"{'':<12} {'sum(axis=1)':>14} {'sum(axis=0)':>14}")
print(f"{'':<12} {'(across rows)':>14} {'(down columns)':>14}")
print("-" * 42)
print(f"{'C-order':<12} {t_c_rows * 1000:>12.1f}ms {t_c_cols * 1000:>12.1f}ms")
print(f"{'F-order':<12} {t_f_rows * 1000:>12.1f}ms {t_f_cols * 1000:>12.1f}ms")

print(f"\n→ each layout is faster along the axis it stores contiguously")

# %% [markdown]
# > [!TIP]
# > You rarely choose the order explicitly, but you inherit it. `arr.T`,
# > `df.values` from a column-oriented source, and some SciPy routines all return
# > non-C-contiguous arrays. If a downstream operation is unexpectedly slow,
# > check `.flags` and consider `np.ascontiguousarray`.

# %%
# When does the copy pay for itself?
transposed = c_matrix.T                       # a view, F-contiguous
contiguous = np.ascontiguousarray(transposed)  # pays a copy up front

t_view = best_of(lambda: transposed.sum(axis=1))
t_copy_then_sum = best_of(lambda: np.ascontiguousarray(transposed).sum(axis=1))
t_precopied = best_of(lambda: contiguous.sum(axis=1))

print(f"sum on the transposed view      : {t_view * 1000:7.1f} ms")
print(f"copy + sum, every time          : {t_copy_then_sum * 1000:7.1f} ms")
print(f"sum on a pre-copied array       : {t_precopied * 1000:7.1f} ms")
print(f"\n→ copy once and reuse if you will touch it more than "
      f"{t_copy_then_sum / max(t_view - t_precopied, 1e-9):.0f} times")

# %% [markdown]
# ## 2. Temporaries — the hidden allocations
#
# Every intermediate in an expression allocates a full array. A chain of
# operations on a large array can use several times the memory you expect.

# %%
n = 20_000_000
x = rng.random(n)

print(f"x is {format_bytes(x.nbytes)}\n")
print("The expression  y = (x * 2 + 1) ** 2  allocates:")
print(f"  x * 2          → {format_bytes(x.nbytes)}  (temporary 1)")
print(f"  ... + 1        → {format_bytes(x.nbytes)}  (temporary 2)")
print(f"  ... ** 2       → {format_bytes(x.nbytes)}  (the result)")
print(f"  peak           ≈ {format_bytes(x.nbytes * 3)}")

# %%
def with_temporaries(arr):
    return (arr * 2 + 1) ** 2


def in_place(arr, out):
    """Same computation, reusing one buffer. No temporaries at all."""
    np.multiply(arr, 2, out=out)
    np.add(out, 1, out=out)
    np.power(out, 2, out=out)
    return out


buffer = np.empty_like(x)

t_temp = best_of(with_temporaries, x, repeats=3)
t_inplace = best_of(in_place, x, buffer, repeats=3)

print(f"with temporaries : {t_temp * 1000:7.1f} ms")
print(f"in place         : {t_inplace * 1000:7.1f} ms   "
      f"({t_temp / t_inplace:.2f}× faster)")
print(f"same result      : {np.allclose(with_temporaries(x), in_place(x, buffer))}")

# %% [markdown]
# The speed gain is modest; the **memory** gain is the real prize. In-place work
# is what lets a job fit in RAM at all.

# %%
# The in-place operators do the same thing more readably, when you own the array.
y = x.copy()
y *= 2       # no temporary
y += 1
y **= 2

print(f"matches : {np.allclose(y, with_temporaries(x))}")

# %% [markdown]
# > [!WARNING]
# > `y *= 2` mutates in place — so if `y` is a **view** of something else, you
# > have just modified the original. This is the notebook-02 lesson meeting the
# > performance advice: in-place is efficient *and* dangerous, and you must know
# > which arrays you own.

# %%
base = np.arange(10.0)
view = base[2:5]
view *= 100          # in-place through a view

print(f"base after mutating the view : {base}")
print("↑ efficient, and almost certainly not what you wanted.")

# %% [markdown]
# ## 3. Measuring memory honestly

# %%
def memory_profile(**arrays) -> None:
    """Print the size of each named array, marking views as borrowing memory."""
    total_owned = 0
    print(f"{'name':<16} {'shape':>16} {'dtype':>10} {'size':>12} {'owns?':>7}")
    print("-" * 66)
    for name, arr in arrays.items():
        owns = arr.base is None
        total_owned += arr.nbytes if owns else 0
        print(f"{name:<16} {str(arr.shape):>16} {str(arr.dtype):>10} "
              f"{format_bytes(arr.nbytes):>12} {'yes' if owns else 'view':>7}")
    print(f"\ntotal actually allocated : {format_bytes(total_owned)}")


data = rng.random((2000, 500))
memory_profile(
    data=data,
    a_view=data[:100],
    a_copy=data[:100].copy(),
    transposed=data.T,
    downcast=data.astype(np.float32),
)

# %% [markdown]
# ## 4. Arrays larger than RAM: `np.memmap`
#
# A memory-mapped array lives on disk and is paged in on demand. It behaves like
# an ndarray but is bounded by disk, not RAM.

# %%
import tempfile
from pathlib import Path

tmpdir = Path(tempfile.mkdtemp())
memmap_path = tmpdir / "large_array.dat"

shape = (10_000, 500)          # 40 MB — small enough for a notebook, real in principle

# Create and write in chunks, never holding it all in memory.
mapped = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=shape)
for start in range(0, shape[0], 1000):
    mapped[start:start + 1000] = rng.random((1000, shape[1]))
mapped.flush()
del mapped

# Reopen read-only and compute over it.
mapped = np.memmap(memmap_path, dtype=np.float64, mode="r", shape=shape)

print(f"on-disk size    : {format_bytes(memmap_path.stat().st_size)}")
print(f"type            : {type(mapped).__name__}")
print(f"column means    : {mapped.mean(axis=0)[:5].round(4)}")
print(f"overall mean    : {mapped.mean():.6f}")
print("\n↑ computed without ever loading the whole array into RAM.")

# %%
# Chunked reduction is the pattern that generalises to genuinely huge files.
total, count = 0.0, 0
for start in range(0, shape[0], 2000):
    block = np.asarray(mapped[start:start + 2000])
    total += block.sum()
    count += block.size

print(f"chunked mean : {total / count:.6f}")
print(f"matches      : {np.isclose(total / count, mapped.mean())}")

del mapped

# %% [markdown]
# ## 5. Choosing the right tool
#
# NumPy is fast when the work is expressible as array operations. When it is not,
# the answer is a different tool, not a cleverer array.

# %%
# A genuinely sequential algorithm: each step depends on the previous one.
def exponential_smoothing_loop(values: np.ndarray, alpha: float) -> np.ndarray:
    """s[t] = alpha*x[t] + (1-alpha)*s[t-1]. Cannot be vectorised directly."""
    out = np.empty_like(values)
    out[0] = values[0]
    for t in range(1, len(values)):
        out[t] = alpha * values[t] + (1 - alpha) * out[t - 1]
    return out


series = rng.random(200_000)
t_loop = best_of(exponential_smoothing_loop, series, 0.3, repeats=3)
print(f"Python loop : {t_loop * 1000:7.1f} ms")

# %%
# SciPy exposes the same recurrence as a digital filter, implemented in C.
from scipy.signal import lfilter


def exponential_smoothing_scipy(values: np.ndarray, alpha: float) -> np.ndarray:
    """Same recurrence, expressed as an IIR filter.

    `zi` sets the filter's initial delay state. Left at its default of zero,
    lfilter produces y[0] = alpha*x[0], whereas our loop starts at y[0] = x[0].
    That seeds a transient which decays but never fully disappears, so we set
    zi = (1-alpha)*x[0] to make the two definitions identical from step 0.
    """
    zi = [(1 - alpha) * values[0]]
    smoothed, _ = lfilter([alpha], [1, -(1 - alpha)], values, zi=zi)
    return smoothed


t_scipy = best_of(exponential_smoothing_scipy, series, 0.3, repeats=3)

loop_result = exponential_smoothing_loop(series, 0.3)
scipy_result = exponential_smoothing_scipy(series, 0.3)

print(f"scipy lfilter : {t_scipy * 1000:7.1f} ms   ({t_loop / t_scipy:.0f}× faster)")
print(f"identical results : {np.allclose(loop_result, scipy_result)}")

# Without the initial condition the results drift — worth seeing once.
naive_scipy = lfilter([0.3], [1, -0.7], series)
print(f"\nwith zi=0 (the default), max deviation : "
      f"{np.abs(loop_result - naive_scipy).max():.4f}   ← a lasting transient")

# %% [markdown]
# ### The escalation ladder
#
# | Symptom | Next step |
# | :--- | :--- |
# | A Python `for` loop over elements | Rewrite as array operations |
# | Loop is genuinely sequential | Look for a SciPy routine (`lfilter`, `cumsum`, `convolve`) |
# | Still sequential, no library exists | **Numba** `@njit` — usually a one-line change |
# | Peak memory is the constraint | In-place ops, chunking, `np.memmap` |
# | Data exceeds one machine | **Dask** or **Spark** (Module 20) |
# | Heterogeneous / labelled columns | **pandas** (Module 03) |
# | Very large matrices, mostly zeros | `scipy.sparse` |

# %%
# Sparse matrices: when most entries are zero, storing them is pure waste.
from scipy import sparse

dense = np.zeros((5_000, 5_000))
# Fill 0.1% of the entries.
n_nonzero = int(dense.size * 0.001)
rows = rng.integers(0, 5_000, n_nonzero)
cols = rng.integers(0, 5_000, n_nonzero)
dense[rows, cols] = rng.random(n_nonzero)

sparse_matrix = sparse.csr_matrix(dense)

print(f"dense  : {format_bytes(dense.nbytes)}")
print(f"sparse : {format_bytes(sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes)}")
print(f"density: {sparse_matrix.nnz / dense.size:.3%}")
print(f"saving : {1 - (sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes) / dense.nbytes:.1%}")

# %% [markdown]
# > [!NOTE]
# > One-hot encoded categorical features and text TF-IDF matrices are almost
# > always over 99% zeros. `scipy.sparse` is why a bag-of-words model over a
# > 50 000-word vocabulary fits in memory at all (Module 18).

# %% [markdown]
# ## 6. A performance checklist
#
# Work down this list before reaching for a bigger machine.

# %%
checklist = [
    ("Is there a Python loop over elements?", "Vectorise it"),
    ("Are you calling sum/max/min (built-ins) on an array?", "Use np.sum / arr.max"),
    ("Is the dtype wider than needed?", "float32 halves memory and often time"),
    ("Are you allocating temporaries in a hot loop?", "Use out= or in-place operators"),
    ("Is the array C-contiguous?", "Check .flags; np.ascontiguousarray if reused"),
    ("Are you reducing over the fastest-varying axis?", "Prefer axis=-1 on C-order"),
    ("Are you sorting to get the top k?", "Use np.argpartition"),
    ("Are you building a huge broadcast intermediate?", "Use einsum or chunk it"),
    ("Is the matrix mostly zeros?", "scipy.sparse"),
    ("Does it still not fit?", "memmap, then Dask/Spark"),
]

print(f"{'check':<50} {'action'}")
print("-" * 96)
for check, action in checklist:
    print(f"{check:<50} {action}")

# %%
# Clean up the memmap file.
import shutil

shutil.rmtree(tmpdir, ignore_errors=True)
print("temporary files removed")

# %% [markdown]
# ## 7. Exercises
#
# ### Exercise 8.1 — Predict the strides
#
# For `a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)`, give `strides` for:
# `a`, `a.T`, `a[::2]`, `a[:, ::2, :]`, `a.reshape(6, 4)`. Verify each.
#
# ### Exercise 8.2 — Row-major versus column-major
#
# Build a 3000×3000 array in both orders. Time `sum(axis=0)` and `sum(axis=1)` on
# each. Explain the pattern in terms of cache lines, and predict which
# combination is fastest before measuring.
#
# ### Exercise 8.3 — Eliminate the temporaries
#
# Rewrite `result = np.sqrt(a**2 + b**2) / (c + 1)` using only `out=` parameters
# and at most two buffers. Verify the results match and measure peak allocations.
#
# ### Exercise 8.4 — Chunked statistics
#
# Write `chunked_stats(path, shape, dtype, chunk_rows)` that computes the mean and
# standard deviation of a memmapped array **in one pass**, without loading it all.
# *Hint:* accumulate the count, the sum, and the sum of squares — then be careful
# about the catastrophic cancellation you met in notebook 03.
#
# ### Exercise 8.5 — Know when to stop
#
# Implement a cumulative maximum (`out[i] = max(x[:i+1])`) three ways: a Python
# loop, `np.maximum.accumulate`, and a chunked version. Time all three on 10
# million elements and explain the ranking.

# %% [markdown]
# ## Key takeaways
#
# - An ndarray is a **flat buffer plus strides**. Reshape and transpose only
#   rewrite the strides, which is why they are free.
# - **Contiguity governs cache behaviour.** Reduce along the axis your array
#   stores contiguously; check `.flags` when something is unexpectedly slow.
# - Every intermediate in an expression **allocates a full array**. Use `out=` or
#   in-place operators when memory is the constraint.
# - **In-place operations on a view mutate the original** — efficiency and the
#   aliasing trap are the same mechanism.
# - `.base is None` tells you whether an array actually owns its memory.
# - **`np.memmap`** handles arrays bigger than RAM; chunked reductions generalise
#   the pattern.
# - When a loop is genuinely sequential, look for a **SciPy routine** before
#   reaching for Numba.
# - **`scipy.sparse`** for matrices that are mostly zeros — one-hot and TF-IDF
#   always are.
#
# ---
#
# **Next:** [09 · Module Project](09_module_project.ipynb) — implement linear
# regression and k-means from scratch, using only what this module has taught.
