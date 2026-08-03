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
# # 02 · ndarray Fundamentals
#
# **Module 02 — NumPy & Numerical Computing**
#
# ---
#
# ## What you will learn
#
# 1. Every way to create an array, and which one to reach for.
# 2. `shape`, `ndim`, `size`, `dtype` — the four attributes you will check constantly.
# 3. Indexing and slicing in N dimensions.
# 4. **Views versus copies** — the rule that *reverses* from Module 01, and the
#    single biggest source of silent bugs in NumPy code.
# 5. Reshaping, transposing, stacking, and splitting.
#
# **Estimated time:** 90 minutes

# %%
import numpy as np

from dstoolkit import setup_plot_style

setup_plot_style()

rng = np.random.default_rng(seed=42)

# %% [markdown]
# ## 1. Creating arrays
#
# ### From existing Python data

# %%
# From a list — the dtype is inferred from the contents.
a = np.array([1, 2, 3, 4])
b = np.array([1.0, 2.0, 3.0])
c = np.array([[1, 2, 3], [4, 5, 6]])       # nested lists → 2-D

print(f"a : {a}   dtype={a.dtype}  shape={a.shape}")
print(f"b : {b}   dtype={b.dtype}  shape={b.shape}")
print(f"c :\n{c}\n    dtype={c.dtype}  shape={c.shape}")

# Being explicit about dtype is a good habit — it documents intent and prevents
# the surprise of getting int64 when you needed floats.
d = np.array([1, 2, 3], dtype=np.float64)
print(f"\nd : {d}   dtype={d.dtype}   ← requested explicitly")

# %% [markdown]
# ### From scratch — the functions you will use daily

# %%
print("zeros(3)          :", np.zeros(3))
print("ones((2, 3))      :\n", np.ones((2, 3)))
print("full((2, 2), 7)   :\n", np.full((2, 2), 7))
print("eye(3)            :\n", np.eye(3))          # identity matrix
print("empty(3)          :", np.empty(3).round(3), " ← UNINITIALISED memory")

# %% [markdown]
# > [!WARNING]
# > `np.empty` does **not** zero the memory — it hands you whatever bytes were
# > already there. It is marginally faster than `zeros`, and it is the right
# > choice only when you are about to overwrite every element. Using it and then
# > forgetting to fill it produces garbage that looks plausible.

# %% [markdown]
# ### Ranges and grids

# %%
# arange: like range(), but returns an array and accepts floats.
print("arange(0, 10, 2)      :", np.arange(0, 10, 2))
print("arange(0, 1, 0.25)    :", np.arange(0, 1, 0.25))

# linspace: N evenly spaced points, INCLUSIVE of the endpoint.
print("linspace(0, 1, 5)     :", np.linspace(0, 1, 5))

# logspace: evenly spaced on a log scale — ideal for hyperparameter grids.
print("logspace(-3, 0, 4)    :", np.logspace(-3, 0, 4))

# %% [markdown]
# > [!TIP]
# > **Prefer `linspace` over `arange` for floats.** `arange` accumulates
# > floating-point error and its endpoint is unpredictable — a direct consequence
# > of the representation problem from Module 01.

# %%
# The trap, concretely:
print(f"arange(0, 1, 0.1) has {len(np.arange(0, 1, 0.1))} elements")
print(f"arange(0, 0.3, 0.1) : {np.arange(0, 0.3, 0.1)}")
print(f"                      ← 0.3 sneaked in; 0.1+0.1+0.1 < 0.3 in binary")
print(f"\nlinspace(0, 0.3, 4) : {np.linspace(0, 0.3, 4)}   ← exact count, exact ends")

# %% [markdown]
# ### Random arrays
#
# Always use the modern `Generator` API (`np.random.default_rng`). The legacy
# `np.random.seed` / `np.random.rand` functions share one global state, which
# makes reproducibility fragile once more than one piece of code draws numbers.

# %%
rng = np.random.default_rng(seed=42)

print("random(3)            :", rng.random(3).round(4))
print("normal(0, 1, 3)      :", rng.normal(0, 1, 3).round(4))
print("integers(0, 10, 5)   :", rng.integers(0, 10, 5))
print("choice(['a','b'], 4) :", rng.choice(["a", "b"], 4))

# Same seed → same numbers, every time, on every machine.
print(f"\nreproducible : "
      f"{np.array_equal(np.random.default_rng(7).random(3), np.random.default_rng(7).random(3))}")

# %% [markdown]
# ## 2. The four attributes you will check constantly
#
# When something goes wrong in NumPy code, the answer is almost always in one of
# these. Print them before you debug anything else.

# %%
arr = rng.normal(size=(3, 4, 2))

print(f"arr.shape    : {arr.shape}      ← size along each dimension")
print(f"arr.ndim     : {arr.ndim}                ← number of dimensions")
print(f"arr.size     : {arr.size}               ← total elements (3×4×2)")
print(f"arr.dtype    : {arr.dtype}         ← type of every element")
print(f"arr.itemsize : {arr.itemsize} bytes")
print(f"arr.nbytes   : {arr.nbytes} bytes         ← size × itemsize")

# %% [markdown]
# ### The distinction that trips everyone: 1-D vs 2-D
#
# `(3,)` and `(3, 1)` and `(1, 3)` hold the same three numbers and behave
# completely differently under broadcasting (notebook 04) and matrix
# multiplication (notebook 06).

# %%
one_d = np.array([1, 2, 3])
column = np.array([[1], [2], [3]])
row = np.array([[1, 2, 3]])

for name, x in [("1-D", one_d), ("column (3,1)", column), ("row (1,3)", row)]:
    print(f"{name:<14} shape={str(x.shape):<8} ndim={x.ndim}")

print(f"\none_d + column →\n{one_d + column}")
print("\n↑ A (3,) plus a (3,1) gives a (3,3) — almost never what you meant.")
print("  Notebook 04 explains exactly why.")

# %% [markdown]
# ## 3. Indexing and slicing

# %%
arr = np.arange(1, 13).reshape(3, 4)
print(f"arr =\n{arr}")

# %%
# Single element: [row, col] — one pair of brackets, not arr[row][col].
print(f"arr[0, 0]   : {arr[0, 0]}")
print(f"arr[2, 3]   : {arr[2, 3]}")
print(f"arr[-1, -1] : {arr[-1, -1]}   ← negative indexing works per axis")

# arr[0][0] also works but is slower: it builds an intermediate row array first.
print(f"\narr[1][2] and arr[1, 2] agree : {arr[1][2] == arr[1, 2]}")

# %%
# Slicing: [start:stop:step] per axis, separated by commas.
print(f"arr[0]      : {arr[0]}          ← first row")
print(f"arr[:, 0]   : {arr[:, 0]}          ← first column")
print(f"arr[1:, 2:] :\n{arr[1:, 2:]}")
print(f"arr[::2]    :\n{arr[::2]}   ← every other row")
print(f"arr[:, ::-1]:\n{arr[:, ::-1]}   ← columns reversed")

# %% [markdown]
# ### `...` (Ellipsis) — "all the remaining axes"
#
# Useful when you have many dimensions and only care about the first or last.

# %%
cube = np.arange(24).reshape(2, 3, 4)

print(f"cube.shape       : {cube.shape}")
print(f"cube[0, ...]     : shape {cube[0, ...].shape}   ← same as cube[0, :, :]")
print(f"cube[..., 0]     : shape {cube[..., 0].shape}   ← same as cube[:, :, 0]")
print(f"cube[..., 0] ==\n{cube[..., 0]}")

# %% [markdown]
# ## 4. Views versus copies — read this twice
#
# ![Views versus copies](../../../assets/images/views_vs_copies.png)
#
# In Module 01 you learned that slicing a list **copies**. In NumPy, **basic
# slicing returns a view** — a second window onto the *same* memory. Writing
# through the view changes the original.
#
# This is a deliberate design decision: it makes slicing free regardless of size.
# It is also the source of a whole category of silent bugs.

# %%
lst = [1, 2, 3, 4]
part = lst[1:3]
part[0] = 99
print(f"list    : lst  = {lst}   ← unchanged (a copy was made)")

arr = np.array([1, 2, 3, 4])
part = arr[1:3]
part[0] = 99
print(f"ndarray : arr  = {arr}   ← CHANGED (part was a view)")

# %% [markdown]
# ### How to tell
#
# `.base` points at the array a view borrows from, and is `None` for an array
# that owns its data.

# %%
original = np.arange(10)
view = original[2:5]
copy = original[2:5].copy()

print(f"view.base is original : {view.base is original}")
print(f"copy.base             : {copy.base}")
print(f"shares memory (view)  : {np.shares_memory(original, view)}")
print(f"shares memory (copy)  : {np.shares_memory(original, copy)}")

# %% [markdown]
# ### What produces a view, and what produces a copy
#
# | Operation | Result | |
# | :--- | :--- | :-- |
# | `arr[1:5]` — basic slicing | **view** | shares memory |
# | `arr[:, 0]` | **view** | shares memory |
# | `arr.reshape(...)` | **view** *(when possible)* | shares memory |
# | `arr.T` / `arr.transpose()` | **view** | shares memory |
# | `arr.ravel()` | **view** *(when possible)* | shares memory |
# | `arr[[0, 2, 4]]` — fancy indexing | **copy** | independent |
# | `arr[arr > 3]` — boolean mask | **copy** | independent |
# | `arr.flatten()` | **copy** | always |
# | `arr.copy()` | **copy** | always |
# | `arr + 1`, `np.sqrt(arr)` | **new array** | independent |

# %%
source = np.arange(12).reshape(3, 4)

operations = {
    "source[1:3]": source[1:3],
    "source[:, 1]": source[:, 1],
    "source.T": source.T,
    "source.reshape(4, 3)": source.reshape(4, 3),
    "source.ravel()": source.ravel(),
    "source[[0, 2]]": source[[0, 2]],
    "source[source > 5]": source[source > 5],
    "source.flatten()": source.flatten(),
    "source + 0": source + 0,
}

print(f"{'operation':<24} {'view?':>6}")
print("-" * 32)
for label, result in operations.items():
    is_view = np.shares_memory(source, result)
    print(f"{label:<24} {'VIEW' if is_view else 'copy':>6}")

# %% [markdown]
# ### The bug this causes in practice

# %%
def normalise_first_column_broken(data: np.ndarray) -> np.ndarray:
    """BUG: modifies the caller's array through a view."""
    column = data[:, 0]        # a view
    column -= column.mean()    # in-place → writes into `data`
    return data


def normalise_first_column(data: np.ndarray) -> np.ndarray:
    """Correct: work on a copy, leave the input alone."""
    out = data.copy()
    out[:, 0] = out[:, 0] - out[:, 0].mean()
    return out


measurements = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])
snapshot = measurements.copy()

normalise_first_column_broken(measurements)
print(f"after the broken version, the INPUT is:\n{measurements}")
print(f"unchanged? {np.array_equal(measurements, snapshot)}")

measurements = snapshot.copy()
result = normalise_first_column(measurements)
print(f"\nafter the correct version, the input is:\n{measurements}")
print(f"unchanged? {np.array_equal(measurements, snapshot)}")

# %% [markdown]
# > [!WARNING]
# > Note *why* the broken version bites: `column -= ...` is an **in-place**
# > operation on a view. Had it been `column = column - column.mean()`, it would
# > merely have rebound a local name and done nothing at all — the same `+=` vs
# > `= ... +` distinction from Module 01, notebook 02, with higher stakes.

# %% [markdown]
# ## 5. Reshaping

# %%
flat = np.arange(12)

print(f"flat            : {flat}")
print(f"reshape(3, 4)   :\n{flat.reshape(3, 4)}")
print(f"reshape(4, 3)   :\n{flat.reshape(4, 3)}")

# -1 means "work this dimension out for me". Use it so the code survives a
# change in the number of rows.
print(f"\nreshape(2, -1)  : shape {flat.reshape(2, -1).shape}")
print(f"reshape(-1, 6)  : shape {flat.reshape(-1, 6).shape}")

# %%
# The most common reshape in machine learning: turning a 1-D array into a
# single-column 2-D array, because scikit-learn requires 2-D feature matrices.
feature = np.array([1.0, 2.0, 3.0, 4.0])

print(f"feature.shape              : {feature.shape}")
print(f"feature.reshape(-1, 1).shape : {feature.reshape(-1, 1).shape}   ← a column")
print(f"feature[:, np.newaxis].shape : {feature[:, np.newaxis].shape}   ← same thing")
print(f"feature[:, None].shape       : {feature[:, None].shape}   ← same, terser")

# %% [markdown]
# ### `ravel` vs `flatten`
#
# Both give you a 1-D array. `ravel` returns a view when it can (cheap);
# `flatten` always copies (safe).

# %%
matrix = np.arange(6).reshape(2, 3)

raveled = matrix.ravel()
flattened = matrix.flatten()

print(f"ravel shares memory   : {np.shares_memory(matrix, raveled)}")
print(f"flatten shares memory : {np.shares_memory(matrix, flattened)}")

raveled[0] = 99
print(f"\nafter raveled[0] = 99, matrix =\n{matrix}   ← changed")

# %% [markdown]
# ### Transposing

# %%
m = np.arange(6).reshape(2, 3)

print(f"m       : shape {m.shape}\n{m}")
print(f"\nm.T     : shape {m.T.shape}\n{m.T}")
print(f"\nis a view : {np.shares_memory(m, m.T)}")
print(f"still contiguous? {m.T.flags['C_CONTIGUOUS']}   ← transposing breaks C-order")

# %% [markdown]
# > [!NOTE]
# > A transpose costs nothing — it only reinterprets the strides. But the result
# > is no longer C-contiguous, which can slow down later operations. Notebook 08
# > covers when to pay for `np.ascontiguousarray`.

# %% [markdown]
# ## 6. Combining and splitting arrays

# %%
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(f"vstack (stack rows, axis=0):\n{np.vstack([a, b])}")
print(f"\nhstack (stack columns, axis=1):\n{np.hstack([a, b])}")
print(f"\nconcatenate(axis=0) is the general form:\n{np.concatenate([a, b], axis=0)}")

# %%
# `stack` differs from `concatenate`: it adds a NEW axis rather than extending
# an existing one. This is how you build a batch out of individual samples.
stacked = np.stack([a, b])          # new axis 0

print(f"concatenate → shape {np.concatenate([a, b]).shape}")
print(f"stack       → shape {stacked.shape}   ← a new leading dimension")
print(f"\nstacked =\n{stacked}")

# %%
# Splitting is the inverse.
data = np.arange(12).reshape(6, 2)

first, second, third = np.split(data, 3)          # into 3 equal parts
print(f"split into 3 → shapes {first.shape}, {second.shape}, {third.shape}")

# Or at explicit indices — this is how a train/validation/test split works
# before scikit-learn does it for you.
train, validate, test = np.split(data, [4, 5])
print(f"\nsplit at [4, 5] → train {train.shape}, "
      f"validate {validate.shape}, test {test.shape}")

# %% [markdown]
# ## 7. A practical checklist
#
# When an array operation misbehaves, run through this in order:

# %%
def describe(arr: np.ndarray, name: str = "array") -> None:
    """Print everything worth knowing about an array in one go.

    Drop this into a notebook when something is not behaving — it answers the
    first five questions you would otherwise ask one at a time.
    """
    print(f"{name}")
    print(f"  shape      : {arr.shape}")
    print(f"  ndim       : {arr.ndim}")
    print(f"  dtype      : {arr.dtype}")
    print(f"  nbytes     : {arr.nbytes:,}")
    print(f"  is a view  : {arr.base is not None}")
    print(f"  contiguous : C={arr.flags['C_CONTIGUOUS']} "
          f"F={arr.flags['F_CONTIGUOUS']}")
    print(f"  writeable  : {arr.flags['WRITEABLE']}")
    if arr.size and np.issubdtype(arr.dtype, np.number):
        print(f"  finite     : {np.isfinite(arr).all()}")
        print(f"  range      : [{arr.min():.4g}, {arr.max():.4g}]")


sample = rng.normal(size=(3, 4))
describe(sample, "sample")

print()
describe(sample[:, 1:3], "sample[:, 1:3]")

# %% [markdown]
# ## 8. Exercises
#
# ### Exercise 2.1 — Predict the shapes
#
# Without running anything, give the shape of each result:
#
# ```python
# a = np.arange(24).reshape(2, 3, 4)
#
# a[0]            # ?
# a[:, 1]         # ?
# a[..., 2]       # ?
# a[0, :, ::2]    # ?
# a.T             # ?
# a.reshape(4, -1)  # ?
# ```
#
# ### Exercise 2.2 — View or copy?
#
# For each expression, state whether it shares memory with `a`, then verify with
# `np.shares_memory`:
#
# ```python
# a = np.arange(20).reshape(4, 5)
#
# a[1:3]         a[:, [0, 1]]      a.T[0]
# a[a % 2 == 0]  a.reshape(5, 4)   a.ravel()[::2]
# ```
#
# ### Exercise 2.3 — Fix the mutation bug
#
# ```python
# def scale_features(X, factor):
#     for j in range(X.shape[1]):
#         col = X[:, j]
#         col *= factor
#     return X
# ```
#
# Show that it corrupts the caller's array, then rewrite it so it does not.
# Write an assertion that would have caught the bug.
#
# ### Exercise 2.4 — Build a checkerboard
#
# Create an 8×8 array of alternating 0s and 1s (like a chessboard) **without any
# Python loop**. Then do it a second way. *Hint:* slicing with a step, or
# `np.indices`.
#
# ### Exercise 2.5 — Reshape for scikit-learn
#
# You have `y = np.array([1.0, 2.0, 3.0])` and a model that demands a 2-D array
# of shape `(n_samples, 1)`. Produce it three different ways, and confirm all
# three share memory with `y`.

# %% [markdown]
# ## Key takeaways
#
# - Check **`shape`, `ndim`, `dtype`, `size`** first when debugging. Most NumPy
#   bugs are shape bugs.
# - `(3,)`, `(3, 1)` and `(1, 3)` are three different things.
# - Prefer **`linspace` over `arange`** for floating-point ranges.
# - Use **`np.random.default_rng(seed)`**, not the legacy global functions.
# - **Basic slicing returns a VIEW** — the opposite of list slicing. Fancy
#   indexing and boolean masks return copies.
# - Check with **`.base`** or **`np.shares_memory`**; force independence with
#   **`.copy()`**.
# - In-place operators (`-=`, `*=`) on a view write through to the original.
# - `ravel` is a cheap view; `flatten` is always a copy.
# - `reshape(-1, 1)` turns a 1-D array into the column vector scikit-learn wants.
#
# ---
#
# **Next:** [03 · Dtypes & Numerical Precision](03_dtypes_and_precision.ipynb) —
# where Module 01's warning about silent integer overflow gets its full treatment.
