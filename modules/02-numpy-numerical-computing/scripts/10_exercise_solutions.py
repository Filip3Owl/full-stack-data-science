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
# # 10 · Exercise Solutions
#
# **Module 02 — NumPy & Numerical Computing**
#
# ---
#
# > [!WARNING]
# > **Attempt every exercise before reading its solution.** Reading a worked
# > solution produces a feeling of understanding that disappears the moment you
# > face a blank cell. Struggling for ten minutes teaches more than reading for
# > sixty.
#
# Forty exercises, one per section of notebooks 01–08. Where more than one answer
# is defensible the alternatives are shown with their trade-offs — on the job the
# interesting question is rarely "does it work?" but "which version would I want
# to maintain, and which one is still correct at 100× the data?"

# %%
import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics.pairwise import cosine_similarity

from dstoolkit import setup_plot_style
from dstoolkit.plotting import COLORS

setup_plot_style()
rng = np.random.default_rng(seed=42)

print(f"NumPy {np.__version__}")


# %%
def bench(fn, *args, repeat: int = 3, **kwargs) -> float:
    """Return the *best* wall-clock time of `repeat` runs, in seconds.

    Best-of rather than mean: a slow run means something else on the machine
    stole the CPU, which tells you nothing about the code being measured.
    """
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return best


# %% [markdown]
# ---
# # Notebook 01 — Why NumPy Exists

# %% [markdown]
# ## Exercise 1.1 — Measure it yourself
#
# Four ways to multiply a million pairs of floats. The only one that leaves the
# interpreter is the last.

# %%
N = 1_000_000
list_a = [float(i) for i in range(N)]
list_b = [float(i) * 2 for i in range(N)]
arr_a = np.array(list_a)
arr_b = np.array(list_b)


def with_for_loop(a, b):
    out = []
    for i in range(len(a)):
        out.append(a[i] * b[i])
    return out


def with_index_comprehension(a, b):
    return [a[i] * b[i] for i in range(len(a))]


def with_zip(a, b):
    return [x * y for x, y in zip(a, b, strict=True)]


def with_numpy(a, b):
    return a * b


approaches = {
    "for loop + append": (with_for_loop, list_a, list_b),
    "comprehension (index)": (with_index_comprehension, list_a, list_b),
    "comprehension (zip)": (with_zip, list_a, list_b),
    "NumPy  a * b": (with_numpy, arr_a, arr_b),
}

timings = {name: bench(fn, a, b) for name, (fn, a, b) in approaches.items()}
slowest = max(timings.values())

print(f"{'approach':<24} {'time':>10} {'speed-up':>10}")
print("-" * 46)
for name, seconds in timings.items():
    print(f"{name:<24} {seconds:>9.4f}s {slowest / seconds:>9.1f}×")

# %% [markdown]
# The three list versions are within a small factor of each other, because they
# are all doing the same thing: a million interpreter iterations, each one
# unboxing two `float` objects, multiplying, and boxing the result. `zip` is the
# fastest of them only because it skips the `__getitem__` calls.
#
# `arr_a * arr_b` is a different kind of operation. One call crosses into C, and
# the loop over a million contiguous `float64` values runs there — with no
# per-element type check, no boxing, and with SIMD instructions handling several
# values per clock cycle.

# %% [markdown]
# ## Exercise 1.2 — The honest memory question
#
# `sys.getsizeof` on a list is **shallow**: it counts the pointer array and the
# list object's own header, and stops. The objects being pointed at are not
# included, and for a list of numbers they are the bulk of the cost.

# %%
def true_list_size(lst: list) -> int:
    """Total bytes a list of numbers really occupies.

    Counts the list object (its header plus its array of pointers) and every
    distinct object it references.

    Notes
    -----
    `set()` on `id` matters: CPython interns small integers (-5 to 256), so a
    list of a million zeros references one shared object a million times.
    Counting it once is the honest answer.
    """
    seen: set[int] = set()
    total = sys.getsizeof(lst)
    for item in lst:
        if id(item) not in seen:
            seen.add(id(item))
            total += sys.getsizeof(item)
    return total


rows = []
for n in (10, 1_000, 100_000):
    # Values above 256 so interning does not flatter the list.
    values = list(range(1000, 1000 + n))
    array = np.array(values, dtype=np.int64)

    shallow = sys.getsizeof(values)
    honest = true_list_size(values)
    # `getsizeof` on an ndarray already includes the buffer, so it is the
    # array's honest total — the one case where the shallow measure is right.
    numpy_total = sys.getsizeof(array)

    rows.append(
        {
            "n": n,
            "getsizeof(list)": shallow,
            "true list size": honest,
            "understated by": f"{honest / shallow:.1f}×",
            "ndarray total": numpy_total,
            "list / ndarray": honest / numpy_total,
        }
    )

report = pd.DataFrame(rows)
print(report.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(report["n"], report["list / ndarray"], marker="o",
        color=COLORS["primary"], linewidth=2)
ax.axhline(1.0, color=COLORS["muted"], linestyle="--", linewidth=1)
ax.set_xscale("log")
ax.set_xlabel("number of elements")
ax.set_ylabel("list bytes ÷ ndarray bytes")
ax.set_title("A Python list of ints costs ~4.5× an int64 array", loc="left")
for x, y in zip(report["n"], report["list / ndarray"], strict=True):
    ax.annotate(f"{y:.1f}×", (x, y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=9)
ax.set_ylim(0, report["list / ndarray"].max() * 1.25)
plt.tight_layout()
plt.show()

# %% [markdown]
# The ratio **rises** with `n` and then flattens near 4.5×. At `n = 10` the list's
# own 56-byte header is a visible share of the total; by `n = 100 000` the cost is
# entirely 8 bytes of pointer plus 28 bytes of `int` object per element, against
# 8 bytes in the array.
#
# The lesson is not the exact number, it is that **`sys.getsizeof` understates a
# list of numbers by roughly 4×** and will happily tell you a list is smaller than
# the array holding the same data.

# %% [markdown]
# ## Exercise 1.3 — Find the break-even point
#
# NumPy is not free. Every call does argument parsing, dtype resolution, shape
# broadcasting, and output allocation before a single element is touched. That
# fixed cost is a few microseconds — which a list comprehension beats outright
# when there are only a handful of elements to process.

# %%
sizes = np.unique(np.geomspace(1, 10_000, 40).astype(int))
list_times, numpy_times = [], []

for n in sizes:
    values = list(range(n))
    array = np.arange(n)
    # More repeats at small n, where a single run is too short to time.
    reps = 200 if n < 1000 else 20
    list_times.append(bench(lambda v=values: [x * 2 + 1 for x in v], repeat=reps))
    numpy_times.append(bench(lambda a=array: a * 2 + 1, repeat=reps))

list_times = np.array(list_times)
numpy_times = np.array(numpy_times)

faster = numpy_times < list_times
break_even = sizes[faster][0] if faster.any() else None
print(f"NumPy first wins at n = {break_even}")
print(f"At n = {sizes[-1]}: NumPy is {list_times[-1] / numpy_times[-1]:.0f}× faster")

# %%
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(sizes, list_times * 1e6, label="list comprehension",
        color=COLORS["danger"], linewidth=2)
ax.plot(sizes, numpy_times * 1e6, label="NumPy  a * 2 + 1",
        color=COLORS["teal"], linewidth=2)
if break_even is not None:
    ax.axvline(break_even, color=COLORS["muted"], linestyle="--", linewidth=1)
    ax.annotate(f"break-even ≈ {break_even}", (break_even, ax.get_ylim()[1] * 0.4),
                xytext=(12, 0), textcoords="offset points",
                fontsize=9, color=COLORS["muted"])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of elements")
ax.set_ylabel("time (µs)")
ax.set_title("NumPy's fixed call overhead only pays off above ~100 elements",
             loc="left")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Why is it not 1?** Below the break-even point the whole runtime is NumPy's
# per-call overhead — the flat left-hand section of the teal line, which does not
# move as `n` grows. The list comprehension has almost no fixed cost but pays per
# element, so it starts far lower and climbs linearly. Where the two meet is the
# break-even point.
#
# The practical consequence: **calling NumPy inside a Python loop over small
# arrays is the worst of both worlds.** You pay the overhead thousands of times
# and never reach the size where the vectorised inner loop earns it back.

# %% [markdown]
# ## Exercise 1.4 — Spot the de-vectorisation
#
# All four go back through the interpreter. The first three are worse than the
# list equivalents, because NumPy has to build a Python object for every element
# it hands back.

# %%
arr = rng.random(1_000_000)

cases = {
    "(a) sum(arr)": (lambda: sum(arr), lambda: arr.sum()),
    "(b) np.array([np.sqrt(v) for v in arr])":
        (lambda: np.array([np.sqrt(v) for v in arr]), lambda: np.sqrt(arr)),
    "(c) max(arr)": (lambda: max(arr), lambda: arr.max()),
    "(d) np.array(list(map(lambda x: x * 2, arr)))":
        (lambda: np.array(list(map(lambda x: x * 2, arr))), lambda: arr * 2),
}

print(f"{'expression':<46} {'slow':>9} {'fixed':>9} {'speed-up':>9}")
print("-" * 76)
for label, (slow, fast) in cases.items():
    t_slow = bench(slow, repeat=1)
    t_fast = bench(fast, repeat=3)
    print(f"{label:<46} {t_slow:>8.4f}s {t_fast:>8.4f}s {t_slow / t_fast:>8.0f}×")

# %% [markdown]
# | | Why it is slow | Fix |
# | :-- | :--- | :--- |
# | **(a)** | Python's `sum` iterates the array, boxing each `float64` into a Python `float`, then adds with the interpreter's dispatch. It is *slower than summing a list*, because a list already holds Python objects. | `arr.sum()` |
# | **(b)** | `np.sqrt` on a **scalar** returns a `np.float64` — one full ufunc dispatch per element, then a list build, then a re-parse into an array. | `np.sqrt(arr)` |
# | **(c)** | Same problem as (a): `max` is an interpreter loop with a comparison per element. | `arr.max()` |
# | **(d)** | `map` with a Python `lambda` is a Python-level loop wearing a functional disguise. | `arr * 2` |
#
# The tell is always the same: **a Python-level `for`, comprehension, `map`, or
# built-in aggregate with an array inside it.** If the array appears as an
# *iterable* rather than as an *operand*, you have left the fast path.

# %% [markdown]
# ## Exercise 1.5 — `out=` in a loop
#
# An iterative computation is where allocation cost stops being noise. Every
# round of the naive version asks the allocator for three fresh 8 MB blocks and
# then throws them away.

# %%
ROUNDS = 100
x = rng.random(1_000_000) + 1.0


def allocating(x, rounds=ROUNDS):
    """Fresh arrays every round — three temporaries per iteration."""
    for _ in range(rounds):
        y = np.sqrt(x) + np.log1p(x)
    return y


def reusing(x, rounds=ROUNDS):
    """Two preallocated buffers, reused for every round."""
    buf_a = np.empty_like(x)
    buf_b = np.empty_like(x)
    for _ in range(rounds):
        np.sqrt(x, out=buf_a)
        np.log1p(x, out=buf_b)
        np.add(buf_a, buf_b, out=buf_a)
    return buf_a


assert np.allclose(allocating(x, 1), reusing(x, 1))

t_alloc = bench(allocating, x, repeat=1)
t_reuse = bench(reusing, x, repeat=1)

print(f"allocating each round : {t_alloc:.3f}s")
print(f"reusing buffers       : {t_reuse:.3f}s")
print(f"speed-up              : {t_alloc / t_reuse:.2f}×")
print(f"\nBytes NOT allocated   : {ROUNDS * 3 * x.nbytes / 1e9:.1f} GB")

# %% [markdown]
# The speed-up is usually modest — somewhere between 1.2× and 1.6× — and that is
# the honest result worth internalising. `sqrt` and `log1p` are transcendental
# functions; the arithmetic genuinely dominates. `out=` buys you the allocation
# and the memory traffic, not the maths.
#
# Where `out=` becomes decisive is when the operation is **cheap relative to its
# memory traffic** (`a + b`, `a * 2`), or when the array is large enough that the
# allocation itself starts hitting the OS for fresh pages. Reach for it when
# profiling points at allocation, not on principle.

# %% [markdown]
# ---
# # Notebook 02 — ndarray Fundamentals

# %% [markdown]
# ## Exercise 2.1 — Predict the shapes
#
# | Expression | Shape | Reasoning |
# | :--- | :-- | :--- |
# | `a[0]` | `(3, 4)` | An integer index **removes** that axis. |
# | `a[:, 1]` | `(2, 4)` | Integer on axis 1 removes it; axes 0 and 2 survive. |
# | `a[..., 2]` | `(2, 3)` | `...` fills in "all remaining axes"; the last one is indexed away. |
# | `a[0, :, ::2]` | `(3, 2)` | Axis 0 removed, axis 1 kept whole, axis 2 taken every other element. |
# | `a.T` | `(4, 3, 2)` | `.T` reverses the axis order — for 3-D it is not "swap the last two". |
# | `a.reshape(4, -1)` | `(4, 6)` | `-1` means "whatever makes 24 elements fit". |

# %%
a = np.arange(24).reshape(2, 3, 4)

expressions = {
    "a[0]": a[0],
    "a[:, 1]": a[:, 1],
    "a[..., 2]": a[..., 2],
    "a[0, :, ::2]": a[0, :, ::2],
    "a.T": a.T,
    "a.reshape(4, -1)": a.reshape(4, -1),
}
for label, result in expressions.items():
    print(f"{label:<20} {str(result.shape):<12}")

# %% [markdown]
# The rule that resolves almost all of these: **an integer index deletes an axis,
# a slice keeps it.** `a[0]` has one fewer dimension than `a`; `a[0:1]` has the
# same number. That single distinction is behind most "why is my shape `(n,)`
# when I wanted `(n, 1)`" confusion.

# %% [markdown]
# ## Exercise 2.2 — View or copy?
#
# The rule: **basic slicing gives a view; fancy indexing and boolean masks give a
# copy.** A view is possible whenever the selected elements can be described by a
# start offset and a fixed stride per axis — which an arbitrary list of indices
# cannot be.

# %%
a = np.arange(20).reshape(4, 5)

candidates = {
    "a[1:3]": (a[1:3], "basic slice → view"),
    "a[:, [0, 1]]": (a[:, [0, 1]], "fancy (list of indices) → copy"),
    "a.T[0]": (a.T[0], "transpose + basic index → view (stride 5)"),
    "a[a % 2 == 0]": (a[a % 2 == 0], "boolean mask → copy"),
    "a.reshape(5, 4)": (a.reshape(5, 4), "reshape of contiguous data → view"),
    "a.ravel()[::2]": (a.ravel()[::2], "ravel is a view; ::2 is a basic slice → view"),
}

print(f"{'expression':<18} {'shares memory':>14}   {'reason'}")
print("-" * 78)
for label, (result, reason) in candidates.items():
    print(f"{label:<18} {str(np.shares_memory(a, result)):>14}   {reason}")

# %% [markdown]
# Two traps hiding in that table:
#
# - **`reshape` is a view only when it can be.** On contiguous data it just
#   reinterprets the shape. On a non-contiguous array it has to copy — so you
#   cannot rely on it either way. Check, do not assume.
# - **`ravel` vs `flatten`.** `ravel` returns a view when it can; `flatten`
#   always copies. If you write to the result of `ravel`, you may be writing to
#   the original.

# %%
# reshape is NOT always a view — proof:
non_contiguous = a[:, ::2]
print(f"a[:, ::2] is contiguous : {non_contiguous.flags['C_CONTIGUOUS']}")
print(f"its reshape shares mem  : "
      f"{np.shares_memory(a, non_contiguous.reshape(-1))}")

# %% [markdown]
# ## Exercise 2.3 — Fix the mutation bug
#
# `col = X[:, j]` is a **view**. `col *= factor` is an in-place multiply, so it
# writes straight through into the caller's array. The function has no `copy`
# anywhere, so it silently mutates its input and *also* returns it — the worst
# combination, because the caller has every reason to think they got something new.

# %%
def scale_features(X, factor):
    """The buggy original."""
    for j in range(X.shape[1]):
        col = X[:, j]
        col *= factor
    return X


X_original = np.arange(12, dtype=float).reshape(4, 3)
X_before = X_original.copy()

scaled = scale_features(X_original, 10.0)

print("Caller's array after the call:")
print(X_original)
print(f"\nWas the caller's array modified? {not np.array_equal(X_original, X_before)}")
print(f"Is the 'result' the same object? {scaled is X_original}")

# %%
def scale_features_fixed(X: np.ndarray, factor: float) -> np.ndarray:
    """Scale every feature by `factor`, leaving the input untouched.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data. Not modified.
    factor : float
        Multiplier applied to every element.

    Returns
    -------
    ndarray
        A new array; never a view onto `X`.
    """
    return X * factor  # `*` allocates; `*=` would not


X_original = np.arange(12, dtype=float).reshape(4, 3)
X_before = X_original.copy()

result = scale_features_fixed(X_original, 10.0)

assert np.array_equal(X_original, X_before), "input was mutated"
assert not np.shares_memory(result, X_original), "result aliases the input"
print("Input untouched, result independent. ✔")

# %% [markdown]
# ### The assertion that would have caught it
#
# In a test, the pattern is: snapshot the input, call, compare.

# %%
def test_scale_features_does_not_mutate(fn) -> str:
    X = np.arange(12, dtype=float).reshape(4, 3)
    snapshot = X.copy()
    out = fn(X, 10.0)
    try:
        assert np.array_equal(X, snapshot), "function mutated its input array"
        assert not np.shares_memory(out, X), "output is a view onto the input"
    except AssertionError as exc:
        return f"FAIL — {exc}"
    return "PASS"


print(f"scale_features       : {test_scale_features_does_not_mutate(scale_features)}")
print(f"scale_features_fixed : "
      f"{test_scale_features_does_not_mutate(scale_features_fixed)}")

# %% [markdown]
# > If a function is documented as pure, `np.shares_memory(out, inp)` is a
# > one-line regression test for it. This class of bug is the NumPy equivalent of
# > Module 01's mutable-default-argument trap, and it is much harder to spot,
# > because the corruption shows up in a cell you ran twenty minutes ago.

# %% [markdown]
# ## Exercise 2.4 — Build a checkerboard

# %%
# Way 1 — strided assignment. Fastest, and the one you will see in the wild.
board_slicing = np.zeros((8, 8), dtype=int)
board_slicing[1::2, ::2] = 1   # odd rows,  even columns
board_slicing[::2, 1::2] = 1   # even rows, odd columns

# Way 2 — arithmetic on coordinates. A square is dark when (row + col) is odd.
rows, cols = np.indices((8, 8))
board_indices = (rows + cols) % 2

# Way 3 — the same idea via broadcasting, without materialising two grids.
board_broadcast = (np.arange(8)[:, None] + np.arange(8)) % 2

assert np.array_equal(board_slicing, board_indices)
assert np.array_equal(board_slicing, board_broadcast)
print(board_slicing)

# %% [markdown]
# **Which would I keep?** `(np.arange(8)[:, None] + np.arange(8)) % 2` — it states
# the rule ("dark when row + col is odd") rather than the mechanics, it
# generalises to any size in one edit, and it never allocates the two full
# coordinate grids `np.indices` builds. The slicing version is fine but you have
# to read it twice to convince yourself the two lines cover every square exactly
# once.

# %% [markdown]
# ## Exercise 2.5 — Reshape for scikit-learn

# %%
y = np.array([1.0, 2.0, 3.0])

as_column = {
    "y.reshape(-1, 1)": y.reshape(-1, 1),
    "y[:, np.newaxis]": y[:, np.newaxis],
    "np.expand_dims(y, axis=1)": np.expand_dims(y, axis=1),
}

for label, result in as_column.items():
    print(f"{label:<28} shape {result.shape}   "
          f"shares memory: {np.shares_memory(y, result)}")

# %% [markdown]
# All three are the same operation with three spellings, and all three are
# **views** — adding a length-1 axis changes only the shape and strides metadata,
# never the buffer.
#
# Prefer **`y.reshape(-1, 1)`**: it works without knowing `len(y)`, it is the
# spelling scikit-learn's own error message suggests, and `-1` makes the intent
# ("one column, however many rows that takes") explicit.
#
# `y[:, None]` is the same as `y[:, np.newaxis]` — `np.newaxis` *is* `None`. It is
# terser and very common in library code; it is also the less readable choice in
# a notebook someone else will inherit.

# %%
# Because they are views, writing through one changes `y`:
view = y.reshape(-1, 1)
view[0, 0] = 99.0
print(f"y after writing to the reshaped view: {y}")

# %% [markdown]
# ---
# # Notebook 03 — Dtypes & Numerical Precision

# %% [markdown]
# ## Exercise 3.1 — Predict the dtype
#
# | Expression | Result | Rule |
# | :--- | :-- | :--- |
# | `int8 + int16` | `int16` | Both are arrays: promote to the smaller type that holds both. |
# | `int32 / 2` | `float64` | **True division always produces a float**, whatever went in. |
# | `float32 + 1.0` | `float32` | A plain Python `1.0` is a *weak* scalar — it adopts the array's dtype. |
# | `float32 + np.float64(1.0)` | `float64` | `np.float64(...)` is a **typed** value, so normal promotion applies. |
# | `bool + int64` | `int64` | `bool` promotes to the integer type. |
# | `np.arange(3) ** 2` | `int64` | `arange` gives the platform default integer; `**` keeps it. |

# %%
predictions = {
    "int8 + int16":
        np.array([1, 2], dtype=np.int8) + np.array([1, 2], dtype=np.int16),
    "int32 / 2":
        np.array([1, 2], dtype=np.int32) / 2,
    "float32 + 1.0":
        np.array([1.0], dtype=np.float32) + 1.0,
    "float32 + np.float64(1.0)":
        np.array([1.0], dtype=np.float32) + np.float64(1.0),
    "bool + int":
        np.array([True, False]) + np.array([1, 2]),
    "np.arange(3) ** 2":
        np.arange(3) ** 2,
}
for label, result in predictions.items():
    print(f"{label:<28} → {result.dtype}")

# %% [markdown]
# Rows 3 and 4 are the pair worth committing to memory, and they are the ones
# NumPy 2.0 changed (NEP 50). **A Python scalar never widens an array; a NumPy
# scalar does.** In a `float32` pipeline built to halve memory, a single stray
# `np.float64(...)` — or a `np.mean()` result fed back in — silently doubles it
# again.

# %%
# The realistic version of that mistake:
features = rng.random((1000, 20)).astype(np.float32)
centred_ok = features - features.mean(axis=0, dtype=np.float32)
centred_oops = features - features.mean(axis=0)  # float64 accumulator by default

print(f"features            : {features.dtype}, {features.nbytes / 1024:.0f} KiB")
print(f"centred (careful)   : {centred_ok.dtype}, {centred_ok.nbytes / 1024:.0f} KiB")
print(f"centred (default)   : {centred_oops.dtype}, "
      f"{centred_oops.nbytes / 1024:.0f} KiB  ← doubled")

# %% [markdown]
# ## Exercise 3.2 — Trigger and then prevent an overflow
#
# There is a twist in this one. Try the obvious thing first:

# %%
values = np.full(100, 1000, dtype=np.int16)
print(f"int16 max        : {np.iinfo(np.int16).max}")
print(f"values.sum()     : {values.sum()}  (dtype {values.sum().dtype})")

# %% [markdown]
# **It did not overflow** — and the answer is not even `int16`. `np.sum` widens
# its accumulator to the platform default integer for any narrower input dtype,
# precisely because summing into an `int16` would be a footgun.
#
# So where does `int16` actually bite? Wherever the accumulator is *not* widened
# for you: element-wise arithmetic, and any reduction where you pin the dtype
# yourself.

# %%
overflow_cases = {
    "values.sum(dtype=np.int16)": values.sum(dtype=np.int16),
    "np.add.reduce(values, dtype=np.int16)": np.add.reduce(values, dtype=np.int16),
    "np.cumsum(values, dtype=np.int16)[-1]": np.cumsum(values, dtype=np.int16)[-1],
    "(values * 400)[0]": (values * 400)[0],
}
true_total = 100 * 1000

print(f"true total: {true_total}\n")
for label, result in overflow_cases.items():
    print(f"{label:<40} → {result:>8}   {'WRONG' if result < 0 else ''}")

# %% [markdown]
# Element-wise `values * 400` is the dangerous one: `1000 * 400 = 400 000`, far
# past `int16`'s 32 767, so it wraps to `6784`. **No warning, no exception, just a
# wrong number** — NumPy 2 raises `RuntimeWarning` for scalar overflow but stays
# silent for arrays.
#
# ### Two ways to get it right

# %%
# Fix 1 — widen the accumulator, keep the storage narrow.
#         Costs nothing: the array stays int16 in memory.
fix_accumulator = values.sum(dtype=np.int64)

# Fix 2 — widen the data before the arithmetic.
#         Costs 4× the memory, but every downstream operation is then safe.
fix_upcast = values.astype(np.int64) * 400

print(f"Fix 1  values.sum(dtype=np.int64)      : {fix_accumulator}  ✔")
print(f"Fix 2  values.astype(np.int64) * 400   : {fix_upcast[0]}  ✔")

# %%
# Fix 3 — the defensive habit: make the failure loud instead of silent.
with np.errstate(over="raise"):
    try:
        _ = np.multiply(values, np.int16(400))
    except FloatingPointError as exc:
        print(f"np.errstate(over='raise') → {type(exc).__name__}: {exc}")

# %% [markdown]
# > **The rule for narrow integer dtypes:** narrow storage is a *storage*
# > decision. The moment you do arithmetic, ask what the largest possible
# > intermediate is — not the largest value in the column today.

# %% [markdown]
# ## Exercise 3.3 — When is float32 not enough?
#
# This exercise has a surprise in it too. Sum a million `float32` copies of 0.1
# and NumPy gets it almost exactly right — because `np.sum` does **pairwise
# summation**, not a naive left-to-right accumulation.

# %%
n = 1_000_000
x32 = np.full(n, 0.1, dtype=np.float32)
exact = n * 0.1

pairwise = float(np.sum(x32))
sequential = float(np.cumsum(x32, dtype=np.float32)[-1])

print(f"exact total              : {exact:,.2f}")
print(f"np.sum      (pairwise)   : {pairwise:,.2f}   "
      f"rel err {abs(pairwise - exact) / exact:.2e}")
print(f"np.cumsum   (sequential) : {sequential:,.2f}   "
      f"rel err {abs(sequential - exact) / exact:.2e}")

# %% [markdown]
# Same numbers, same dtype, **five orders of magnitude** difference in accuracy —
# the only variable is the order of the additions.
#
# The mechanism: once the running total reaches ~2 000, the gap between adjacent
# `float32` values exceeds the 0.1 being added, so each addition rounds part of
# the increment away. Pairwise summation keeps the partial sums small and similar
# in magnitude, so this never happens.

# %%
sizes = np.unique(np.geomspace(10, 3_000_000, 30).astype(int))
records = []

for size in sizes:
    truth = size * 0.1
    for dtype, name in ((np.float32, "float32"), (np.float64, "float64")):
        chunk = np.full(size, 0.1, dtype=dtype)
        records.append({
            "n": size,
            "dtype": name,
            "sequential": abs(float(np.cumsum(chunk, dtype=dtype)[-1]) - truth) / truth,
            "pairwise": abs(float(np.sum(chunk)) - truth) / truth,
        })

errors = pd.DataFrame(records)

threshold = 1e-3
drifted = errors.query("dtype == 'float32' and sequential > @threshold")
first_bad = int(drifted["n"].iloc[0]) if len(drifted) else None
print(f"float32 sequential summation passes 0.1% error at n ≈ {first_bad:,}")

# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
styles = {
    ("float32", "sequential"): (COLORS["danger"], "-", "float32, sequential"),
    ("float32", "pairwise"): (COLORS["accent"], "--", "float32, np.sum (pairwise)"),
    ("float64", "sequential"): (COLORS["primary"], "-", "float64, sequential"),
    ("float64", "pairwise"): (COLORS["teal"], "--", "float64, np.sum (pairwise)"),
}
for (dtype, method), (color, style, label) in styles.items():
    subset = errors[errors["dtype"] == dtype]
    # Clip at the float64 floor so log scale stays readable when error is 0.
    ax.plot(subset["n"], np.maximum(subset[method], 1e-18),
            color=color, linestyle=style, linewidth=2, label=label)

ax.axhline(threshold, color=COLORS["muted"], linestyle=":", linewidth=1.2)
ax.annotate("0.1% error", (sizes[0], threshold), xytext=(0, 6),
            textcoords="offset points", fontsize=9, color=COLORS["muted"])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of values summed")
ax.set_ylabel("relative error")
ax.set_title("Summation error: the dtype matters, the algorithm matters more",
             loc="left")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# **What to take away.** `float32` halves your memory and, on a GPU, roughly
# doubles your throughput — that is why deep learning lives there. It is fine for
# storing features and model weights. It is *not* fine for long accumulations:
# sums, means, variances, dot products of long vectors, and anything iterative.
#
# The practical rule is to **store narrow and accumulate wide**:
# `x32.sum(dtype=np.float64)`. You keep the memory saving and lose the error.

# %%
print(f"x32.sum()                  : {float(x32.sum()):,.4f}")
print(f"x32.sum(dtype=np.float64)  : {x32.sum(dtype=np.float64):,.4f}")
print(f"exact                      : {exact:,.4f}")

# %% [markdown]
# ## Exercise 3.4 — Build a memory report

# %%
def narrowest_dtype(values: np.ndarray) -> np.dtype:
    """Smallest dtype that represents every element of `values` exactly.

    Integers are checked against each candidate's range. Floats are checked by
    round-tripping — a float32 is only "safe" if casting down and back returns
    the identical array, which is a far stricter test than `allclose`.
    """
    kind = values.dtype.kind

    if kind in "iu":
        low, high = values.min(), values.max()
        candidates = ([np.uint8, np.uint16, np.uint32, np.uint64] if low >= 0
                      else [np.int8, np.int16, np.int32, np.int64])
        for candidate in candidates:
            info = np.iinfo(candidate)
            if info.min <= low and high <= info.max:
                return np.dtype(candidate)

    elif kind == "f":
        for candidate in (np.float16, np.float32):
            # NaNs compare unequal to themselves, so compare them positionally.
            down = values.astype(candidate).astype(values.dtype)
            same = np.array_equal(down, values, equal_nan=True)
            if same:
                return np.dtype(candidate)

    return values.dtype


def memory_report(arrays: dict[str, np.ndarray]) -> None:
    """Print a per-column downcasting report.

    Columns whose name ends in ``_id`` are never recommended for downcasting: an
    identifier is a label that happens to be stored as a number, and today's
    maximum is no guarantee about tomorrow's.
    """
    header = f"{'column':<16}{'dtype':>9}{'size':>11}{'suggested':>11}{'saving':>10}"
    print(header)
    print("-" * len(header))

    total_now = total_after = 0

    for name, values in arrays.items():
        current_bytes = values.nbytes
        protected = name.endswith("_id")
        target = values.dtype if protected else narrowest_dtype(values)
        new_bytes = values.size * np.dtype(target).itemsize

        total_now += current_bytes
        total_after += new_bytes

        if protected:
            note, saving = "— (id)", ""
        elif target == values.dtype:
            note, saving = "—", ""
        else:
            note = str(target)
            saving = f"{100 * (1 - new_bytes / current_bytes):.0f}%"

        print(f"{name:<16}{str(values.dtype):>9}"
              f"{current_bytes / 1024:>10.0f}K{note:>11}{saving:>10}")

    print("-" * len(header))
    print(f"{'TOTAL':<16}{'':>9}{total_now / 1024:>10.0f}K"
          f"{'':>11}{100 * (1 - total_after / total_now):>9.0f}%")


# %%
n_rows = 200_000
frame = {
    "customer_id": rng.integers(1, 5_000, n_rows),            # protected
    "order_id": rng.integers(1, 200_000, n_rows),             # protected
    "age": rng.integers(18, 90, n_rows),                      # → uint8
    "items": rng.integers(1, 30, n_rows),                     # → uint8
    "postcode": rng.integers(1000, 9999, n_rows),             # → uint16
    "price": np.round(rng.uniform(1, 500, n_rows), 2),        # stays float64
    "discount": rng.choice([0.0, 0.25, 0.5], n_rows),         # → float16
}
memory_report(frame)

# %% [markdown]
# `price` refuses to narrow, and that is the report working correctly: two-decimal
# money values are not exactly representable in `float32`, so the round-trip test
# fails. `discount` narrows all the way to `float16` because `0.0`, `0.25` and
# `0.5` are exact binary fractions.
#
# The `_id` guard is a real-world scar. Downcasting an ID column to `int32`
# because today's maximum is 2 million works right up until the table passes
# 2.1 billion rows — and then it wraps, and two different customers become the
# same customer. **Identifiers are labels, not quantities.** Ideally they are not
# stored as numbers at all.

# %% [markdown]
# ## Exercise 3.5 — Catastrophic cancellation in the wild

# %%
p = 1 - 1e-17
n = 1000

print(f"p == 1.0 exactly?        {p == 1.0}")
print(f"spacing of float64 at 1: {np.spacing(1.0):.3e}")
print(f"the offset we asked for: {1e-17:.3e}   ← 22× smaller than the spacing")

# %%
# The direct formula.
se_direct = np.sqrt(p * (1 - p) / n)

# The rearranged one: parameterise by q = 1 - p and never form `p` at all.
q = 1e-17
se_stable = np.sqrt(q * (1 - q) / n)

print(f"\nsqrt(p * (1 - p) / n) : {se_direct:.6e}   ← every digit lost")
print(f"sqrt(q * (1 - q) / n) : {se_stable:.6e}   ← correct")
print(f"exact (sqrt(1e-17/1000)) : {np.sqrt(1e-17 / 1000):.6e}")

# %% [markdown]
# ### What actually happened
#
# The damage was done **before any arithmetic ran**. `1 - 1e-17` cannot be
# represented in `float64`: near 1.0 the gap between consecutive representable
# values is `2.22e-16`, and we asked for an offset 22× smaller than that. The
# literal rounds to exactly `1.0`, so by the time the formula sees `p`, the
# quantity that carried all the information — the `1e-17` — is simply gone.
# Then `1 - p` is `0.0`, and the standard error is reported as zero: **a
# confidence interval of zero width around a probability of 1.**
#
# The fix is not more precision, it is a **change of variable**. Carry the small
# quantity `q = 1 - p` directly and never materialise `p`. `q * (1 - q)` is
# evaluated near `1e-17`, where float64 has plenty of resolution.

# %%
# The same disease, and the same cure, in three places you will actually meet it.
tiny = 1e-18
print(f"{'naive':<28}{'stable':<28}{'why'}")
print("-" * 92)
print(f"{f'log(1+{tiny:g}) = {np.log(1 + tiny):.3e}':<28}"
      f"{f'log1p({tiny:g}) = {np.log1p(tiny):.3e}':<28}"
      f"1 + tiny rounds to 1")
print(f"{f'exp({tiny:g})-1 = {np.exp(tiny) - 1:.3e}':<28}"
      f"{f'expm1({tiny:g}) = {np.expm1(tiny):.3e}':<28}"
      f"exp(tiny) rounds to 1")

variance_naive = lambda v: (v ** 2).mean() - v.mean() ** 2  # noqa: E731
big_offset = np.array([1e9, 1e9 + 1, 1e9 + 2], dtype=np.float64)
print(f"{f'E[x²]-E[x]² = {variance_naive(big_offset):.4f}':<28}"
      f"{f'np.var(x) = {np.var(big_offset):.4f}':<28}"
      f"two huge, nearly equal terms")

# %% [markdown]
# > **The signature to watch for:** subtracting two numbers that are nearly
# > equal, or adding a small number to a large one. Both destroy the leading
# > digits of the small quantity. Whenever a formula contains `1 - p`, `a - b`,
# > or `E[x²] - E[x]²`, ask whether there is an algebraically identical form that
# > keeps the small quantity small. There usually is, and it usually already
# > exists in `numpy` or `scipy` (`log1p`, `expm1`, `logaddexp`, `np.var`).

# %% [markdown]
# ---
# # Notebook 04 — Broadcasting

# %% [markdown]
# ## Exercise 4.1 — Compatible or not?
#
# Work right to left. Pad the shorter shape with `1`s **on the left**, then a
# pair of dimensions is compatible when they are equal or one of them is 1.
#
# | Pair | Padded | Result | Why |
# | :--- | :--- | :-- | :--- |
# | `(5, 4)` and `(4,)` | `(1, 4)` | `(5, 4)` | 4=4 ✔, 5 vs 1 → stretch ✔ |
# | `(5, 4)` and `(5,)` | `(1, 5)` | **error** | 4 vs 5, neither is 1 |
# | `(5, 4)` and `(5, 1)` | — | `(5, 4)` | 4 vs 1 → stretch ✔, 5=5 ✔ |
# | `(3, 1, 4)` and `(2, 4)` | `(1, 2, 4)` | `(3, 2, 4)` | 4=4, 1 vs 2 → stretch, 3 vs 1 → stretch |
# | `(2, 3, 4)` and `(3, 1)` | `(1, 3, 1)` | `(2, 3, 4)` | 4 vs 1 → stretch, 3=3, 2 vs 1 → stretch |
# | `(8, 1, 6)` and `(7, 1)` | `(1, 7, 1)` | `(8, 7, 6)` | 6 vs 1, 1 vs 7, 8 vs 1 — all stretch |

# %%
pairs = [
    ((5, 4), (4,)),
    ((5, 4), (5,)),
    ((5, 4), (5, 1)),
    ((3, 1, 4), (2, 4)),
    ((2, 3, 4), (3, 1)),
    ((8, 1, 6), (7, 1)),
]

for left, right in pairs:
    try:
        result = str(np.broadcast_shapes(left, right))
    except ValueError:
        result = "incompatible"
    print(f"{str(left):<12} and {str(right):<8} → {result}")

# %% [markdown]
# The second row is the one that catches people, and it catches them because the
# *intent* was almost certainly legal. `(5, 4)` with `(5,)` reads like "one value
# per row", but broadcasting aligns from the right, so it tries to match the 5
# against the 4 columns. Say what you mean with `right.reshape(-1, 1)`.

# %%
left = np.zeros((5, 4))
per_row = np.arange(5)
print(f"left + per_row.reshape(-1, 1) → "
      f"{(left + per_row.reshape(-1, 1)).shape}   ✔ one value per row")

# %% [markdown]
# ## Exercise 4.2 — Fix the silent bug

# %%
y_true = np.array([[1.0], [2.0], [3.0]])   # (3, 1) — a column
y_pred = np.array([1.1, 1.9, 3.2])         # (3,)   — a flat vector

residuals = y_true - y_pred
print(f"y_true    : {y_true.shape}")
print(f"y_pred    : {y_pred.shape}")
print(f"residuals : {residuals.shape}   ← should be (3,)")
print(residuals)

# %% [markdown]
# `(3,)` pads to `(1, 3)`; against `(3, 1)` both dimensions stretch and you get
# the full `(3, 3)` **outer difference** — every true value minus every predicted
# value. Only the diagonal is meaningful.
#
# The reason this is dangerous rather than merely wrong: it does not raise. A
# downstream `.mean()` returns a number, the notebook runs clean, and you report
# an MSE computed over nine residuals instead of three.

# %%
mse_wrong = (residuals ** 2).mean()
mse_right = ((y_true.ravel() - y_pred) ** 2).mean()
print(f"MSE from the (3, 3) array : {mse_wrong:.4f}   ← wrong, and plausible")
print(f"MSE from the (3,) array   : {mse_right:.4f}   ← correct")
print(f"the diagonal is the real one: {np.diag(residuals)}")

# %%
def residuals_safe(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Element-wise residuals, with the shape contract enforced.

    Flattening both inputs is the right call here: scikit-learn returns `(n,)`
    from `predict` but `(n, 1)` targets are extremely common, and silently
    accepting either is safer than the outer product you get otherwise.
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"length mismatch: y_true has {y_true.size}, y_pred has {y_pred.size}"
        )
    return y_true - y_pred


fixed = residuals_safe(y_true, y_pred)
assert fixed.shape == (3,)
print(f"residuals_safe → {fixed.shape}  {np.round(fixed, 2)}")

# The assertion that would have caught the original bug:
assert residuals.ndim == 1, (
    f"residuals should be 1-D, got {residuals.shape} — "
    "an unintended outer broadcast"
)

# %% [markdown]
# > **The habit:** after any subtraction of two arrays that came from different
# > places, assert the output shape. `assert out.shape == expected` is one line
# > and it converts an entire class of silent numerical bugs into a loud
# > traceback on the line that caused them.

# %% [markdown]
# ## Exercise 4.3 — Normalise rows instead of columns

# %%
X = rng.random((100, 5))

row_normalised = X / X.sum(axis=1, keepdims=True)
print(f"X.sum(axis=1)                shape {X.sum(axis=1).shape}")
print(f"X.sum(axis=1, keepdims=True) shape {X.sum(axis=1, keepdims=True).shape}")
print(f"every row sums to 1: "
      f"{np.allclose(row_normalised.sum(axis=1), 1.0)}")

# %%
try:
    X / X.sum(axis=1)
except ValueError as exc:
    print(f"X / X.sum(axis=1) → ValueError: {exc}")

# %% [markdown]
# **Why it fails.** `X.sum(axis=1)` collapses axis 1, leaving shape `(100,)`.
# Broadcasting pads that to `(1, 100)` and tries to line it up against
# `(100, 5)`: 100 against 5, neither is 1, error.
#
# **Why `keepdims=True` works.** It leaves the collapsed axis in place with
# length 1, so the shape is `(100, 1)`. Against `(100, 5)`: 1 stretches to 5,
# 100 matches 100. Each row is divided by its own total, which is what "normalise
# rows" means.
#
# ### The version that does not raise
#
# The `ValueError` above is a gift. On a **square** matrix the same bug produces a
# perfectly valid array of completely wrong numbers.

# %%
square = rng.random((5, 5))
wrong = square / square.sum(axis=1)              # runs, silently
right = square / square.sum(axis=1, keepdims=True)

print(f"wrong.sum(axis=1) : {np.round(wrong.sum(axis=1), 3)}   ← not 1")
print(f"right.sum(axis=1) : {np.round(right.sum(axis=1), 3)}   ← correct")
print("\nWhat `wrong` computed: X[i, j] / rowsum[j] — "
      "each COLUMN divided by the\ncorresponding ROW's total. "
      "A quantity with no meaning at all.")

# %% [markdown]
# > **The rule:** if you are dividing an array by one of its own reductions, you
# > want `keepdims=True` essentially always. It documents which axis you meant
# > and it turns the square-matrix silent failure into a loud one.

# %% [markdown]
# ## Exercise 4.4 — Distance to centroids
#
# One k-means iteration. The `None` indices are what make it work: line the
# points up along a new axis 1 and the centroids along a new axis 0, and the
# subtraction produces every point-centroid pair.

# %%
X = rng.normal(size=(1000, 3))
centroids = rng.normal(size=(8, 3))

# (1000, 1, 3) - (1, 8, 3) → (1000, 8, 3), one difference vector per pair.
diff = X[:, None, :] - centroids[None, :, :]
distances = np.sqrt((diff ** 2).sum(axis=-1))
nearest = distances.argmin(axis=1)

print(f"X          : {X.shape}")
print(f"centroids  : {centroids.shape}")
print(f"diff       : {diff.shape}   ← the intermediate")
print(f"distances  : {distances.shape}")
print(f"nearest    : {nearest.shape}  e.g. {nearest[:10]}")
print(f"\ncluster sizes: {np.bincount(nearest, minlength=8)}")

# %% [markdown]
# ### The cost nobody mentions
#
# That `diff` tensor is `n × k × d`. Here it is 24 000 floats — nothing. At
# realistic scale it is the whole problem:

# %%
for n_points, k, d in [(1_000, 8, 3), (100_000, 50, 128), (1_000_000, 100, 128)]:
    tensor = n_points * k * d * 8
    matrix = n_points * k * 8
    print(f"n={n_points:>9,}  k={k:>3}  d={d:>3}  →  "
          f"diff tensor {tensor / 1e9:>7.1f} GB   vs   "
          f"distances {matrix / 1e9:>5.2f} GB")

# %% [markdown]
# The fix is the same algebraic expansion as Exercise 4.5:
#
# $$\|x - c\|^2 = \|x\|^2 - 2\,x \cdot c + \|c\|^2$$
#
# Each term is either a row-norm or a matrix product, so the `(n, k, d)`
# intermediate never exists.

# %%
def distances_expanded(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances without the (n, k, d) intermediate.

    Returns
    -------
    ndarray of shape (n_samples, n_centroids)
    """
    x_sq = (X ** 2).sum(axis=1)[:, None]          # (n, 1)
    c_sq = (centroids ** 2).sum(axis=1)[None, :]  # (1, k)
    cross = X @ centroids.T                        # (n, k)
    # Rounding can push near-zero distances slightly negative; clip before sqrt.
    return np.maximum(x_sq - 2 * cross + c_sq, 0.0)


expanded = np.sqrt(distances_expanded(X, centroids))
print(f"matches the broadcast version: {np.allclose(expanded, distances)}")
print(f"peak intermediate: {X.shape[0] * centroids.shape[0] * 8 / 1e3:.0f} KB "
      f"instead of {diff.nbytes / 1e3:.0f} KB")

# %% [markdown]
# The expanded form is what scikit-learn's `euclidean_distances` uses, and the
# trade-off it makes is worth knowing: it is **less numerically accurate**. When
# `x_sq` and `c_sq` are large and the true distance is small, you are subtracting
# two nearly equal big numbers — Exercise 3.5's cancellation, again. For
# clustering that is irrelevant; if you need exact small distances, keep the
# broadcast version and chunk over rows.

# %% [markdown]
# ## Exercise 4.5 — Avoid the intermediate
#
# Expand the square and the double sum separates:
#
# $$\sum_i \sum_j (a_i - b_j)^2
#  = \sum_i \sum_j \left( a_i^2 - 2 a_i b_j + b_j^2 \right)
#  = m \sum_i a_i^2 \;-\; 2 \left(\sum_i a_i\right)\!\left(\sum_j b_j\right)
#    \;+\; n \sum_j b_j^2$$
#
# where `n = len(a)` and `m = len(b)`. Three reductions and one product — no
# pairwise anything.

# %%
def pairwise_sq_diff_naive(a: np.ndarray, b: np.ndarray) -> float:
    """Direct version. Allocates an (n, m) intermediate."""
    return float(((a[:, None] - b[None, :]) ** 2).sum())


def pairwise_sq_diff_expanded(a: np.ndarray, b: np.ndarray) -> float:
    """Same quantity, O(n + m) memory."""
    n, m = a.size, b.size
    return float(m * (a @ a) - 2 * a.sum() * b.sum() + n * (b @ b))


small_a = rng.normal(size=200)
small_b = rng.normal(size=300)
naive = pairwise_sq_diff_naive(small_a, small_b)
expanded_result = pairwise_sq_diff_expanded(small_a, small_b)
print(f"naive    : {naive:.6f}")
print(f"expanded : {expanded_result:.6f}")
print(f"relative difference: {abs(naive - expanded_result) / abs(naive):.2e}")

# %%
big_a = rng.normal(size=50_000)
big_b = rng.normal(size=50_000)

intermediate_gb = big_a.size * big_b.size * 8 / 1e9
print(f"the naive intermediate would be {intermediate_gb:.1f} GB — not attempted\n")

start = time.perf_counter()
answer = pairwise_sq_diff_expanded(big_a, big_b)
elapsed = time.perf_counter() - start
print(f"expanded form: {answer:,.1f}")
print(f"time         : {elapsed * 1000:.2f} ms")
print(f"peak memory  : {2 * big_a.nbytes / 1e6:.1f} MB (just the inputs)")

# %% [markdown]
# 20 GB of RAM and minutes of compute, replaced by three dot products and a
# millisecond. This is the single highest-leverage NumPy skill there is: **when
# an expression builds an `(n, m)` or `(n, m, d)` intermediate, expand the
# algebra first.** Squared distances, Gram matrices, variances, and covariances
# all factor this way.

# %% [markdown]
# ---
# # Notebook 05 — Aggregations & Boolean Masking

# %% [markdown]
# ## Exercise 5.1 — Axis drill
#
# `axis=` names the dimension that **disappears**. For `a` of shape `(2, 3, 4)`:
#
# | Expression | Shape | Reasoning |
# | :--- | :-- | :--- |
# | `a.sum(axis=0)` | `(3, 4)` | Axis 0 (length 2) collapses. |
# | `a.sum(axis=1)` | `(2, 4)` | Axis 1 (length 3) collapses. |
# | `a.sum(axis=2)` | `(2, 3)` | Axis 2 (length 4) collapses. |
# | `a.sum(axis=(0, 2))` | `(3,)` | Both named axes collapse; only axis 1 survives. |
# | `a.mean(axis=1, keepdims=True)` | `(2, 1, 4)` | Axis 1 stays, with length 1. |
# | `a.max(axis=-1)` | `(2, 3)` | `-1` is the last axis, i.e. axis 2. |

# %%
a = np.arange(24).reshape(2, 3, 4)

drills = {
    "a.sum(axis=0)": a.sum(axis=0),
    "a.sum(axis=1)": a.sum(axis=1),
    "a.sum(axis=2)": a.sum(axis=2),
    "a.sum(axis=(0, 2))": a.sum(axis=(0, 2)),
    "a.mean(axis=1, keepdims=True)": a.mean(axis=1, keepdims=True),
    "a.max(axis=-1)": a.max(axis=-1),
}
for label, result in drills.items():
    print(f"{label:<32} {str(result.shape):<12}")

# %% [markdown]
# A cross-check that makes the rule concrete: **the result shape is the input
# shape with the named entries deleted.** `(2, 3, 4)` minus entry 0 is `(3, 4)`;
# minus entries 0 and 2 is `(3,)`. With `keepdims=True` they become `1` instead of
# vanishing.

# %%
for axis in (0, 1, 2, (0, 2)):
    kept = tuple(d for i, d in enumerate(a.shape)
                 if i not in (axis if isinstance(axis, tuple) else (axis,)))
    print(f"axis={str(axis):<7} → predicted {str(kept):<10} "
          f"actual {a.sum(axis=axis).shape}")

# %% [markdown]
# ## Exercise 5.2 — The keepdims trap

# %%
X = rng.random((5, 5))

without = X / X.sum(axis=1)
withkd = X / X.sum(axis=1, keepdims=True)

print(f"X / X.sum(axis=1)                row sums: "
      f"{np.round(without.sum(axis=1), 3)}")
print(f"X / X.sum(axis=1, keepdims=True) row sums: "
      f"{np.round(withkd.sum(axis=1), 3)}")

# %% [markdown]
# ### Precisely what the first one computed
#
# `X.sum(axis=1)` has shape `(5,)`. Broadcasting pads it to `(1, 5)` and stretches
# it down the rows — so it is treated as **one value per column**, when it holds
# one value per row.
#
# The result is `out[i, j] = X[i, j] / rowsum[j]`: element `(i, j)` divided by the
# total of row `j`, a row that has nothing to do with element `(i, j)` unless
# `i == j`.

# %%
row_sums = X.sum(axis=1)
manual = np.empty_like(X)
for i in range(5):
    for j in range(5):
        manual[i, j] = X[i, j] / row_sums[j]   # note: rowsum[j], not rowsum[i]

print(f"the loop reproduces `without` exactly: {np.allclose(manual, without)}")
print(f"\nX / X.sum(axis=1) is identical to X / X.sum(axis=1)[None, :]: "
      f"{np.allclose(without, X / row_sums[None, :])}")
print(f"what we wanted was      X / X.sum(axis=1)[:, None]        : "
      f"{np.allclose(withkd, X / row_sums[:, None])}")

# %% [markdown]
# It only runs at all because the matrix is square. Make it `(5, 4)` and the same
# code raises — which is why this bug survives so much longer in code that
# happens to deal with square matrices (correlation matrices, adjacency matrices,
# confusion matrices, attention scores).

# %% [markdown]
# ## Exercise 5.3 — Masking practice

# %%
data = rng.normal(100, 20, 10_000)

mean, std = data.mean(), data.std()
outliers = np.abs(data - mean) > 2 * std

print(f"mean {mean:.2f}, std {std:.2f}")
print(f"1. beyond ±2σ : {outliers.sum():,} values "
      f"({outliers.mean():.2%})   [theory for a normal: 4.55%]")

# %%
# 2. Replace, without a loop. `np.where` picks element-wise between two arrays.
#    `mean` was computed BEFORE the replacement — using the post-replacement
#    mean would be circular.
cleaned = np.where(outliers, mean, data)

print(f"2. replaced {outliers.sum():,} values with the mean")
print(f"   std before {data.std():.2f} → after {cleaned.std():.2f}")
print(f"   mean before {data.mean():.4f} → after {cleaned.mean():.4f} (unchanged)")

# %%
# 3. Share in [80, 120]. `&` is element-wise AND; the parentheses are required
#    because `&` binds tighter than `>=` in Python.
in_range = (data >= 80) & (data <= 120)
print(f"3. within [80, 120]: {in_range.mean():.2%} "
      f"({in_range.sum():,} of {data.size:,})")

# %% [markdown]
# ### Two notes on the replacement
#
# **Winsorising beats replacing with the mean.** Collapsing outliers onto the mean
# invents a spike of identical values at the centre of the distribution, which
# distorts the variance and every quantile. Clipping to a percentile keeps the
# ordering intact:

# %%
lower, upper = np.percentile(data, [1, 99])
winsorised = np.clip(data, lower, upper)

comparison = pd.DataFrame({
    "original": [data.mean(), data.std(), data.min(), data.max()],
    "mean-replaced": [cleaned.mean(), cleaned.std(), cleaned.min(), cleaned.max()],
    "winsorised": [winsorised.mean(), winsorised.std(),
                   winsorised.min(), winsorised.max()],
}, index=["mean", "std", "min", "max"])
print(comparison.round(3).to_string())

# %% [markdown]
# **And a warning about the ±2σ rule itself.** The standard deviation is computed
# *from the data that includes the outliers*, so a few extreme values inflate `σ`
# and hide themselves behind the wider threshold. Robust alternatives —
# median ± k·MAD, or an IQR rule — do not have that problem.

# %%
median = np.median(data)
mad = np.median(np.abs(data - median))
robust_sigma = 1.4826 * mad   # scale factor makes MAD comparable to σ for normals
robust_outliers = np.abs(data - median) > 2 * robust_sigma

print(f"classic  ±2σ  (σ = {std:.2f})          : {outliers.sum():,} flagged")
print(f"robust   ±2·MAD (σ̂ = {robust_sigma:.2f}) : {robust_outliers.sum():,} flagged")

# %% [markdown]
# ## Exercise 5.4 — Group-by from scratch
#
# The trick is `np.unique(..., return_inverse=True)`. It gives you the sorted
# unique keys *and* an integer code per row — and once every key is an integer in
# `0..n_groups-1`, `np.bincount` does the aggregation in C.

# %%
def group_stats(keys: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    """Per-key count, sum, mean and max, in NumPy only.

    Parameters
    ----------
    keys : ndarray of shape (n,)
        Group labels. Any dtype `np.unique` accepts, including strings.
    values : ndarray of shape (n,)
        Values to aggregate.

    Returns
    -------
    DataFrame indexed by the sorted unique keys.
    """
    keys = np.asarray(keys)
    values = np.asarray(values, dtype=np.float64)
    if keys.shape != values.shape:
        raise ValueError(f"shape mismatch: {keys.shape} vs {values.shape}")

    unique, codes = np.unique(keys, return_inverse=True)
    n_groups = unique.size

    counts = np.bincount(codes, minlength=n_groups)
    sums = np.bincount(codes, weights=values, minlength=n_groups)

    # `bincount` cannot do max, so use the ufunc's unbuffered `.at`: for each i,
    # maxima[codes[i]] = max(maxima[codes[i]], values[i]).
    maxima = np.full(n_groups, -np.inf)
    np.maximum.at(maxima, codes, values)

    return pd.DataFrame(
        {"count": counts, "sum": sums, "mean": sums / counts, "max": maxima},
        index=pd.Index(unique, name="key"),
    )


# %%
n = 100_000
group_keys = rng.choice(np.array(["alpha", "beta", "gamma", "delta"]), size=n,
                        p=[0.4, 0.3, 0.2, 0.1])
group_values = rng.gamma(shape=2.0, scale=50.0, size=n)

mine = group_stats(group_keys, group_values)
print(mine.round(3).to_string())

# %%
theirs = (
    pd.DataFrame({"key": group_keys, "value": group_values})
    .groupby("key")["value"]
    .agg(["count", "sum", "mean", "max"])
)
print(f"matches pandas: {np.allclose(mine.to_numpy(), theirs.to_numpy())}")

t_numpy = bench(group_stats, group_keys, group_values)
frame = pd.DataFrame({"key": group_keys, "value": group_values})
t_pandas = bench(lambda: frame.groupby("key")["value"].agg(["count", "sum",
                                                            "mean", "max"]))
print(f"\nNumPy  : {t_numpy * 1000:6.1f} ms")
print(f"pandas : {t_pandas * 1000:6.1f} ms")

# %% [markdown]
# ### Why `np.maximum.at` and not a loop
#
# `maxima[codes] = np.maximum(maxima[codes], values)` looks equivalent and is
# **wrong**. Fancy-index assignment is buffered: NumPy reads all the old values,
# computes, and writes back — so when a code appears twice, the second write
# overwrites the first instead of combining with it. `ufunc.at` is the unbuffered
# version that exists precisely for this.

# %%
demo_codes = np.array([0, 0, 0, 1])
demo_values = np.array([5.0, 9.0, 2.0, 7.0])

buffered = np.full(2, -np.inf)
buffered[demo_codes] = np.maximum(buffered[demo_codes], demo_values)

unbuffered = np.full(2, -np.inf)
np.maximum.at(unbuffered, demo_codes, demo_values)

print(f"buffered   (wrong) : {buffered}   ← group 0 should be 9, not 2")
print(f"ufunc.at   (right) : {unbuffered}")

# %% [markdown]
# `ufunc.at` is correct but slow — it cannot vectorise. For a max specifically,
# the fast alternative is to sort by code and use `np.maximum.reduceat`; use it
# when this shows up in a profile, not before.

# %% [markdown]
# ## Exercise 5.5 — Top-k per row
#
# `argsort` on `(1000, 50)` is `O(n · m log m)` and returns 50 indices per user
# when you wanted 5. `argpartition` is `O(n · m)` — it does just enough work to
# get the top `k` into the first `k` slots, in no particular order. Sorting only
# those five costs nothing.

# %%
n_users, n_items, k = 1000, 50, 5
scores = rng.random((n_users, n_items))


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's k largest values, best first.

    Returns
    -------
    ndarray of shape (n_rows, k)
    """
    # Negate so "k largest" becomes "k smallest", which is what kth refers to.
    partitioned = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]

    rows = np.arange(scores.shape[0])[:, None]           # (n, 1), broadcasts
    candidate_scores = scores[rows, partitioned]         # (n, k)
    order = np.argsort(-candidate_scores, axis=1)        # sort just the k
    return partitioned[rows, order]


top_items = top_k(scores, k)
print(f"top_items: {top_items.shape}")
print(f"user 0 → items {top_items[0]}")
print(f"their scores  → {np.round(scores[0, top_items[0]], 4)}  (descending ✔)")

# %%
reference = np.argsort(-scores, axis=1)[:, :k]
print(f"matches a full argsort: {np.array_equal(top_items, reference)}")

t_partition = bench(top_k, scores, k, repeat=20)
t_full_sort = bench(lambda: np.argsort(-scores, axis=1)[:, :k], repeat=20)
print(f"\nargpartition + partial sort : {t_partition * 1e6:7.0f} µs")
print(f"full argsort                : {t_full_sort * 1e6:7.0f} µs")
print(f"speed-up                    : {t_full_sort / t_partition:7.1f}×")

# %% [markdown]
# ### Where the gap actually matters
#
# At 50 items the saving is small. A recommender serves from a catalogue of
# hundreds of thousands, and that is where the complexity difference bites:

# %%
for catalogue in (50, 1_000, 100_000):
    wide = rng.random((200, catalogue))
    fast = bench(top_k, wide, 5, repeat=3)
    slow = bench(lambda w=wide: np.argsort(-w, axis=1)[:, :5], repeat=3)
    print(f"{catalogue:>7,} items  →  argpartition {fast * 1000:6.1f} ms   "
          f"argsort {slow * 1000:6.1f} ms   {slow / fast:.1f}×")

# %% [markdown]
# > **One caveat for production.** `argpartition` gives no guarantee about ties,
# > so two items with identical scores may come back in either order — and the
# > order can change between NumPy versions. If your ranking must be
# > deterministic, break ties explicitly on a stable secondary key (item ID) via
# > `np.lexsort`.

# %% [markdown]
# ---
# # Notebook 06 — Linear Algebra

# %% [markdown]
# ## Exercise 6.1 — Shapes
#
# `@` has one rule for 2-D — inner dimensions must match — plus two special cases
# for 1-D operands and a batching rule for anything higher.
#
# | Expression | Result | Rule |
# | :--- | :-- | :--- |
# | `(3, 4) @ (4, 5)` | `(3, 5)` | Inner 4s match; outer dimensions survive. |
# | `(3, 4) @ (3, 4)` | **error** | Inner 4 vs 3. |
# | `(4,) @ (4,)` | `()` | Both 1-D → inner product, a scalar. |
# | `(3, 4) @ (4,)` | `(3,)` | 1-D on the right is a column; the axis is dropped after. |
# | `(10, 3, 4) @ (4, 2)` | `(10, 3, 2)` | Leading axes are batch dimensions and broadcast. |
# | `(3,) @ (3, 4)` | `(4,)` | 1-D on the left is a row; the axis is dropped after. |

# %%
shape_cases = [
    ((3, 4), (4, 5)),
    ((3, 4), (3, 4)),
    ((4,), (4,)),
    ((3, 4), (4,)),
    ((10, 3, 4), (4, 2)),
    ((3,), (3, 4)),
]

for left_shape, right_shape in shape_cases:
    left, right = np.ones(left_shape), np.ones(right_shape)
    try:
        result = str(np.shape(left @ right))
    except ValueError:
        result = "error"
    print(f"{str(left_shape):<12} @ {str(right_shape):<10} → {result}")

# %% [markdown]
# The 1-D cases are the ones to be careful with. NumPy promotes a 1-D operand to
# a row or a column depending on which side it is on, does the multiply, then
# **removes the axis it added**. Convenient, and the reason `X @ beta` returns
# `(n,)` rather than `(n, 1)` — which then broadcasts against a `(n, 1)` target
# into Exercise 4.2's bug.

# %% [markdown]
# ## Exercise 6.2 — Cosine similarity matrix
#
# Cosine similarity is the dot product of unit-normalised vectors. Normalise the
# rows once, and a single matrix product produces all million pairs.

# %%
E = rng.normal(size=(1000, 128))


def cosine_similarity_matrix(E: np.ndarray) -> np.ndarray:
    """Full pairwise cosine similarity for row-vectors of `E`.

    Returns
    -------
    ndarray of shape (n, n), symmetric, with 1.0 on the diagonal.
    """
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    # A zero-length vector has no direction; guard rather than divide by 0.
    normalised = E / np.where(norms == 0, 1.0, norms)
    return normalised @ normalised.T


similarity = cosine_similarity_matrix(E)

print(f"similarity : {similarity.shape}")
print(f"symmetric  : {np.allclose(similarity, similarity.T)}")
print(f"diagonal ≈ 1: {np.allclose(np.diag(similarity), 1.0)}")
print(f"range      : [{similarity.min():.3f}, {similarity.max():.3f}]")

# %%
print(f"matches scikit-learn: "
      f"{np.allclose(similarity, cosine_similarity(E))}")

t_mine = bench(cosine_similarity_matrix, E, repeat=5)
t_sklearn = bench(cosine_similarity, E, repeat=5)
print(f"\nours        : {t_mine * 1000:6.1f} ms")
print(f"scikit-learn: {t_sklearn * 1000:6.1f} ms")

# %% [markdown]
# It is competitive with scikit-learn because it is the *same* algorithm calling
# the *same* BLAS. `E @ E.T` on `(1000, 128)` is 128 million multiply-adds, and a
# tuned `dgemm` runs that in a few milliseconds using every core.
#
# **The memory to watch is the output, not the input.** `E` is 1 MB; the
# similarity matrix is 8 MB. At 100 000 documents `E` is 100 MB and the matrix is
# **80 GB** — you must chunk the rows, or use an approximate nearest-neighbour
# index and never form the matrix at all.

# %%
for n_docs in (1_000, 10_000, 100_000):
    print(f"{n_docs:>7,} docs → embeddings {n_docs * 128 * 8 / 1e6:>7.1f} MB, "
          f"similarity matrix {n_docs ** 2 * 8 / 1e9:>7.1f} GB")

# %% [markdown]
# ## Exercise 6.3 — Implement least squares

# %%
def fit_linear(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = True,
    alpha: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Fit ordinary or ridge least squares.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    fit_intercept : bool, default True
        Whether to fit a constant term.
    alpha : float, default 0.0
        L2 penalty. ``0.0`` gives ordinary least squares via `lstsq`.

    Returns
    -------
    coef : ndarray of shape (n_features,)
    intercept : float

    Notes
    -----
    The intercept is obtained by **centring**, not by appending a column of
    ones. That matters for ridge: scikit-learn does not penalise the intercept,
    and an appended ones-column would be penalised like any other coefficient.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if fit_intercept:
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
    else:
        x_mean, y_mean = np.zeros(X.shape[1]), 0.0
        Xc, yc = X, y

    if alpha == 0.0:
        # `lstsq` uses an SVD: it is stable, and it returns the minimum-norm
        # solution rather than raising when the columns are rank-deficient.
        coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    else:
        gram = Xc.T @ Xc + alpha * np.eye(X.shape[1])
        # `solve`, never `inv` — same answer, better conditioned, twice as fast.
        coef = np.linalg.solve(gram, Xc.T @ yc)

    return coef, float(y_mean - x_mean @ coef)


# %%
n_samples, n_features = 500, 8
X_data = rng.normal(size=(n_samples, n_features))
true_coef = rng.normal(size=n_features) * 3
y_data = X_data @ true_coef + 7.5 + rng.normal(scale=0.5, size=n_samples)

coef, intercept = fit_linear(X_data, y_data)
reference_ols = LinearRegression().fit(X_data, y_data)

print("OLS")
print(f"  coefficients match : "
      f"{np.allclose(coef, reference_ols.coef_)}")
print(f"  intercept match    : "
      f"{np.isclose(intercept, reference_ols.intercept_)}")
print(f"  ours      : {np.round(coef[:4], 4)} ... intercept {intercept:.4f}")
print(f"  sklearn   : {np.round(reference_ols.coef_[:4], 4)} ... "
      f"intercept {reference_ols.intercept_:.4f}")

# %%
print("Ridge")
for alpha in (0.1, 1.0, 10.0, 100.0):
    ridge_coef, ridge_intercept = fit_linear(X_data, y_data, alpha=alpha)
    reference = Ridge(alpha=alpha).fit(X_data, y_data)
    print(f"  alpha={alpha:>6.1f}  coef match {np.allclose(ridge_coef, reference.coef_)}"
          f"   intercept match "
          f"{np.isclose(ridge_intercept, reference.intercept_)}"
          f"   ‖β‖ = {np.linalg.norm(ridge_coef):.3f}")

# %% [markdown]
# `‖β‖` shrinking as `alpha` grows is ridge doing exactly what it is for. The two
# implementation details that make this match scikit-learn rather than
# *approximately* match it:
#
# 1. **Centre rather than append a ones-column.** Appending would put the
#    intercept inside the penalty, so every coefficient would come out slightly
#    wrong — a bug that produces plausible numbers and no error.
# 2. **`np.linalg.solve(A, b)`, never `np.linalg.inv(A) @ b`.** Same answer in
#    exact arithmetic; in floating point `solve` is more accurate and about
#    twice as fast, because forming an explicit inverse does strictly more work
#    and accumulates more error.

# %% [markdown]
# ## Exercise 6.4 — Condition number and collinearity

# %%
n_samples = 300
base = rng.normal(size=n_samples)

# Two nearly identical columns, plus a third on a wildly different scale.
X_ill = np.column_stack([
    base,
    base + rng.normal(scale=0.014, size=n_samples),   # ~0.9999 correlated
    rng.normal(loc=50_000, scale=1_000, size=n_samples),
])
y_ill = X_ill @ np.array([2.0, -1.0, 0.001]) + rng.normal(scale=0.1, size=n_samples)

corr = np.corrcoef(X_ill[:, 0], X_ill[:, 1])[0, 1]
gram = X_ill.T @ X_ill

print(f"corr(col0, col1)   : {corr:.6f}")
print(f"cond(X)            : {np.linalg.cond(X_ill):.3e}")
print(f"cond(XᵀX)          : {np.linalg.cond(gram):.3e}   ← the square of it")

# %%
coef_normal = np.linalg.solve(gram, X_ill.T @ y_ill)
coef_lstsq, *_ = np.linalg.lstsq(X_ill, y_ill, rcond=None)

print(f"{'':<22}{'col 0':>12}{'col 1':>12}{'col 2':>12}")
print(f"{'true':<22}{2.0:>12.4f}{-1.0:>12.4f}{0.001:>12.6f}")
print(f"{'normal equations':<22}" + "".join(f"{c:>12.4f}" for c in coef_normal))
print(f"{'lstsq (SVD)':<22}" + "".join(f"{c:>12.4f}" for c in coef_lstsq))
print(f"\nmax |difference| between the two methods: "
      f"{np.abs(coef_normal - coef_lstsq).max():.2e}")

# %%
# Standardise: zero mean, unit variance per column.
X_std = (X_ill - X_ill.mean(axis=0)) / X_ill.std(axis=0)

print(f"cond(XᵀX)  raw          : {np.linalg.cond(X_ill.T @ X_ill):.3e}")
print(f"cond(XᵀX)  standardised : {np.linalg.cond(X_std.T @ X_std):.3e}")
print(f"improvement             : "
      f"{np.linalg.cond(X_ill.T @ X_ill) / np.linalg.cond(X_std.T @ X_std):.1e}×")

# %% [markdown]
# ### Reading the numbers honestly
#
# Standardising improves the condition number by orders of magnitude — but it has
# **not fixed the collinearity**, and the remaining condition number (still large)
# is the collinearity talking. Two distinct problems were tangled together:
#
# | Problem | Cause | Fixed by standardising? |
# | :--- | :--- | :-- |
# | Column 2 is 50 000× the scale of the others | units | **Yes** |
# | Columns 0 and 1 are 0.9999 correlated | the data | **No** |
#
# Scaling is a change of units; it cannot create information that is not there.
# Two nearly identical columns mean the data genuinely does not identify their
# individual coefficients — the *sum* is well determined, the split is not. The
# real options are to drop one column, combine them, or accept a biased-but-stable
# estimate from ridge.
#
# A practical reading of `cond(XᵀX)`: below `1e3` fine, `1e3`–`1e8` treat
# coefficients with suspicion, above `1e10` individual coefficients are noise.
# You lose roughly `log10(cond)` of your 16 significant digits.

# %%
print(f"{'method':<28}{'col 0':>10}{'col 1':>10}{'col 0 + col 1':>16}")
print("-" * 64)
print(f"{'true':<28}{2.0:>10.3f}{-1.0:>10.3f}{1.0:>16.3f}")
print(f"{'OLS (lstsq)':<28}{coef_lstsq[0]:>10.3f}{coef_lstsq[1]:>10.3f}"
      f"{coef_lstsq[0] + coef_lstsq[1]:>16.3f}")
for alpha in (1.0, 100.0):
    ridge_coef, _ = fit_linear(X_std, y_ill, alpha=alpha)
    print(f"{f'ridge alpha={alpha:g} (std.)':<28}{ridge_coef[0]:>10.3f}"
          f"{ridge_coef[1]:>10.3f}{ridge_coef[0] + ridge_coef[1]:>16.3f}")

# %% [markdown]
# ## Exercise 6.5 — Image compression with SVD
#
# Every matrix factors as `A = U Σ Vᵀ` with the singular values in `Σ` in
# descending order. Keeping the largest `r` of them gives the **best possible**
# rank-`r` approximation in the least-squares sense (Eckart–Young). Storage drops
# from `m·n` numbers to `r·(m + n + 1)`.

# %%
size = 200
grid_y, grid_x = np.mgrid[0:size, 0:size] / size

image = (
    np.sin(6 * np.pi * grid_x) * np.cos(4 * np.pi * grid_y)
    + 0.6 * np.exp(-((grid_x - 0.3) ** 2 + (grid_y - 0.7) ** 2) / 0.02)
    + 0.4 * (grid_x + grid_y)
)
image += rng.normal(scale=0.03, size=image.shape)

U, singular_values, Vt = np.linalg.svd(image, full_matrices=False)
print(f"image  : {image.shape}, {image.nbytes / 1024:.0f} KiB")
print(f"U {U.shape}   Σ {singular_values.shape}   Vᵀ {Vt.shape}")
print(f"top 10 singular values: {np.round(singular_values[:10], 2)}")

# %%
def reconstruct(U, s, Vt, rank: int) -> np.ndarray:
    """Rank-`rank` truncated-SVD reconstruction."""
    return (U[:, :rank] * s[:rank]) @ Vt[:rank]


ranks = [5, 20, 50]
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

axes[0].imshow(image, cmap="viridis")
axes[0].set_title(f"original\nrank {np.linalg.matrix_rank(image)} · "
                  f"{image.size:,} numbers", fontsize=10)

for ax, rank in zip(axes[1:], ranks, strict=True):
    approx = reconstruct(U, singular_values, Vt, rank)
    stored = rank * (image.shape[0] + image.shape[1] + 1)
    energy = (singular_values[:rank] ** 2).sum() / (singular_values ** 2).sum()
    ax.imshow(approx, cmap="viridis")
    ax.set_title(f"rank {rank}\n{stored:,} numbers "
                 f"({stored / image.size:.0%}) · {energy:.1%} energy", fontsize=10)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Truncated SVD: the first few singular values carry the picture",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %%
print(f"{'rank':>6}{'numbers stored':>18}{'compression':>14}"
      f"{'energy kept':>14}{'RMSE':>10}")
print("-" * 62)
for rank in [1, 5, 20, 50, 100, 200]:
    approx = reconstruct(U, singular_values, Vt, rank)
    stored = rank * (image.shape[0] + image.shape[1] + 1)
    energy = (singular_values[:rank] ** 2).sum() / (singular_values ** 2).sum()
    rmse = np.sqrt(((image - approx) ** 2).mean())
    print(f"{rank:>6}{stored:>18,}{stored / image.size:>13.0%}"
          f"{energy:>14.2%}{rmse:>10.4f}")

# %% [markdown]
# Rank 20 keeps well over 99% of the energy in about a fifth of the storage. The
# residual is mostly the noise we added — the structured part of the image
# genuinely is low-rank.
#
# **The reason this matters far beyond images:** truncated SVD *is* PCA. Same
# factorisation, same truncation, different vocabulary — singular values become
# explained variance, and `Vᵀ` rows become principal components. Latent semantic
# analysis, recommender matrix factorisation, and noise reduction are all this
# same operation. You will meet it again in Module 15.

# %%
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.semilogy(singular_values, color=COLORS["primary"], linewidth=2)
for rank, color in zip(ranks, [COLORS["accent"], COLORS["teal"], COLORS["purple"]],
                       strict=True):
    ax.axvline(rank, color=color, linestyle="--", linewidth=1.2,
               label=f"rank {rank}")
ax.set_xlabel("singular value index")
ax.set_ylabel("singular value (log scale)")
ax.set_title("The spectrum decays fast — which is why truncation works", loc="left")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# # Notebook 07 — Randomness & Simulation

# %% [markdown]
# ## Exercise 7.1 — Break and fix reproducibility
#
# The legacy API keeps **one global stream**. Anything that draws from it moves
# the shared cursor — including code you did not write and cannot see.

# %%
def helper_that_looks_harmless() -> float:
    """A utility that happens to use randomness internally."""
    return float(np.random.random())   # legacy global API


def draw_legacy(seed: int, call_helper: bool) -> np.ndarray:
    np.random.seed(seed)
    if call_helper:
        helper_that_looks_harmless()   # consumes from the same global stream
    return np.random.normal(size=3)


without_helper = draw_legacy(42, call_helper=False)
with_helper = draw_legacy(42, call_helper=True)

print(f"same seed, no helper : {np.round(without_helper, 4)}")
print(f"same seed, helper    : {np.round(with_helper, 4)}")
print(f"identical? {np.array_equal(without_helper, with_helper)}   ← reproducibility lost")

# %% [markdown]
# Nothing about the seeding changed. A function that has nothing to do with the
# analysis silently altered its results — and in a real pipeline that function is
# three imports away, inside a library, added in someone else's pull request.

# %%
def draw_modern(rng: np.random.Generator, call_helper: bool) -> np.ndarray:
    """Randomness is a parameter, not an ambient global."""
    if call_helper:
        helper_that_looks_harmless()   # still touches the global stream...
    return rng.normal(size=3)          # ...which this generator does not use


clean = draw_modern(np.random.default_rng(42), call_helper=False)
polluted = draw_modern(np.random.default_rng(42), call_helper=True)

print(f"same seed, no helper : {np.round(clean, 4)}")
print(f"same seed, helper    : {np.round(polluted, 4)}")
print(f"identical? {np.array_equal(clean, polluted)}   ← cannot be broken")

# %% [markdown]
# ### And for parallel work, `spawn`
#
# Reusing one seed across workers gives every worker the *same* numbers — a bug
# that looks like a suspiciously small variance. `SeedSequence.spawn` derives
# independent streams that are still fully reproducible from the parent seed.

# %%
def worker_draws(parent_seed: int, n_workers: int) -> list[np.ndarray]:
    """Give each worker its own independent stream, derived from one seed."""
    children = np.random.SeedSequence(parent_seed).spawn(n_workers)
    return [np.random.default_rng(child).normal(size=3) for child in children]


first_run = worker_draws(42, 4)
second_run = worker_draws(42, 4)

print("four independent worker streams from the single parent seed 42:")
for i, draws in enumerate(first_run):
    print(f"  worker {i}: {np.round(draws, 4)}")

print(f"\nworkers differ from each other: "
      f"{not np.array_equal(first_run[0], first_run[1])}")
print(f"whole set reproduces exactly  : "
      f"{all(np.array_equal(a, b) for a, b in zip(first_run, second_run, strict=True))}")

# %% [markdown]
# ## Exercise 7.2 — Choose the distribution
#
# | # | Quantity | Distribution | Why |
# | :-- | :--- | :--- | :--- |
# | 1 | Support tickets per hour | **Poisson(λ)** | Counts of independent events in a fixed window; the defining use case. |
# | 2 | Time between two tickets | **Exponential(λ)** | The waiting time between Poisson events — memoryless by construction. |
# | 3 | Whether each of 500 emails is opened | **Bernoulli(p)** per email, **Binomial(500, p)** for the total | A binary outcome repeated independently. |
# | 4 | Annual revenue per customer | **Lognormal** (or Gamma) | Positive, right-skewed, multiplicative: many small customers, a few enormous ones. |
# | 5 | Sensor measurement error | **Normal(0, σ)** | A sum of many small independent perturbations — the CLT's home turf. |

# %%
fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
palette = [COLORS["primary"], COLORS["accent"], COLORS["teal"],
           COLORS["purple"], COLORS["danger"]]

samples = [
    ("Poisson(λ=12)\ntickets per hour", rng.poisson(12, 10_000), palette[0]),
    ("Exponential(1/12 h)\ngap between tickets", rng.exponential(1 / 12, 10_000), palette[1]),
    ("Binomial(500, 0.22)\nopens per campaign", rng.binomial(500, 0.22, 10_000), palette[2]),
    ("Lognormal(μ=5, σ=1.1)\nrevenue per customer", rng.lognormal(5, 1.1, 10_000), palette[3]),
    ("Normal(0, 0.4)\nsensor error", rng.normal(0, 0.4, 10_000), palette[4]),
]

for ax, (title, data, color) in zip(axes, samples, strict=True):
    ax.hist(data, bins=45, color=color, alpha=0.85, edgecolor="none")
    ax.set_title(title, fontsize=9)
    ax.set_yticks([])

fig.suptitle("Five questions, five distributions", fontsize=13, y=1.06)
plt.tight_layout()
plt.show()

# %% [markdown]
# The pair worth internalising is **1 and 2**: they describe the same process from
# two angles. If arrivals are Poisson with rate λ, the gaps between them are
# Exponential with mean 1/λ. Getting this backwards — modelling *durations* as
# Normal because they are "roughly symmetric" — produces negative waiting times
# and confidence intervals that cover impossible values.
#
# And **4**: revenue is almost never Normal. Testing a mean difference in revenue
# with a t-test on raw values is testing the wrong statistic; work on logs, use a
# rank test, or bootstrap. Module 08 goes into this properly.

# %%
gaps = rng.exponential(1 / 12, 100_000)
print(f"Poisson(12) mean : {rng.poisson(12, 100_000).mean():.3f}   (theory 12)")
print(f"gap mean         : {gaps.mean():.4f} h = {gaps.mean() * 60:.2f} min "
      f"(theory {60 / 12:.2f} min)")
print(f"revenue: mean {np.exp(5 + 1.1 ** 2 / 2):.0f} vs median {np.exp(5):.0f} "
      f"— the mean sits well above the typical customer")

# %% [markdown]
# ## Exercise 7.3 — The birthday problem

# %%
def birthday_simulation(k: int, trials: int, rng: np.random.Generator) -> float:
    """Estimated P(at least two of k people share a birthday)."""
    # (trials, k) birthdays; sort each row and look for equal neighbours.
    birthdays = rng.integers(0, 365, size=(trials, k))
    birthdays.sort(axis=1)
    has_match = (np.diff(birthdays, axis=1) == 0).any(axis=1)
    return float(has_match.mean())


def birthday_exact(k: int) -> float:
    """1 - (365/365)(364/365)...((365-k+1)/365), computed in log space."""
    if k > 365:
        return 1.0
    return 1.0 - float(np.exp(np.log(365 - np.arange(k)).sum() - k * np.log(365)))


TRIALS = 20_000
sim_rng = np.random.default_rng(seed=7)
group_sizes = np.arange(2, 61)

simulated = np.array([birthday_simulation(int(k), TRIALS, sim_rng)
                      for k in group_sizes])
exact = np.array([birthday_exact(int(k)) for k in group_sizes])

first_over_half_sim = int(group_sizes[simulated > 0.5][0])
first_over_half_exact = int(group_sizes[exact > 0.5][0])

print(f"smallest k with P > 50%  — simulated: {first_over_half_sim}, "
      f"exact: {first_over_half_exact}")
print(f"P(23) simulated {simulated[group_sizes == 23][0]:.4f}  "
      f"exact {exact[group_sizes == 23][0]:.4f}")
print(f"max |simulated - exact| over all k: {np.abs(simulated - exact).max():.4f}")

# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(group_sizes, exact, color=COLORS["primary"], linewidth=2,
        label="exact formula")
ax.scatter(group_sizes, simulated, color=COLORS["accent"], s=18, zorder=3,
           label=f"simulation ({TRIALS:,} trials)")
ax.axhline(0.5, color=COLORS["muted"], linestyle="--", linewidth=1)
ax.axvline(23, color=COLORS["teal"], linestyle=":", linewidth=1.5)
ax.annotate("k = 23 → 50.7%", (23, 0.5), xytext=(14, -34),
            textcoords="offset points", fontsize=10, color=COLORS["teal"],
            arrowprops={"arrowstyle": "->", "color": COLORS["teal"]})
ax.set_xlabel("people in the group (k)")
ax.set_ylabel("P(at least one shared birthday)")
ax.set_title("The birthday problem: simulation against the exact answer",
             loc="left")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **23 people.** The answer feels wrong because intuition silently substitutes a
# different question — "does someone share *my* birthday?", which needs 253
# people. The real question is about **pairs**, and 23 people make
# `23 × 22 / 2 = 253` of them.
#
# The simulation matching the exact formula to ~0.005 is the point of the
# exercise: when a closed form exists, simulation is a way to *check* it. When it
# does not, simulation is often the only route — and Exercise 7.5 is that case.

# %% [markdown]
# ## Exercise 7.4 — Bootstrap a difference

# %%
boot_rng = np.random.default_rng(seed=2024)

# Two groups of customer spend. Right-skewed, as spend always is.
spend_a = boot_rng.lognormal(mean=3.9, sigma=0.7, size=240)
spend_b = boot_rng.lognormal(mean=4.05, sigma=0.7, size=260)

observed = np.median(spend_b) - np.median(spend_a)
print(f"group A: n={spend_a.size}, median {np.median(spend_a):.2f}, "
      f"mean {spend_a.mean():.2f}")
print(f"group B: n={spend_b.size}, median {np.median(spend_b):.2f}, "
      f"mean {spend_b.mean():.2f}")
print(f"observed difference in medians: {observed:+.2f}")

# %%
def bootstrap_median_difference(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap distribution of median(b) - median(a).

    Resamples each group independently with replacement, `n_boot` times,
    in a single vectorised pass.
    """
    idx_a = rng.integers(0, a.size, size=(n_boot, a.size))
    idx_b = rng.integers(0, b.size, size=(n_boot, b.size))
    return np.median(b[idx_b], axis=1) - np.median(a[idx_a], axis=1)


N_BOOT = 10_000
differences = bootstrap_median_difference(spend_a, spend_b, N_BOOT, boot_rng)
ci_low, ci_high = np.percentile(differences, [2.5, 97.5])

print(f"bootstrap distribution: {N_BOOT:,} resamples")
print(f"95% CI for the difference in medians: [{ci_low:+.2f}, {ci_high:+.2f}]")
print(f"contains zero? {ci_low <= 0 <= ci_high}")
print(f"P(difference > 0) under the bootstrap: {(differences > 0).mean():.3f}")

# %%
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.hist(differences, bins=60, color=COLORS["primary"], alpha=0.8,
        edgecolor="none")
ax.axvline(0, color=COLORS["danger"], linewidth=2, label="no difference")
ax.axvline(observed, color=COLORS["accent"], linewidth=2,
           label=f"observed {observed:+.2f}")
ax.axvspan(ci_low, ci_high, color=COLORS["teal"], alpha=0.18,
           label=f"95% CI [{ci_low:+.2f}, {ci_high:+.2f}]")
ax.set_xlabel("median(B) − median(A)")
ax.set_ylabel("bootstrap resamples")
ax.set_title("Bootstrap CI for a difference in medians", loc="left")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### What the interval does and does not let you say
#
# **If the interval excludes zero**, you may say: the data are inconsistent with
# the two population medians being equal, at the 5% level. The plausible range
# for the difference is the interval.
#
# **You may not say:**
#
# - *"There is a 95% probability the true difference is in this interval."* The
#   true difference is a fixed number, not a random one. 95% is a property of the
#   **procedure** — intervals built this way cover the truth 95% of the time.
# - *"B causes higher spend."* Unless assignment was randomised, this is a
#   correlation. Module 09 is about earning the causal claim.
# - *"The difference is large enough to act on."* Statistical and practical
#   significance are different questions. A £1.20 difference can be
#   unambiguously real and commercially irrelevant.
#
# **If the interval contains zero**, that is *not* evidence the groups are the
# same — it is a failure to distinguish them, which is also what you get from too
# small a sample. "Absence of evidence is not evidence of absence" is the whole
# reason Exercise 7.5 exists.
#
# One more caveat specific to the median: the bootstrap distribution of a median
# is **discrete and lumpy**, because a resampled median can only take values
# present in the data. It works, but percentile intervals for medians are less
# accurate than for means, especially in small samples.

# %%
print(f"distinct values in the bootstrap distribution: "
      f"{np.unique(differences).size:,} of {N_BOOT:,} resamples")

# %% [markdown]
# ## Exercise 7.5 — Sample size by simulation

# %%
BASELINE, LIFT = 0.10, 0.11
ALPHA, TARGET_POWER = 0.05, 0.80
N_SIMULATIONS = 2_000


def simulate_power(n_per_arm: int, p_control: float, p_treatment: float,
                   n_sims: int, rng: np.random.Generator) -> float:
    """Share of simulated A/B tests that reach significance.

    Uses the two-proportion z-test with a pooled standard error — the test an
    experimentation platform actually runs.
    """
    # One binomial draw per arm per simulated experiment: fully vectorised.
    successes_c = rng.binomial(n_per_arm, p_control, size=n_sims)
    successes_t = rng.binomial(n_per_arm, p_treatment, size=n_sims)

    rate_c = successes_c / n_per_arm
    rate_t = successes_t / n_per_arm
    pooled = (successes_c + successes_t) / (2 * n_per_arm)

    standard_error = np.sqrt(pooled * (1 - pooled) * 2 / n_per_arm)
    # A pooled rate of exactly 0 or 1 gives SE = 0; those tests cannot reject.
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(standard_error > 0, (rate_t - rate_c) / standard_error, 0.0)

    critical = 1.959963984540054   # two-sided 5%
    return float((np.abs(z) > critical).mean())


power_rng = np.random.default_rng(seed=99)
candidate_sizes = np.array([2_000, 5_000, 8_000, 11_000, 14_000, 17_000,
                            20_000, 25_000, 30_000])

powers = np.array([
    simulate_power(int(n), BASELINE, LIFT, N_SIMULATIONS, power_rng)
    for n in candidate_sizes
])

print(f"{'n per arm':>12}{'simulated power':>18}")
print("-" * 30)
for n, power in zip(candidate_sizes, powers, strict=True):
    marker = "  ← ≥ 80%" if power >= TARGET_POWER else ""
    print(f"{n:>12,}{power:>17.1%}{marker}")

# %%
# Interpolate the crossing rather than reporting the nearest grid point.
n_required = float(np.interp(TARGET_POWER, powers, candidate_sizes))
print(f"\nsimulated requirement: ≈ {n_required:,.0f} per arm")

# The analytic normal approximation, for comparison.
z_alpha, z_beta = 1.959963984540054, 0.8416212335729143
p_bar = (BASELINE + LIFT) / 2
numerator = (
    z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
    + z_beta * np.sqrt(BASELINE * (1 - BASELINE) + LIFT * (1 - LIFT))
) ** 2
n_analytic = numerator / (LIFT - BASELINE) ** 2

print(f"analytic formula     : ≈ {n_analytic:,.0f} per arm")
print(f"agreement            : {n_required / n_analytic:.2f}×")
print(f"\ntotal traffic needed : {2 * n_analytic:,.0f} visitors")

# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(candidate_sizes, powers, marker="o", color=COLORS["primary"],
        linewidth=2, label=f"simulated ({N_SIMULATIONS:,} tests per point)")
ax.axhline(TARGET_POWER, color=COLORS["danger"], linestyle="--", linewidth=1.4,
           label="80% power target")
ax.axvline(n_analytic, color=COLORS["teal"], linestyle=":", linewidth=1.6,
           label=f"analytic n ≈ {n_analytic:,.0f}")
ax.set_xlabel("sample size per arm")
ax.set_ylabel("power  (P(detect the lift))")
ax.set_title("A 10% → 11% lift needs ~15 000 users per arm", loc="left")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Why this is the exercise that changes behaviour
#
# Roughly **30 000 users** to detect a one-percentage-point lift. Teams routinely
# run a test on 2 000 visitors, see no significant difference, and conclude the
# change did nothing. At that size the power is around 15% — the test was almost
# guaranteed to find nothing whatever the truth.
#
# Two properties of the curve are worth carrying:
#
# - **Required n scales with `1 / effect²`.** Halving the effect you want to
#   detect quadruples the sample. Detecting 10% → 10.5% needs about 4× the
#   traffic of 10% → 11%.
# - **Power must be computed before the test, not after.** "Observed power"
#   calculated from the result you got is a deterministic function of the
#   p-value and adds no information.
#
# The simulation approach generalises where the formula does not: unequal
# allocation, sequential peeking, ratio metrics, clustered assignment, CUPED
# variance reduction. When the analytic formula stops existing, simulate.
# Module 09 builds all of this out.

# %% [markdown]
# ---
# # Notebook 08 — Performance & Memory

# %% [markdown]
# ## Exercise 8.1 — Predict the strides
#
# A stride is **how many bytes to skip to move one step along that axis**. For
# `float64` (8 bytes) with shape `(2, 3, 4)` in C order, the last axis steps by 8,
# the middle by `4 × 8 = 32`, and the first by `3 × 4 × 8 = 96`.
#
# | Expression | Strides | Why |
# | :--- | :--- | :--- |
# | `a` | `(96, 32, 8)` | Row-major default. |
# | `a.T` | `(8, 32, 96)` | Reversing the axes reverses the strides. **No data moves.** |
# | `a[::2]` | `(192, 32, 8)` | Step 2 along axis 0 doubles that stride. |
# | `a[:, ::2, :]` | `(96, 64, 8)` | Step 2 along axis 1 doubles *its* stride. |
# | `a.reshape(6, 4)` | `(32, 8)` | Contiguous data reinterpreted; strides recomputed from the new shape. |

# %%
a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

stride_cases = {
    "a": a,
    "a.T": a.T,
    "a[::2]": a[::2],
    "a[:, ::2, :]": a[:, ::2, :],
    "a.reshape(6, 4)": a.reshape(6, 4),
}

print(f"{'expression':<18}{'shape':<14}{'strides':<18}{'C-contig':>9}{'view':>7}")
print("-" * 68)
for label, view in stride_cases.items():
    print(f"{label:<18}{str(view.shape):<14}{str(view.strides):<18}"
          f"{str(view.flags['C_CONTIGUOUS']):>9}"
          f"{str(np.shares_memory(a, view)):>7}")

# %% [markdown]
# Every one of them is a **view**. This is the single most important thing about
# strides: transposing a 10 GB array, reversing it, or taking every other row
# costs *nothing* — NumPy edits two small tuples of metadata and hands back a new
# object pointing at the same bytes.
#
# The bill arrives later. `a.T` is free, but it is no longer C-contiguous, so any
# operation that wants contiguous memory (a BLAS call, `.ravel()`, `.tobytes()`)
# has to make a copy at that point.

# %%
big = np.ones((3000, 3000))
t_transpose = bench(lambda: big.T, repeat=100)
t_copy = bench(lambda: np.ascontiguousarray(big.T), repeat=3)
print(f"big.T                        : {t_transpose * 1e6:8.2f} µs   (metadata only)")
print(f"np.ascontiguousarray(big.T)  : {t_copy * 1e3:8.2f} ms   "
      f"({big.nbytes / 1e6:.0f} MB actually moved)")

# %% [markdown]
# ## Exercise 8.2 — Row-major versus column-major
#
# **Prediction first.** A cache line is 64 bytes — eight `float64` values. Walking
# memory in order gets all eight for one fetch; jumping by a full row wastes seven
# eighths of every fetch. So the fast combinations should be the ones whose inner
# loop runs along contiguous memory: `axis=1` on C-order, `axis=0` on F-order.

# %%
SIDE = 3000
c_order = np.ascontiguousarray(rng.random((SIDE, SIDE)))
f_order = np.asfortranarray(c_order)

print(f"each array: {c_order.nbytes / 1e6:.0f} MB")
print(f"C-order strides: {c_order.strides}")
print(f"F-order strides: {f_order.strides}\n")

results = {}
for order, array in (("C", c_order), ("F", f_order)):
    for axis in (0, 1):
        results[(order, axis)] = bench(lambda a=array, ax=axis: a.sum(axis=ax),
                                       repeat=5)

print(f"{'layout':<10}{'axis=0':>12}{'axis=1':>12}")
print("-" * 34)
for order in ("C", "F"):
    print(f"{order + '-order':<10}{results[(order, 0)] * 1000:>11.1f}ms"
          f"{results[(order, 1)] * 1000:>11.1f}ms")

fastest = min(results, key=results.get)
slowest = max(results, key=results.get)
print(f"\nfastest: {fastest[0]}-order, axis={fastest[1]}  "
      f"({results[fastest] * 1000:.1f} ms)")
print(f"slowest: {slowest[0]}-order, axis={slowest[1]}  "
      f"({results[slowest] * 1000:.1f} ms)  "
      f"→ {results[slowest] / results[fastest]:.1f}× slower")

# %% [markdown]
# ### Reading the result honestly
#
# The prediction is directionally right — the mismatched combinations are the slow
# ones — but the gap is usually smaller than the "8× fewer useful bytes per cache
# line" story suggests, and sometimes `axis=0` on C-order is nearly as fast as
# `axis=1`. Two things blunt the effect:
#
# 1. **Hardware prefetching.** A constant stride is a pattern the CPU recognises,
#    so it fetches ahead and hides much of the latency. Random access would not
#    get this help.
# 2. **The accumulator fits in cache.** Summing a C-order array along `axis=0`
#    can stream one contiguous row at a time into a 3000-element accumulator that
#    stays resident in L1. The reads are sequential; only the writes scatter.
#
# The reliable takeaway is not a fixed multiplier, it is: **layout is a real
# variable, and it is nearly free to control.** When an array is written once and
# reduced along one axis many times, match the layout to the access pattern.

# %%
# scikit-learn, for one, has strong opinions about layout.
column_slice = bench(lambda a=c_order: a[:, 500].copy(), repeat=200)
row_slice = bench(lambda a=c_order: a[500, :].copy(), repeat=200)
print(f"C-order: copying one ROW    : {row_slice * 1e6:7.1f} µs  (contiguous)")
print(f"C-order: copying one COLUMN : {column_slice * 1e6:7.1f} µs  "
      f"({column_slice / row_slice:.0f}× slower — 3000 separate fetches)")

del c_order, f_order

# %% [markdown]
# ## Exercise 8.3 — Eliminate the temporaries
#
# `np.sqrt(a**2 + b**2) / (c + 1)` allocates five full-size temporaries: `a**2`,
# `b**2`, their sum, its square root, and `c + 1`. Every one is an allocation, a
# full pass through memory, and a deallocation.

# %%
n = 5_000_000
a_arr = rng.random(n)
b_arr = rng.random(n)
c_arr = rng.random(n)


def naive(a, b, c):
    """Five temporaries."""
    return np.sqrt(a ** 2 + b ** 2) / (c + 1)


def with_out(a, b, c, buf1, buf2):
    """Two caller-supplied buffers, zero new allocations."""
    np.multiply(a, a, out=buf1)      # buf1 = a²
    np.multiply(b, b, out=buf2)      # buf2 = b²
    np.add(buf1, buf2, out=buf1)     # buf1 = a² + b²
    np.sqrt(buf1, out=buf1)          # buf1 = sqrt(a² + b²)
    np.add(c, 1.0, out=buf2)         # buf2 = c + 1
    np.divide(buf1, buf2, out=buf1)  # buf1 = result
    return buf1


buffer_1 = np.empty(n)
buffer_2 = np.empty(n)

expected = naive(a_arr, b_arr, c_arr)
actual = with_out(a_arr, b_arr, c_arr, buffer_1, buffer_2)
print(f"results identical: {np.array_equal(expected, actual)}")

# %%
def peak_allocation(fn, *args) -> tuple[float, float]:
    """Return (best time in seconds, peak traced memory in MB)."""
    elapsed = bench(fn, *args, repeat=3)
    tracemalloc.start()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1e6


t_naive, mem_naive = peak_allocation(naive, a_arr, b_arr, c_arr)
t_out, mem_out = peak_allocation(with_out, a_arr, b_arr, c_arr,
                                 buffer_1, buffer_2)

array_mb = a_arr.nbytes / 1e6
print(f"one array = {array_mb:.0f} MB\n")
print(f"{'version':<22}{'time':>10}{'peak alloc':>14}{'× array':>10}")
print("-" * 56)
print(f"{'naive':<22}{t_naive * 1000:>9.1f}ms{mem_naive:>13.0f}MB"
      f"{mem_naive / array_mb:>9.1f}×")
print(f"{'out= with 2 buffers':<22}{t_out * 1000:>9.1f}ms{mem_out:>13.0f}MB"
      f"{mem_out / array_mb:>9.1f}×")
print(f"\nspeed-up {t_naive / t_out:.2f}×, "
      f"allocation reduced by {mem_naive - mem_out:.0f} MB")

# %% [markdown]
# The allocation saving is dramatic and unambiguous. The **time** saving is
# modest, and that is worth being honest about: the arithmetic still has to
# happen, and the temporaries were being reused out of NumPy's small-block cache
# anyway.
#
# Where `out=` earns its keep:
#
# - **The array does not fit comfortably in RAM.** Five temporaries of a 20 GB
#   array is not a slowdown, it is an `OOMKilled`.
# - **The expression is inside a loop** (Exercise 1.5), where the allocations
#   repeat thousands of times.
# - **You are writing a library** and cannot know how big the caller's data is.
#
# And where it does not: one-off notebook cells on data that fits in memory. The
# `out=` version is meaningfully harder to read — six statements where there was
# one, with the result hiding in `buf1`. Do not pay that cost without a profile
# that justifies it.

# %%
del a_arr, b_arr, c_arr, buffer_1, buffer_2, expected, actual

# %% [markdown]
# ## Exercise 8.4 — Chunked statistics
#
# One pass, bounded memory. Accumulate the count, the sum, and the sum of squares
# — then remember Exercise 3.5, because `E[x²] − E[x]²` is a textbook
# catastrophic cancellation when the mean is far from zero.

# %%
scratch = Path("chunked_demo.dat")
ROWS, COLS = 500_000, 8
DTYPE = np.float64

# Write a file too big to want in memory all at once, with a large offset in the
# values — which is exactly the condition that breaks the naive formula.
source = np.memmap(scratch, dtype=DTYPE, mode="w+", shape=(ROWS, COLS))
write_rng = np.random.default_rng(seed=11)
for start in range(0, ROWS, 100_000):
    stop = min(start + 100_000, ROWS)
    source[start:stop] = write_rng.normal(loc=1e6, scale=3.0,
                                          size=(stop - start, COLS))
source.flush()
del source

print(f"file: {scratch.stat().st_size / 1e6:.0f} MB "
      f"({ROWS:,} × {COLS} {np.dtype(DTYPE).name})")


# %%
def chunked_stats(path, shape, dtype, chunk_rows: int) -> tuple[float, float]:
    """Mean and standard deviation of a memmapped array, in one pass.

    Parameters
    ----------
    path : str or Path
        File holding the raw array.
    shape : tuple
        Shape of the stored array.
    dtype : dtype
        Element type.
    chunk_rows : int
        Rows to hold in memory at a time.

    Returns
    -------
    (mean, std) : tuple of float

    Notes
    -----
    Accumulating ``sum(x**2)`` and using ``E[x²] - E[x]²`` loses catastrophic
    precision when the mean is large relative to the spread. This version
    subtracts a **shift** — the first chunk's mean — before squaring, which
    costs one extra subtraction and restores every digit.
    """
    data = np.memmap(path, dtype=dtype, mode="r", shape=shape)

    shift = float(data[:min(chunk_rows, shape[0])].mean())

    count = 0
    total = 0.0
    total_sq = 0.0

    for start in range(0, shape[0], chunk_rows):
        chunk = np.asarray(data[start:start + chunk_rows], dtype=np.float64)
        centred = chunk - shift
        count += centred.size
        total += centred.sum()
        total_sq += (centred ** 2).sum()

    mean_centred = total / count
    # Population variance of the shifted data equals that of the original.
    variance = total_sq / count - mean_centred ** 2
    return shift + mean_centred, float(np.sqrt(max(variance, 0.0)))


def chunked_stats_naive(path, shape, dtype, chunk_rows: int) -> tuple[float, float]:
    """The version without the shift — kept to show what it costs."""
    data = np.memmap(path, dtype=dtype, mode="r", shape=shape)
    count = 0
    total = 0.0
    total_sq = 0.0
    for start in range(0, shape[0], chunk_rows):
        chunk = np.asarray(data[start:start + chunk_rows], dtype=np.float64)
        count += chunk.size
        total += chunk.sum()
        total_sq += (chunk ** 2).sum()
    mean = total / count
    return mean, float(np.sqrt(max(total_sq / count - mean ** 2, 0.0)))


# %%
truth = np.memmap(scratch, dtype=DTYPE, mode="r", shape=(ROWS, COLS))
true_mean, true_std = float(truth.mean()), float(truth.std())
del truth

shifted_mean, shifted_std = chunked_stats(scratch, (ROWS, COLS), DTYPE, 50_000)
naive_mean, naive_std = chunked_stats_naive(scratch, (ROWS, COLS), DTYPE, 50_000)

print(f"{'method':<26}{'mean':>18}{'std':>14}{'std rel. error':>18}")
print("-" * 76)
print(f"{'np.std on the whole array':<26}{true_mean:>18.6f}{true_std:>14.9f}"
      f"{'—':>18}")
print(f"{'chunked, shifted':<26}{shifted_mean:>18.6f}{shifted_std:>14.9f}"
      f"{abs(shifted_std - true_std) / true_std:>18.2e}")
print(f"{'chunked, naive E[x²]-E[x]²':<26}{naive_mean:>18.6f}{naive_std:>14.9f}"
      f"{abs(naive_std - true_std) / true_std:>18.2e}")

print(f"\npeak memory: {50_000 * COLS * 8 / 1e6:.0f} MB per chunk "
      f"instead of {ROWS * COLS * 8 / 1e6:.0f} MB")

# %% [markdown]
# With a mean of 10⁶ and a spread of 3, `E[x²]` is about 10¹² and `E[x]²` is about
# 10¹² as well — subtracting them destroys the leading digits of a quantity worth
# only 9. The shifted version subtracts a rough mean *before* squaring, so it
# squares numbers near zero and keeps full precision. The cost is one extra
# subtraction per chunk.
#
# For production, prefer the **Chan parallel-merge** form: it combines
# `(count, mean, M2)` triples from independent chunks, so it also works when the
# chunks are processed out of order or on different machines.

# %%
def merge_moments(left, right):
    """Chan et al. pairwise merge of (count, mean, M2) accumulators."""
    n_a, mean_a, m2_a = left
    n_b, mean_b, m2_b = right
    if n_a == 0:
        return right
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    m2 = m2_a + m2_b + delta ** 2 * n_a * n_b / n
    return n, mean, m2


data = np.memmap(scratch, dtype=DTYPE, mode="r", shape=(ROWS, COLS))
state = (0, 0.0, 0.0)
for start in range(0, ROWS, 50_000):
    chunk = np.asarray(data[start:start + 50_000], dtype=np.float64).ravel()
    state = merge_moments(state, (chunk.size, chunk.mean(),
                                  ((chunk - chunk.mean()) ** 2).sum()))
del data

chan_count, chan_mean, chan_m2 = state
chan_std = float(np.sqrt(chan_m2 / chan_count))
print(f"Chan merge → mean {chan_mean:.6f}, std {chan_std:.9f}")
print(f"relative error vs np.std: {abs(chan_std - true_std) / true_std:.2e}")

scratch.unlink()
print(f"\ncleaned up {scratch.name}")

# %% [markdown]
# ## Exercise 8.5 — Know when to stop
#
# A cumulative maximum is inherently sequential: `out[i]` depends on `out[i-1]`.
# That is exactly the shape of problem people assume NumPy cannot help with — and
# `np.maximum.accumulate` does it in C anyway.

# %%
N = 10_000_000
values = rng.random(N)


def cummax_python(x):
    """The obvious loop."""
    out = np.empty_like(x)
    running = -np.inf
    for i in range(x.size):
        running = max(running, x[i])
        out[i] = running
    return out


def cummax_numpy(x):
    """One ufunc call."""
    return np.maximum.accumulate(x)


def cummax_chunked(x, chunk: int = 1_000_000):
    """Chunked, carrying the running maximum across chunk boundaries."""
    out = np.empty_like(x)
    running = -np.inf
    for start in range(0, x.size, chunk):
        stop = min(start + chunk, x.size)
        block = np.maximum.accumulate(x[start:stop])
        np.maximum(block, running, out=out[start:stop])
        running = out[stop - 1]
    return out


reference = cummax_numpy(values)
assert np.array_equal(cummax_chunked(values), reference)
assert np.array_equal(cummax_python(values[:100_000]), reference[:100_000])
print("all three agree ✔")

# %%
t_numpy = bench(cummax_numpy, values, repeat=3)
t_chunked = bench(cummax_chunked, values, repeat=3)
t_python = bench(cummax_python, values, repeat=1)

print(f"{'implementation':<28}{'time':>12}{'vs NumPy':>12}")
print("-" * 52)
for name, seconds in (("np.maximum.accumulate", t_numpy),
                      ("chunked accumulate", t_chunked),
                      ("Python loop", t_python)):
    print(f"{name:<28}{seconds:>11.3f}s{seconds / t_numpy:>11.1f}×")

# %% [markdown]
# ### The ranking, and why
#
# 1. **`np.maximum.accumulate`** — one call into C, one pass over memory, no
#    Python objects created. This is the answer.
# 2. **Chunked** — the same C loop, plus a handful of Python-level iterations and
#    an extra `np.maximum` per chunk. Slightly slower for no benefit *here*,
#    because the array already fits in memory. It earns its place when the input
#    is a memmap or a stream larger than RAM.
# 3. **The Python loop** — two orders of magnitude behind, because it pays the
#    interpreter's dispatch and boxing costs ten million times.
#
# ### Where the exercise title comes from
#
# The chunked version is more code, more state, and more places to get an
# off-by-one wrong — and on this input it is *slower*. That is the lesson:
# **optimisation without a measured constraint is a cost, not a benefit.**
#
# Reach for the more complex version when a profile or a memory ceiling tells you
# to. Ask, in order: (1) is it vectorised? (2) is it the right algorithm? (3) does
# it fit in memory? Only after all three should the answer involve chunking,
# Numba, or Cython.

# %%
del values, reference

# %% [markdown]
# ---
# ## You have finished Module 02
#
# Forty exercises. The threads worth carrying into Module 03:
#
# | Idea | Where it came back |
# | :--- | :--- |
# | Views alias; copies do not | 2.2, 2.3 — and every `SettingWithCopyWarning` you will meet in pandas |
# | Shapes right-align, then stretch | 4.1, 4.2, 4.3, 5.2 — the source of most silent numerical bugs |
# | `keepdims=True` when dividing by your own reduction | 4.3, 5.2 |
# | Expand the algebra before allocating `(n, m, d)` | 4.4, 4.5, 6.2 |
# | Store narrow, accumulate wide | 3.2, 3.3, 8.4 |
# | Never subtract two nearly equal large numbers | 3.5, 4.4, 8.4 |
# | `lstsq`/`solve`, never `inv` | 6.3, 6.4 |
# | Randomness is a parameter, not a global | 7.1, 7.4, 7.5 |
# | Measure before optimising | 1.3, 1.5, 8.3, 8.5 |
#
# **Next:** Module 03 — pandas & Data Wrangling. A DataFrame is a set of labelled
# NumPy arrays, and every one of the rules above still applies underneath —
# including, especially, the ones about views.
