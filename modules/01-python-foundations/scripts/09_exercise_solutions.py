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
# # 09 · Exercise Solutions
#
# **Module 01 — Python Foundations for Data Science**
#
# ---
#
# > [!WARNING]
# > **Attempt every exercise before reading its solution.** Reading a worked
# > solution produces a feeling of understanding that disappears the moment you
# > face a blank cell. Struggling for ten minutes teaches more than reading for
# > sixty.
#
# Where a solution has more than one reasonable form, the alternatives are shown
# and the trade-off explained — because on the job the interesting question is
# rarely "does it work?" but "which version would I want to maintain?"

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dstoolkit import setup_plot_style

setup_plot_style()

# %% [markdown]
# ---
# # Notebook 01 — Course Introduction & Setup

# %% [markdown]
# ## Exercise 1.1 — Environment audit

# %%
import matplotlib

packages = {"numpy": np, "pandas": pd, "matplotlib": matplotlib}

print(f"Python version : {sys.version_info[:3]}")
print(f"In a venv      : {sys.prefix != sys.base_prefix}")
print(f"Interpreter    : {sys.executable}")
print("\nInstalled packages:")
for name, module in packages.items():
    # `getattr` with a default keeps this working for the occasional package
    # that does not expose __version__.
    print(f"  {name:<12} {getattr(module, '__version__', 'unknown')}")

# %% [markdown]
# ## Exercise 1.2 — Reproduce with a different seed
#
# The point of this exercise is to build intuition for **sampling variability**:
# a single sample is one draw from a distribution of possible samples.

# %%
def simulate_customers(seed: int, n: int = 2_000) -> pd.DataFrame:
    """Regenerate the notebook-01 dataset with a given seed."""
    rng = np.random.default_rng(seed)
    channel = rng.choice(["organic", "paid"], size=n, p=[0.55, 0.45])
    spend = np.where(
        channel == "organic",
        rng.lognormal(4.30, 0.55, size=n),
        rng.lognormal(4.05, 0.60, size=n),
    )
    return pd.DataFrame({"channel": channel, "spend_usd": spend.round(2)})


comparison = []
for seed in (42, 7, 123, 2024, 99):
    df = simulate_customers(seed)
    means = df.groupby("channel")["spend_usd"].mean()
    comparison.append({
        "seed": seed,
        "organic": means["organic"],
        "paid": means["paid"],
        "difference": means["organic"] - means["paid"],
    })

results = pd.DataFrame(comparison).set_index("seed").round(2)
results

# %% [markdown]
# **Answers**
#
# - **Do the means change?** Yes — by several dollars between seeds.
# - **Does the conclusion change?** No. Organic spend exceeds paid in every
#   sample, and the difference keeps the same sign and rough magnitude.
# - **What does that tell you?** A single sample gives a *noisy* estimate of the
#   quantity you care about. The direction is stable here because the true effect
#   is large relative to the noise. When an effect is small, a single sample can
#   easily point the wrong way — which is precisely why Module 08 teaches
#   confidence intervals rather than point estimates.

# %%
print(f"spread of the estimated difference across seeds : "
      f"${results['difference'].min():.2f} – ${results['difference'].max():.2f}")
print(f"every seed agrees on the sign                   : "
      f"{(results['difference'] > 0).all()}")

# %% [markdown]
# ## Exercise 1.3 — Break the notebook deliberately
#
# **What happens:** `summary` was computed from the 2 000-row `customers`. After
# rebuilding `customers` with 50 rows, `summary` still holds the *old* result —
# it is a separate object, not a live view. The notebook now displays statistics
# that describe data no longer in memory.
#
# **The menu command:** *Kernel → Restart Kernel and Run All Cells*. It rebuilds
# every variable from a clean state in document order, which is the only way to
# be sure what you see matches what the code says.

# %% [markdown]
# ## Exercise 1.4 — Lifecycle mapping (example: a delivery-ETA model)
#
# | Stage | What the team had to do |
# | :--- | :--- |
# | 1. Business understanding | Decide whether the goal is accuracy or *not being late* — the two imply different loss functions. |
# | 2. Data collection | Join historical trips, GPS traces, restaurant prep times, and weather. |
# | 3. Cleaning | Handle GPS dropouts, orders cancelled mid-delivery, clock skew between systems. |
# | 4. EDA | Discover ETA error is far worse at peak hours and in dense city centres. |
# | 5. Feature engineering | Build features for time-of-day, courier load, distance, historical restaurant delay. |
# | 6. Modelling | Gradient boosting on tabular features; possibly a quantile model to target lateness. |
# | 7. Evaluation | Optimise a quantile loss, not MAE — a 5-minute overestimate is far cheaper than a 5-minute underestimate. |
# | 8. Deployment & monitoring | Serve at sub-100 ms latency; monitor drift when a new city launches. |

# %% [markdown]
# ---
# # Notebook 02 — Python Fundamentals

# %% [markdown]
# ## Exercise 2.1 — Predict before you run

# %%
# --- (a) b = b + [4] --------------------------------------------------------
a = [1, 2, 3]
b = a
b = b + [4]
print(f"(a) a = {a}      ← unchanged")
print(f"    b = {b}")

# --- (b) b += [4] -----------------------------------------------------------
a = [1, 2, 3]
b = a
b += [4]
print(f"\n(b) a = {a}   ← CHANGED")
print(f"    b = {b}")

# --- (c) mutating a list inside a tuple -------------------------------------
t = ([1, 2], 3)
t[0].append(99)
print(f"\n(c) t = {t}")

# %% [markdown]
# **Why (a) and (b) differ**
#
# - `b = b + [4]` calls `__add__`, which **builds a new list** and rebinds the
#   name `b` to it. `a` still points at the original.
# - `b += [4]` calls `__iadd__`, which for a list is **`extend` in place**. Both
#   names still point at the one mutated object.
#
# So `+=` is *not* sugar for `x = x + y` when the object is mutable — it is a
# genuinely different operation. For immutable types (`int`, `str`, `tuple`)
# there is no in-place option, so `+=` falls back to rebinding, which is why the
# distinction is invisible until you meet a list.
#
# **Why (c) works**
#
# A tuple freezes its *references*, not the objects they point to. `t[0]` is a
# list, and the list is mutable. What the tuple forbids is `t[0] = something`,
# which would change which object the slot refers to.

# %%
# The identity check makes the difference concrete.
a = [1, 2, 3]
b = a
b = b + [4]
print(f"after  b = b + [4] : same object? {a is b}")

a = [1, 2, 3]
b = a
b += [4]
print(f"after  b += [4]    : same object? {a is b}")

# %% [markdown]
# ## Exercise 2.2 — A float comparison bug

# %%
def is_fully_allocated_broken(weights: list[float]) -> bool:
    return sum(weights) == 1.0


def is_fully_allocated(weights: list[float], *, tolerance: float = 1e-9) -> bool:
    """Return True when portfolio weights sum to 1, within floating-point tolerance.

    Parameters
    ----------
    weights : list of float
        Portfolio weights.
    tolerance : float, default 1e-9
        Absolute tolerance. The default is far tighter than any real portfolio
        needs, while still absorbing accumulated representation error.
    """
    return np.isclose(sum(weights), 1.0, atol=tolerance, rtol=0.0)


weights = [0.1, 0.2, 0.3, 0.4]

print(f"sum          : {sum(weights)!r}")
print(f"the error    : {sum(weights) - 1.0:.2e}")
print(f"broken       : {is_fully_allocated_broken(weights)}")
print(f"fixed        : {is_fully_allocated(weights)}")

# It must still reject genuinely wrong input.
print(f"\nrejects 0.99 : {is_fully_allocated([0.5, 0.49])}")

# %% [markdown]
# **The bug:** `0.1 + 0.2 + 0.3 + 0.4` accumulates four representation errors and
# lands a few times 10⁻¹⁶ away from exactly 1.0. `==` on computed floats is
# almost always wrong.
#
# > For money specifically, the better answer is `Decimal` — but for *weights*,
# > a tolerance is the right tool.

# %% [markdown]
# ## Exercise 2.3 — Safe enrichment

# %%
def add_discount(orders: list[dict], pct: float) -> list[dict]:
    """Return new order dicts with a `discounted_price` field.

    Parameters
    ----------
    orders : list of dict
        Each must contain a numeric `price`.
    pct : float
        Discount as a fraction, e.g. 0.15 for 15%.

    Returns
    -------
    list of dict
        New dicts. The input is never modified.
    """
    if not 0 <= pct < 1:
        raise ValueError(f"pct must be in [0, 1), got {pct}")

    # `{**order, ...}` builds a NEW dict — the key to not mutating the input.
    return [
        {**order, "discounted_price": round(order["price"] * (1 - pct), 2)}
        for order in orders
    ]


orders = [{"id": 1, "price": 100.0}, {"id": 2, "price": 250.0}]
import copy

snapshot = copy.deepcopy(orders)

discounted = add_discount(orders, 0.15)

assert orders == snapshot, "add_discount must not modify its input"
assert all("discounted_price" not in o for o in orders)

print(f"input     : {orders}")
print(f"output    : {discounted}")
print("\n✔ assertion passed — the input was not modified")

# %% [markdown]
# ## Exercise 2.4 — Text cleaning

# %%
def clean_city(raw: str) -> str:
    """Normalise a messy city name.

    Examples
    --------
    >>> clean_city("  sÃO   paulo\\n")
    'São Paulo'
    """
    # `.split()` with no argument splits on ANY run of whitespace (including
    # newlines and tabs) and discards empties — so join+split collapses
    # everything in one step.
    collapsed = " ".join(raw.split())
    return collapsed.title()


for raw in ["  sÃO   paulo\n", "RIO   de\tjaneiro ", "  belo horizonte  "]:
    print(f"{raw!r:<28} → {clean_city(raw)!r}")

# %% [markdown]
# > [!NOTE]
# > `.title()` is imperfect for real names — it produces `"Rio De Janeiro"` where
# > Portuguese style wants a lowercase `"de"`. Handling that properly requires a
# > list of particles to keep lowercase. The exercise's point is the
# > `" ".join(raw.split())` idiom; the naming nuance is a good reminder that text
# > normalisation is always locale-specific.

# %%
def clean_city_locale_aware(raw: str, particles: frozenset[str] = frozenset(
    {"de", "da", "do", "das", "dos", "e"}
)) -> str:
    """Title-case a place name while keeping Portuguese particles lowercase."""
    words = " ".join(raw.split()).lower().split(" ")
    return " ".join(
        # The first word is always capitalised, even if it is a particle.
        word.capitalize() if i == 0 or word not in particles else word
        for i, word in enumerate(words)
    )


for raw in ["RIO   de\tjaneiro ", "  são joão dos campos "]:
    print(f"{raw!r:<28} → {clean_city_locale_aware(raw)!r}")

# %% [markdown]
# ## Exercise 2.5 — Missing-value audit

# %%
values = [1.0, np.nan, 3.0, None, 5.0]

# `x != x` is True only for NaN — the defining property from notebook 02.
n_missing = sum(1 for v in values if v is None or (isinstance(v, float) and np.isnan(v)))

print(f"values      : {values}")
print(f"n_missing   : {n_missing}")

# Why .count() fails:
print(f"\nvalues.count(np.nan) : {values.count(np.nan)}")

# %% [markdown]
# **Why `.count(np.nan)` is unreliable**
#
# `list.count` compares with `==`, and `nan == nan` is `False` by IEEE 754. It
# returns 1 here only because `count` short-circuits on **identity** first
# (`x is nan or x == nan`), and this list happens to hold the *same* `np.nan`
# object. Build the list differently — say from `float("nan")` values, which are
# distinct objects — and the count silently becomes 0.

# %%
# Proof that the behaviour depends on object identity, not value:
same_object = [float("nan")] * 3          # one object, repeated
distinct_objects = [float("nan") for _ in range(3)]   # three separate objects

print(f"same object     : count = {same_object.count(float('nan'))}"
      f"  (searching for a DIFFERENT nan object)")
print(f"distinct objects: count = {distinct_objects.count(float('nan'))}")
print("\n→ Never test for NaN with == or .count(). Use np.isnan / pd.isna.")

# The reliable ways:
print(f"\nnp.isnan on an array : {np.isnan(np.array([1.0, np.nan])).sum()}")
print(f"pd.isna handles both : {pd.isna(pd.Series(values)).sum()}")

# %% [markdown]
# ---
# # Notebook 03 — Data Structures

# %% [markdown]
# ## Exercise 3.1 — Pick the structure
#
# | # | Requirement | Choice | Why |
# | :-: | :--- | :--- | :--- |
# | 1 | 50M processed user IDs, to skip duplicates | **`set`** | O(1) membership; a list would make each check a 50-million-element scan. |
# | 2 | Last 100 sensor readings for a moving average | **`collections.deque(maxlen=100)`** | O(1) at both ends, and `maxlen` evicts old items automatically. |
# | 3 | Country code → country name | **`dict`** | Exactly a key-value mapping, with O(1) lookup. |
# | 4 | An RGB colour that must never change | **`tuple`** | Immutable, hashable, and position carries meaning. |
# | 5 | Word frequencies over 10 000 reviews | **`collections.Counter`** | Purpose-built; `.most_common()` comes free. |

# %% [markdown]
# ## Exercise 3.2 — Fix the slow function

# %%
import time


def find_new_customers_slow(all_ids: list[int], known_ids: list[int]) -> list[int]:
    """O(n × m) — scans `known_ids` once per element of `all_ids`."""
    return [i for i in all_ids if i not in known_ids]


def find_new_customers(all_ids: list[int], known_ids: list[int]) -> list[int]:
    """O(n + m) — one pass to build the set, one to filter."""
    known = set(known_ids)          # the entire fix is this line
    return [i for i in all_ids if i not in known]


rng = np.random.default_rng(42)
# Deliberately modest sizes so the slow version still finishes in this notebook.
all_ids = rng.integers(0, 200_000, size=20_000).tolist()
known_ids = rng.integers(0, 200_000, size=20_000).tolist()

start = time.perf_counter()
slow_result = find_new_customers_slow(all_ids, known_ids)
slow_time = time.perf_counter() - start

start = time.perf_counter()
fast_result = find_new_customers(all_ids, known_ids)
fast_time = time.perf_counter() - start

print(f"slow : {slow_time * 1000:9.2f} ms")
print(f"fast : {fast_time * 1000:9.2f} ms")
print(f"→ {slow_time / fast_time:,.0f}× faster, identical output: "
      f"{slow_result == fast_result}")
print(f"\nAt the exercise's 100k × 100k this gap becomes minutes versus milliseconds.")

# %% [markdown]
# ## Exercise 3.3 — Comprehension practice

# %%
sales = [
    {"product": "A", "units": 10, "price": 5.0},
    {"product": "B", "units": 3, "price": 45.0},
    {"product": "C", "units": 25, "price": 2.5},
    {"product": "D", "units": 8, "price": 30.0},
]

# 1. A list of revenues.
revenues = [s["units"] * s["price"] for s in sales]

# 2. A dict mapping product → revenue.
revenue_by_product = {s["product"]: s["units"] * s["price"] for s in sales}

# 3. The set of products with revenue above 100.
high_revenue = {s["product"] for s in sales if s["units"] * s["price"] > 100}

print(f"1. revenues           : {revenues}")
print(f"2. revenue_by_product : {revenue_by_product}")
print(f"3. high_revenue       : {sorted(high_revenue)}")

# %% [markdown]
# ## Exercise 3.4 — Group without pandas

# %%
from collections import defaultdict
from typing import Any


def group_by_key(records: list[dict], key: str) -> dict[Any, list[dict]]:
    """Group a list of dicts by the value of one field.

    Parameters
    ----------
    records : list of dict
        The records to group.
    key : str
        The field to group by. Every record must contain it.

    Returns
    -------
    dict
        Mapping of field value → list of records having that value.

    Raises
    ------
    KeyError
        If any record lacks `key`.
    """
    grouped: dict[Any, list[dict]] = defaultdict(list)
    for record in records:
        if key not in record:
            raise KeyError(f"record missing group key {key!r}: {record}")
        grouped[record[key]].append(record)
    # Return a plain dict: a defaultdict silently creates entries on lookup,
    # which surprises callers who expect a KeyError for a missing group.
    return dict(grouped)


employees = [
    {"name": "Ana", "dept": "data"},
    {"name": "Bruno", "dept": "eng"},
    {"name": "Carla", "dept": "data"},
]

for dept, members in group_by_key(employees, "dept").items():
    print(f"  {dept:<6} {[m['name'] for m in members]}")

# It agrees with pandas, which is the point.
print(f"\npandas agrees : "
      f"{pd.DataFrame(employees).groupby('dept')['name'].apply(list).to_dict()}")

# %% [markdown]
# ## Exercise 3.5 — Multi-key sorting

# %%
employees = [
    {"name": "Ana", "dept": "data", "salary": 95_000},
    {"name": "Bruno", "dept": "eng", "salary": 105_000},
    {"name": "Carla", "dept": "data", "salary": 110_000},
    {"name": "Diego", "dept": "eng", "salary": 95_000},
    {"name": "Elena", "dept": "data", "salary": 95_000},
]

# A single sorted() call. Negating the salary flips just that field to descending
# while the others stay ascending — the standard trick for mixed-direction sorts
# on numeric keys.
result = sorted(employees, key=lambda e: (e["dept"], -e["salary"], e["name"]))

for e in result:
    print(f"  {e['dept']:<6} ${e['salary']:>7,}  {e['name']}")

# %% [markdown]
# > [!TIP]
# > Negation only works for numbers. To sort a *string* field descending while
# > others ascend, do a stable multi-pass sort instead (notebook 03, section 9):
# > sort by the least significant key first, then by the most significant.

# %% [markdown]
# ---
# # Notebook 04 — Functions & Functional Programming

# %% [markdown]
# ## Exercise 4.1 — Fix the signature

# %%
def summarise(
    data: pd.DataFrame,
    *,
    include_ci: bool = True,
    confidence: float = 0.95,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute summary statistics for the numeric columns of a DataFrame.

    Parameters
    ----------
    data : DataFrame
        The data to summarise.
    include_ci : bool, default True
        Whether to add confidence-interval bounds for the mean.
    confidence : float, default 0.95
        Confidence level, in (0, 1). Ignored when ``include_ci`` is False.
    columns : list of str, optional
        Columns to summarise. Defaults to every numeric column.

    Returns
    -------
    DataFrame
        One row per summarised column.

    Raises
    ------
    ValueError
        If ``confidence`` is outside (0, 1), or a requested column is absent.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if columns is None:
        columns = data.select_dtypes("number").columns.tolist()
    else:
        missing = set(columns) - set(data.columns)
        if missing:
            raise ValueError(f"columns not found: {sorted(missing)}")

    stats = data[columns].agg(["count", "mean", "std", "min", "median", "max"]).T

    if include_ci:
        from scipy import stats as sps

        # Standard error of the mean, then a t-based interval.
        sem = stats["std"] / np.sqrt(stats["count"])
        t_crit = sps.t.ppf((1 + confidence) / 2, df=stats["count"] - 1)
        stats["ci_low"] = stats["mean"] - t_crit * sem
        stats["ci_high"] = stats["mean"] + t_crit * sem

    return stats.round(3)


demo = pd.DataFrame({
    "revenue": np.random.default_rng(42).normal(100, 15, 200),
    "units": np.random.default_rng(7).integers(1, 10, 200),
})
summarise(demo)

# %% [markdown]
# **What changed and why**
#
# - **Keyword-only options** (`*`) — `summarise(df, True, 0.95, None)` was
#   unreadable and would break if the parameter order ever changed.
# - **Type hints** document the contract and let editors catch mistakes.
# - **`columns: list[str] | None = None`** — `None` sentinel instead of a mutable
#   default.
# - **Validation up front** with informative messages.
# - **A docstring** stating what it raises, so callers can handle failures.

# %% [markdown]
# ## Exercise 4.2 — Spot the trap

# %%
def add_feature_broken(df_columns, new_cols=[]):
    new_cols.append("engineered")
    return df_columns + new_cols


print("Three calls with only the first argument:")
for i in range(1, 4):
    print(f"  call {i}: {add_feature_broken(['a'])}")

# %% [markdown]
# **What is happening:** the default list is created **once**, when the `def`
# statement executes. Every call that does not pass `new_cols` appends to that
# same list, so it grows monotonically across calls.

# %%
def add_feature(df_columns: list[str], new_cols: list[str] | None = None) -> list[str]:
    """Return the columns plus an engineered marker. Input is never modified."""
    new_cols = list(new_cols) if new_cols is not None else []
    return [*df_columns, *new_cols, "engineered"]


print("Fixed:")
for i in range(1, 4):
    print(f"  call {i}: {add_feature(['a'])}")

# %% [markdown]
# ## Exercise 4.3 — Write a decorator

# %%
import functools


def validate_positive(func):
    """Raise ValueError if any numeric positional argument is negative.

    Non-numeric arguments (DataFrames, strings, …) are ignored rather than
    rejected, so the decorator can be applied to mixed-signature functions.
    """
    @functools.wraps(func)              # preserve __name__, __doc__, signature
    def wrapper(*args, **kwargs):
        for position, value in enumerate(args):
            # `bool` is a subclass of int — exclude it, since False would
            # otherwise be a perfectly valid "non-negative" flag.
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value < 0:
                    raise ValueError(
                        f"{func.__name__}() argument {position} must be "
                        f"non-negative, got {value}"
                    )
        return func(*args, **kwargs)

    return wrapper


@validate_positive
def compute_growth(current: float, previous: float) -> float:
    """Percentage growth from `previous` to `current`."""
    return (current - previous) / previous * 100


print(f"valid   : {compute_growth(120, 100):.1f}%")

try:
    compute_growth(120, -100)
except ValueError as exc:
    print(f"rejected: {exc}")

print(f"\nmetadata preserved → name: {compute_growth.__name__!r}, "
      f"doc: {compute_growth.__doc__!r}")

# %% [markdown]
# ## Exercise 4.4 — Generator pipeline

# %%
def parse(lines):
    """Yield {'level', 'message'} dicts from raw log lines."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        level, _, message = line.partition(":")
        yield {"level": level.strip().upper(), "message": message.strip()}


def only_errors(records):
    """Yield only records whose level is ERROR."""
    for record in records:
        if record["level"] == "ERROR":
            yield record


def take(records, n: int):
    """Yield at most `n` records, then stop."""
    for i, record in enumerate(records):
        if i >= n:
            return
        yield record


log_lines = [
    "INFO: pipeline started",
    "ERROR: failed to parse row 42",
    "WARNING: 3 nulls imputed",
    "ERROR: connection timeout",
    "INFO: retrying",
    "ERROR: giving up",
]

pipeline = take(only_errors(parse(log_lines)), 2)
print(f"pipeline type : {type(pipeline).__name__}\n")
for record in pipeline:
    print(f"  {record}")

# %%
# Memory proof: build the chain over 10 million lines and measure it.
def endless_lines(n: int):
    for i in range(n):
        yield f"{'ERROR' if i % 3 == 0 else 'INFO'}: message {i}"


for n in (1_000, 1_000_000, 10_000_000):
    chain = take(only_errors(parse(endless_lines(n))), 5)
    total = sys.getsizeof(chain)
    print(f"n = {n:>12,}  →  pipeline holds {total} bytes")

print("\n→ Constant, regardless of input size. A list-based version would need GBs.")

# %% [markdown]
# ## Exercise 4.5 — Closure counter

# %%
def make_counter():
    """Return a function that yields 1, 2, 3, … on successive calls.

    `nonlocal` rebinds a name in the ENCLOSING scope — unlike `global`, which
    would reach all the way out to module level. This keeps the state private to
    the closure, so two counters are fully independent.
    """
    count = 0

    def increment() -> int:
        nonlocal count
        count += 1
        return count

    return increment


counter_a = make_counter()
counter_b = make_counter()

print(f"a: {[counter_a() for _ in range(4)]}")
print(f"b: {[counter_b() for _ in range(2)]}   ← independent state")
print(f"a: {counter_a()}                  ← continues from where it was")

# %% [markdown]
# ---
# # Notebook 05 — Object-Oriented Programming

# %% [markdown]
# ## Exercise 5.1 — Class or function?
#
# | # | Task | Choice | Why |
# | :-: | :--- | :--- | :--- |
# | 1 | Celsius → Fahrenheit | **function** | Stateless, one input, one output. A class would be pure ceremony. |
# | 2 | Database connection settings | **frozen `@dataclass`** | A fixed set of fields, no behaviour, and it must not change mid-run. |
# | 3 | Tokeniser that learns a vocabulary, then encodes | **class** | It has *fitted state* (the vocabulary) that `encode` depends on — the canonical case. |
# | 4 | Median of a list | **function** | Stateless computation. |

# %% [markdown]
# ## Exercise 5.2 — Dunder practice

# %%
class Measurement:
    """A value with a unit, supporting arithmetic and formatting."""

    def __init__(self, value: float, unit: str) -> None:
        self.value = value
        self.unit = unit

    def __repr__(self) -> str:
        return f"Measurement(value={self.value!r}, unit={self.unit!r})"

    def __str__(self) -> str:
        return f"{self.value:g} {self.unit}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Measurement):
            return NotImplemented
        return (self.value, self.unit) == (other.value, other.unit)

    def __hash__(self) -> int:
        return hash((self.value, self.unit))

    def _check_unit(self, other: "Measurement") -> None:
        if self.unit != other.unit:
            raise ValueError(f"unit mismatch: {self.unit} vs {other.unit}")

    def __add__(self, other: "Measurement") -> "Measurement":
        self._check_unit(other)
        return Measurement(self.value + other.value, self.unit)

    def __sub__(self, other: "Measurement") -> "Measurement":
        self._check_unit(other)
        return Measurement(self.value - other.value, self.unit)

    def __mul__(self, factor: float) -> "Measurement":
        """Scale by a plain number. Multiplying two measurements is undefined here."""
        if not isinstance(factor, (int, float)):
            return NotImplemented
        return Measurement(self.value * factor, self.unit)

    # Makes `2 * m` work as well as `m * 2`.
    __rmul__ = __mul__

    def __format__(self, spec: str) -> str:
        """Support f"{m:.2f}" by delegating the spec to the numeric value."""
        if not spec:
            return str(self)
        return f"{self.value:{spec}} {self.unit}"


a = Measurement(23.456, "°C")
b = Measurement(1.5, "°C")

print(f"a          : {a}")
print(f"a + b      : {a + b}")
print(f"a - b      : {a - b}")
print(f"a * 2      : {a * 2}")
print(f"3 * b      : {3 * b}")
print(f"f'{{a:.2f}}' : {a:.2f}")
print(f"f'{{a:>12.1f}}' : '{a:>12.1f}'")

try:
    a + Measurement(1.0, "°F")
except ValueError as exc:
    print(f"\nunit safety : {exc}")

# %% [markdown]
# ## Exercise 5.3 — Dataclass conversion

# %%
from dataclasses import dataclass, field


@dataclass
class Dataset:
    """A named collection of numeric observations."""

    name: str
    values: list[float] = field(default_factory=list)

    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    def summary(self) -> dict[str, float]:
        arr = np.asarray(self.values)
        return {
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }


sales = Dataset("q1_sales", [120.0, 340.5, 89.0, 560.25, 210.0])

print(f"repr for free : {sales}")
print(f"eq for free   : {sales == Dataset('q1_sales', [120.0, 340.5, 89.0, 560.25, 210.0])}")
print(f"mean          : {sales.mean():.2f}")

# %% [markdown]
# **Lines removed:** `__init__` (3 lines), and we gained `__repr__` and `__eq__`
# that the original never had — roughly 15 lines of equivalent hand-written code.
#
# > [!WARNING]
# > One behavioural difference: the original did `self.values = list(values)`,
# > defensively copying the caller's list. The dataclass stores the reference it
# > was given. If that matters, add a `__post_init__` that copies.

# %%
@dataclass
class SafeDataset:
    name: str
    values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Restore the defensive copy the hand-written __init__ had.
        self.values = list(self.values)


caller_list = [1.0, 2.0]
ds = SafeDataset("x", caller_list)
caller_list.append(999.0)

print(f"caller's list : {caller_list}")
print(f"dataset holds : {ds.values}   ← insulated from the caller")

# %% [markdown]
# ## Exercise 5.4 — Build an estimator

# %%
from sklearn.base import BaseEstimator, TransformerMixin


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Replace infrequent categories with a single ``__other__`` label.

    High-cardinality categorical columns explode into thousands of one-hot
    features, most of which are supported by a handful of rows. Collapsing the
    rare tail is a standard, high-value preprocessing step.

    Parameters
    ----------
    min_frequency : int, default 10
        Categories appearing in fewer than this many training rows are grouped.
    other_label : str, default "__other__"
        Replacement label for grouped categories.

    Attributes
    ----------
    frequent_categories_ : dict of {int: set}
        Per column index, the categories that met the threshold during ``fit``.
    """

    def __init__(self, min_frequency: int = 10,
                 other_label: str = "__other__") -> None:
        # scikit-learn contract: store arguments unchanged, under the same names.
        self.min_frequency = min_frequency
        self.other_label = other_label

    def fit(self, X, y=None):
        if self.min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {self.min_frequency}")

        X = pd.DataFrame(X)
        self.frequent_categories_ = {}

        for i, column in enumerate(X.columns):
            counts = X[column].value_counts()
            # A set gives O(1) membership in transform (notebook 03).
            self.frequent_categories_[i] = set(
                counts[counts >= self.min_frequency].index
            )

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "frequent_categories_"):
            raise RuntimeError("call fit() before transform()")

        X = pd.DataFrame(X).copy()
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; fitted on {self.n_features_in_}"
            )

        for i, column in enumerate(X.columns):
            keep = self.frequent_categories_[i]
            X[column] = X[column].where(X[column].isin(keep), self.other_label)

        return X


rng = np.random.default_rng(42)
# A realistic long tail: 3 common categories, 60 rare ones.
categories = (["common_a"] * 300 + ["common_b"] * 200 + ["common_c"] * 150
              + [f"rare_{i}" for i in range(60)])
X_demo = pd.DataFrame({"city": rng.permutation(categories)})

grouper = RareCategoryGrouper(min_frequency=10)
X_grouped = grouper.fit_transform(X_demo)

print(f"distinct categories before : {X_demo['city'].nunique()}")
print(f"distinct categories after  : {X_grouped['city'].nunique()}")
print(f"\nvalue counts after grouping:")
print(X_grouped["city"].value_counts())

# %%
# The payoff: it composes with the rest of scikit-learn.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

encoder_pipeline = Pipeline([
    ("group_rare", RareCategoryGrouper(min_frequency=10)),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
])

encoded = encoder_pipeline.fit_transform(X_demo)

print(f"one-hot columns WITHOUT grouping : {X_demo['city'].nunique()}")
print(f"one-hot columns WITH grouping    : {encoded.shape[1]}")
print(f"→ {X_demo['city'].nunique() - encoded.shape[1]} useless features avoided")

# %% [markdown]
# ## Exercise 5.5 — Find the bug

# %%
class Model:
    history = []        # BUG: a class attribute, shared by every instance

    def fit(self, X, y):
        self.history.append("fitted")
        return self


m1, m2 = Model(), Model()
m1.fit(None, None)
m2.fit(None, None)

print(f"len(m1.history) : {len(m1.history)}")
print(f"len(m2.history) : {len(m2.history)}")
print(f"same list?      : {m1.history is m2.history}")
print(f"even on the class itself : {Model.history}")

# %% [markdown]
# **Explanation:** `history` is defined in the class body, so exactly one list
# exists and it belongs to `Model` itself. `self.history.append(...)` does not
# assign to `self` — it *looks up* `history` (finding the class attribute) and
# mutates it. Both instances therefore share one growing list, and it persists
# for the lifetime of the process.
#
# This is the notebook-02 mutability lesson wearing a class costume.

# %%
class FixedModel:
    def __init__(self) -> None:
        self.history: list[str] = []      # a fresh list per instance

    def fit(self, X, y):
        self.history.append("fitted")
        return self


m1, m2 = FixedModel(), FixedModel()
m1.fit(None, None)
m2.fit(None, None)

print(f"len(m1.history) : {len(m1.history)}")
print(f"len(m2.history) : {len(m2.history)}")
print(f"same list?      : {m1.history is m2.history}")

# %% [markdown]
# ---
# # Notebook 06 — Files & Data Formats

# %% [markdown]
# ## Exercise 6.1 — Path manipulation

# %%
source = Path("data/raw/sales_2024_q1.csv")

# 1. Move to processed/ and change the extension.
processed = source.parent.parent / "processed" / source.with_suffix(".parquet").name

# 2. Extract year and quarter from the stem.
_, year, quarter = source.stem.split("_")

# 3. A sibling backup path.
backup = source.with_name(f"{source.stem}_backup{source.suffix}")

print(f"source    : {source}")
print(f"1. parquet: {processed}")
print(f"2. year   : {year}, quarter: {quarter}")
print(f"3. backup : {backup}")

# %% [markdown]
# ## Exercise 6.2 — Diagnose a broken CSV

# %%
import tempfile

tmpdir = Path(tempfile.mkdtemp())
messy_csv = tmpdir / "messy.csv"

messy_csv.write_text(
    "id;name;price;date\n"
    "001;Café;1.234,56;15/03/2024\n"
    "002;Naïve;987,65;16/03/2024\n",
    encoding="utf-8",
)

df = pd.read_csv(
    messy_csv,
    sep=";",                     # semicolon separator
    encoding="utf-8",
    dtype={"id": str},           # keep the leading zeros
    decimal=",",                 # comma is the decimal mark
    thousands=".",               # dot is the thousands separator
    parse_dates=["date"],
    date_format="%d/%m/%Y",      # day-first, or 15/03 would be misread
)

print(df)
print()
print(df.dtypes)
print()
print(f"✔ leading zeros kept : {df['id'].tolist()}")
print(f"✔ price is numeric   : {df['price'].tolist()}")
print(f"✔ dates parsed       : {df['date'].dt.strftime('%Y-%m-%d').tolist()}")

# %% [markdown]
# > [!WARNING]
# > Without `date_format`, `15/03/2024` may be read as *March 15th* or fail
# > entirely depending on the pandas version — and `05/03/2024` would silently
# > become **May 3rd** instead of March 5th. Always state the format for
# > non-ISO dates.

# %% [markdown]
# ## Exercise 6.3 — Nested JSON

# %%
import json

order = {
    "order_id": 1,
    "customer": {"name": "Ana", "city": "SP"},
    "items": [{"sku": "A", "qty": 2}, {"sku": "B", "qty": 1}],
}

flat = pd.json_normalize(
    order,
    record_path="items",                       # one row per item
    meta=["order_id", ["customer", "name"], ["customer", "city"]],
)
flat.columns = [c.replace(".", "_") for c in flat.columns]
flat

# %% [markdown]
# ## Exercise 6.4 — Prove the column-pruning claim

# %%
import time

rng = np.random.default_rng(42)
n_rows, n_cols = 200_000, 50      # scaled down so the notebook stays quick

wide = pd.DataFrame(
    rng.standard_normal((n_rows, n_cols)),
    columns=[f"feature_{i:02d}" for i in range(n_cols)],
)

wide_csv = tmpdir / "wide.csv"
wide_parquet = tmpdir / "wide.parquet"

wide.to_csv(wide_csv, index=False)
wide.to_parquet(wide_parquet, index=False)

def time_read(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


t_csv = time_read(lambda: pd.read_csv(wide_csv, usecols=["feature_00"]))
t_parquet = time_read(lambda: pd.read_parquet(wide_parquet, columns=["feature_00"]))

print(f"file sizes     : csv {wide_csv.stat().st_size / 1024**2:6.1f} MB   "
      f"parquet {wide_parquet.stat().st_size / 1024**2:6.1f} MB")
print(f"read 1 of {n_cols} cols : csv {t_csv * 1000:7.1f} ms   "
      f"parquet {t_parquet * 1000:7.1f} ms")
print(f"\n→ parquet is {t_csv / t_parquet:.0f}× faster for this read")

# %% [markdown]
# **The explanation.** CSV is row-oriented: every value of every column is
# interleaved on each line, so the reader must scan and parse the whole file's
# bytes just to find the one column it wants. Parquet stores each column as a
# separate contiguous block with an index in the footer, so the reader seeks
# directly to that block and touches nothing else. The saving is proportional to
# how many columns you skip.

# %% [markdown]
# ## Exercise 6.5 — Safe writing with backup

# %%
from contextlib import contextmanager


@contextmanager
def atomic_write(path: Path, *, encoding: str = "utf-8", backup: bool = False):
    """Write atomically, optionally keeping the previous version as `<name>.bak`.

    Parameters
    ----------
    path : Path
        Final destination.
    encoding : str, default "utf-8"
        Text encoding.
    backup : bool, default False
        When True and `path` already exists, copy it to `<name>.bak` before
        replacing it.
    """
    temp_path = path.with_suffix(path.suffix + ".tmp")
    handle = temp_path.open("w", encoding=encoding)
    try:
        yield handle
    except Exception:
        handle.close()
        temp_path.unlink(missing_ok=True)
        raise
    else:
        handle.close()
        if backup and path.exists():
            # Take the backup only once we know the new content is complete.
            path.replace(path.with_suffix(path.suffix + ".bak"))
        temp_path.replace(path)


target = tmpdir / "report.txt"

with atomic_write(target) as f:
    f.write("version 1\n")

with atomic_write(target, backup=True) as f:
    f.write("version 2\n")

print(f"current : {target.read_text().strip()!r}")
print(f"backup  : {target.with_suffix('.txt.bak').read_text().strip()!r}")

# A failure leaves both the current file and the backup untouched.
try:
    with atomic_write(target, backup=True) as f:
        f.write("version 3\n")
        raise RuntimeError("crash")
except RuntimeError:
    pass

print(f"\nafter a crash — current : {target.read_text().strip()!r}")
print(f"                backup  : {target.with_suffix('.txt.bak').read_text().strip()!r}")

# %% [markdown]
# ---
# # Notebook 07 — Errors, Logging & Testing

# %% [markdown]
# ## Exercise 7.1 — Narrow the catch
#
# **Three distinct failures the bare `except:` hides:**
#
# 1. `FileNotFoundError` — the path is wrong. Recoverable by the caller, and it
#    needs a completely different response from a parsing problem.
# 2. `KeyError` / `pd.errors.ParserError` — the file exists but has the wrong
#    columns or is malformed. This is a schema problem for the data engineer.
# 3. `ValueError` from `astype(float)` — a specific cell contains junk. This is a
#    data-quality problem worth reporting per row.
#
# Worse, a bare `except:` also swallows `KeyboardInterrupt`, so the job cannot be
# stopped, and returning an empty DataFrame means every downstream aggregate
# quietly becomes zero.

# %%
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)


class SourceError(Exception):
    """Base class for load failures."""


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load a CSV and coerce its types, failing informatively.

    Raises
    ------
    SourceError
        If the file is missing, malformed, or lacks required columns.
    """
    log = logging.getLogger("loader")

    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SourceError(f"source file not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise SourceError(f"source file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise SourceError(f"malformed CSV: {path}") from exc

    required = {"date", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise SourceError(f"{path.name}: missing column(s) {sorted(missing)}")

    # `errors="coerce"` turns unparseable values into NaT/NaN rather than
    # raising — which is right ONLY because we then count and report them.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    n_bad_dates = int(df["date"].isna().sum())
    n_bad_amounts = int(df["amount"].isna().sum())
    if n_bad_dates or n_bad_amounts:
        log.warning("%s: %d unparseable date(s), %d unparseable amount(s)",
                    path.name, n_bad_dates, n_bad_amounts)

    return df


# Demonstrate each failure mode.
good_csv = tmpdir / "good.csv"
good_csv.write_text("date,amount\n2024-01-01,10.5\n2024-01-02,oops\n")

df = load_and_clean(good_csv)
print(f"\nloaded {len(df)} rows, {df['amount'].isna().sum()} amount(s) unusable")

for bad_path, label in [(tmpdir / "nope.csv", "missing file")]:
    try:
        load_and_clean(bad_path)
    except SourceError as exc:
        print(f"{label:<14} → SourceError: {exc}")

# %% [markdown]
# ## Exercise 7.2 — Custom exception hierarchy

# %%
class ETLError(Exception):
    """Base class for every failure in the ETL pipeline.

    Catch this to handle any pipeline failure uniformly; catch a subclass to
    respond to one specific kind.
    """


class SourceUnreachableError(ETLError):
    """The source system could not be contacted. Usually transient → retry."""

    def __init__(self, source: str, detail: str = "") -> None:
        self.source = source
        self.retryable = True
        super().__init__(f"cannot reach source {source!r}{': ' + detail if detail else ''}")


class SchemaMismatchError(ETLError):
    """The source's structure is not what we expect. Never retryable."""

    def __init__(self, source: str, missing: set[str], unexpected: set[str]) -> None:
        self.retryable = False
        parts = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if unexpected:
            parts.append(f"unexpected {sorted(unexpected)}")
        super().__init__(f"{source} schema mismatch: {'; '.join(parts)}")


class ValidationFailedError(ETLError):
    """The data arrived but violates business rules."""

    def __init__(self, rule: str, n_violations: int, n_total: int) -> None:
        self.retryable = False
        self.rate = n_violations / n_total if n_total else 0.0
        super().__init__(
            f"rule {rule!r} violated by {n_violations:,}/{n_total:,} "
            f"rows ({self.rate:.1%})"
        )


class DestinationWriteError(ETLError):
    """The transformed data could not be written. Often transient."""

    def __init__(self, destination: str, detail: str = "") -> None:
        self.retryable = True
        super().__init__(f"failed writing to {destination}{': ' + detail if detail else ''}")


# The payoff: one handler can route by retryability.
failures = [
    SourceUnreachableError("crm_api", "connection timed out"),
    SchemaMismatchError("orders.csv", {"amount"}, {"amt"}),
    ValidationFailedError("amount > 0", 1_204, 50_000),
    DestinationWriteError("s3://warehouse/orders", "access denied"),
]

for exc in failures:
    action = "retry" if getattr(exc, "retryable", False) else "alert a human"
    print(f"  {type(exc).__name__:<26} → {action}")
    print(f"      {exc}")

# %% [markdown]
# ## Exercise 7.3 — Instrument a function

# %%
def add_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add revenue columns, with production-grade logging."""
    log = logging.getLogger("features.revenue")

    log.info("Starting: %d rows, %d columns", len(df), df.shape[1])

    missing = {"amount", "quantity"} - set(df.columns)
    if missing:
        raise ValueError(f"missing required column(s): {sorted(missing)}")

    null_rate = df["amount"].isna().mean() if len(df) else 0.0
    if null_rate > 0.05:
        log.warning("%.1f%% of amounts are null (threshold 5%%) — "
                    "check the upstream export", null_rate * 100)

    out = df.copy()
    out["revenue"] = out["amount"] * out["quantity"]
    total = out["revenue"].sum()
    out["revenue_share"] = out["revenue"] / total if total else 0.0

    log.debug("Finished: shape %s, total revenue %.2f", out.shape, total)
    log.info("Added 2 feature column(s)")

    return out


clean = pd.DataFrame({"amount": [10.0, 20.0, 30.0], "quantity": [1, 2, 3]})
dirty = pd.DataFrame({"amount": [10.0, np.nan, np.nan], "quantity": [1, 2, 3]})

print("— clean input —")
_ = add_revenue_features(clean)
print("\n— input with 67% nulls —")
_ = add_revenue_features(dirty)

# %% [markdown]
# ## Exercise 7.4 — Write the tests first
#
# The tests define the contract, including the decisions the prose left open:
# what should a zero denominator produce? What about an empty series? Answering
# those *before* implementing is the entire value of test-first design.

# %%
TEST_SOURCE = '''
import numpy as np
import pandas as pd
import pytest


def pct_change(series):
    """Period-over-period percentage change.

    Returns
    -------
    Series
        First element is always NaN (no prior period). A zero denominator
        yields NaN rather than inf, because "infinite growth" is never a
        useful number to carry into a report.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series, dtype="float64")

    previous = series.shift(1)
    change = (series - previous) / previous
    # Division by zero gives ±inf in pandas; convert to NaN so downstream
    # aggregations (mean, sum) treat it as missing rather than exploding.
    return change.replace([np.inf, -np.inf], np.nan) * 100


class TestPctChange:
    def test_normal_series(self):
        s = pd.Series([100.0, 110.0, 99.0])
        result = pct_change(s)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(10.0)
        assert result.iloc[2] == pytest.approx(-10.0)

    def test_zero_denominator_gives_nan_not_inf(self):
        s = pd.Series([0.0, 50.0])
        result = pct_change(s)
        assert pd.isna(result.iloc[1])
        assert not np.isinf(result).any()

    def test_single_element_returns_single_nan(self):
        result = pct_change(pd.Series([42.0]))
        assert len(result) == 1
        assert pd.isna(result.iloc[0])

    def test_empty_series_returns_empty(self):
        result = pct_change(pd.Series([], dtype="float64"))
        assert len(result) == 0

    def test_nulls_propagate(self):
        s = pd.Series([100.0, np.nan, 120.0])
        result = pct_change(s)
        # A null input makes both the change INTO and OUT OF it undefined.
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

    def test_negative_values(self):
        s = pd.Series([-100.0, -50.0])
        # From -100 to -50 is a +50% change relative to the -100 base.
        assert pct_change(s).iloc[1] == pytest.approx(-50.0)
'''

import subprocess

with tempfile.TemporaryDirectory() as td:
    test_path = Path(td) / "test_pct_change.py"
    test_path.write_text(TEST_SOURCE)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-v",
         "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True,
    )

for line in completed.stdout.splitlines():
    if "::" in line or " passed" in line or " failed" in line:
        print(line.split("/")[-1] if "::" in line else line)

# %% [markdown]
# > [!NOTE]
# > `test_negative_values` documents a genuinely debatable choice. Going from
# > −100 to −50 is an *improvement*, yet the formula reports −50%. Percentage
# > change is ill-defined across sign changes — the right move is to state the
# > behaviour in a test rather than let each caller discover it independently.

# %% [markdown]
# ## Exercise 7.5 — Catch a real bug

# %%
TEST_SOURCE = '''
import pandas as pd
import pytest


def merge_customer_orders(customers, orders):
    """BUG: silently multiplies rows when `orders` has duplicate customer_ids."""
    return customers.merge(orders, on="customer_id", how="left")


def merge_customer_orders_safe(customers, orders):
    """Same merge, but it refuses to silently change the customer row count."""
    return customers.merge(
        orders,
        on="customer_id",
        how="left",
        validate="one_to_one",   # raises MergeError if either side has dup keys
    )


class TestMergeRowExplosion:
    def test_duplicate_order_ids_multiply_customer_rows(self):
        """Document the bug: 2 customers go in, 3 rows come out.

        This test PASSES against the buggy implementation — that is the point.
        It pins the broken behaviour down so nobody mistakes it for correct,
        and it fails the moment someone "fixes" the merge without thinking
        about what the right row count should be.
        """
        customers = pd.DataFrame({"customer_id": [1, 2], "name": ["Ana", "Bruno"]})
        orders = pd.DataFrame({"customer_id": [1, 1], "amount": [10.0, 20.0]})

        result = merge_customer_orders(customers, orders)

        assert len(result) == 3, "expected the silent row explosion"
        assert (result["customer_id"] == 1).sum() == 2, "Ana was duplicated"
        # The damage: any per-customer aggregate is now double-counted.
        assert result["name"].tolist().count("Ana") == 2

    def test_safe_version_raises_instead_of_exploding(self):
        """The fix: fail loudly at the merge rather than silently downstream."""
        customers = pd.DataFrame({"customer_id": [1, 2], "name": ["Ana", "Bruno"]})
        orders = pd.DataFrame({"customer_id": [1, 1], "amount": [10.0, 20.0]})

        with pytest.raises(pd.errors.MergeError):
            merge_customer_orders_safe(customers, orders)

    def test_row_count_preserved_when_keys_are_unique(self):
        customers = pd.DataFrame({"customer_id": [1, 2], "name": ["Ana", "Bruno"]})
        orders = pd.DataFrame({"customer_id": [1, 2], "amount": [10.0, 20.0]})

        result = merge_customer_orders_safe(customers, orders)
        assert len(result) == len(customers)
'''

with tempfile.TemporaryDirectory() as td:
    test_path = Path(td) / "test_merge.py"
    test_path.write_text(TEST_SOURCE)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-v",
         "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True,
    )

for line in completed.stdout.splitlines():
    if "::" in line or " passed" in line or " failed" in line:
        print(line.split("/")[-1] if "::" in line else line)

# %% [markdown]
# **The lesson.** `validate="one_to_one"` (or `"many_to_one"`) turns a silent,
# catastrophic row explosion into a loud `MergeError` at the point of the
# mistake. Add it to every `merge` where you believe the keys are unique — the
# cost is one keyword argument, and it has caught more production bugs than any
# other single pandas habit.
#
# The general rule: **assert the shape you expect after every join.**

# %%
# Clean up the temporary directory used by the exercises above.
import shutil

shutil.rmtree(tmpdir, ignore_errors=True)
print("temporary files removed")

# %% [markdown]
# ---
#
# ## Finished Module 01
#
# If you worked through these rather than reading them, you now have the
# foundations the rest of the course is built on.
#
# **Next:** Module 02 — NumPy & Numerical Computing.
