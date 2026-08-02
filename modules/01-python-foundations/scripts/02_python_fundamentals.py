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
# # 02 · Python Fundamentals
#
# **Module 01 — Python Foundations for Data Science**
#
# ---
#
# ## What you will learn
#
# 1. Python's object model — and why `b = a` does not copy anything.
# 2. Mutability, aliasing, and the silent bug it causes in every data pipeline.
# 3. Numeric types: arbitrary-precision integers, floating-point traps, and when to reach for `Decimal`.
# 4. Strings, f-strings, and text encoding — the source of most "it worked on my machine" data bugs.
# 5. Truthiness, `None`, and the comparison operators that catch people out.
# 6. Control flow, including `for...else` and structural pattern matching.
#
# **Estimated time:** 90 minutes
#
# > This is the most important notebook in the module. Nearly every subtle bug you
# > will write in pandas traces back to something on this page.

# %%
import sys

from dstoolkit import setup_plot_style

setup_plot_style()

print(f"Python {sys.version.split()[0]}")

# %% [markdown]
# ## 1. Everything is an object
#
# In Python there are no "primitive types" that behave differently from everything
# else. An integer, a function, a class, a module — all are objects, and every
# object has exactly three things:
#
# | Property | Accessed with | Can it change? |
# | :--- | :--- | :--- |
# | **Identity** — its address in memory | `id(obj)` | Never |
# | **Type** — what it is | `type(obj)` | Never |
# | **Value** — what it holds | the object itself | *Only if the type is mutable* |

# %%
x = 42

print(f"value    : {x}")
print(f"type     : {type(x)}")
print(f"identity : {id(x)}")     # a memory address; yours will differ

# Even a function is an object with the same three properties.
def greet(name: str) -> str:
    return f"Hello, {name}"

print(f"\nfunctions are objects too : {type(greet)}")
print(f"…so they can be assigned  : {greet('World')}")

# Assigning a function to a new name does not call it — it just binds
# another name to the same function object.
say_hi = greet
print(f"…and passed around        : {say_hi('again')}")

# %% [markdown]
# ## 2. Names are labels, not boxes
#
# This is the single most important concept in the notebook.
#
# If you learned C, Java, or C#, you probably picture a variable as a **box that
# holds a value**. Python does not work that way. In Python, a name is a **label
# tied to an object**, and assignment moves the label — it never copies the object.
#
# ![Variables in Python: names, not boxes](../../../assets/images/names_and_objects.png)

# %%
a = [1, 2, 3]
b = a               # does NOT copy the list — binds a second label to the SAME list

print(f"a          : {a}")
print(f"b          : {b}")
print(f"same object? {a is b}")          # `is` compares identity
print(f"id(a)      : {id(a)}")
print(f"id(b)      : {id(b)}")

# %% [markdown]
# ### `==` versus `is`
#
# These ask genuinely different questions, and confusing them is a classic bug:
#
# - `a == b` — "do these have the same **value**?"
# - `a is b` — "are these the **same object** in memory?"

# %%
list_1 = [1, 2, 3]
list_2 = [1, 2, 3]      # a separate list that happens to hold equal values
list_3 = list_1         # another label for list_1

print(f"list_1 == list_2 : {list_1 == list_2}   ← same contents")
print(f"list_1 is list_2 : {list_1 is list_2}   ← but different objects")
print(f"list_1 is list_3 : {list_1 is list_3}   ← the same object")

# %% [markdown]
# > [!WARNING]
# > **Use `is` only for `None`, `True`, and `False`.** For everything else, use `==`.
# >
# > The reason is that CPython caches ("interns") some small immutable objects,
# > so `is` can appear to work and then break on a larger value. This is an
# > implementation detail you must never rely on:

# %%
# Small integers (-5 to 256) are pre-allocated and shared by CPython…
small_a, small_b = 100, 100
print(f"100 is 100     : {small_a is small_b}")

# …but larger ones are not.
big_a, big_b = 1000, 1000
print(f"1000 is 1000   : {big_a is big_b}")

# Both are equal, which is what you actually care about.
print(f"1000 == 1000   : {big_a == big_b}")
print("\nLesson: compare values with ==, identity with is (and only for None/True/False).")

# %% [markdown]
# ## 3. Mutability — where the bugs live
#
# Every Python type is either **mutable** (can be changed in place) or **immutable**
# (cannot). This single property determines whether `b = a` is harmless or a
# time bomb.
#
# | Immutable | Mutable |
# | :--- | :--- |
# | `int`, `float`, `bool`, `complex` | `list` |
# | `str` | `dict` |
# | `tuple` | `set` |
# | `frozenset` | `bytearray` |
# | `bytes` | **`numpy.ndarray`**, **`pandas.DataFrame`** |
#
# ![Mutable versus immutable](../../../assets/images/mutable_vs_immutable.png)

# %%
# --- Immutable: safe -------------------------------------------------------
a = 10
b = a
b += 1          # cannot modify 10 in place → creates a NEW int, rebinds b

print("IMMUTABLE (int)")
print(f"  a = {a}   ← untouched")
print(f"  b = {b}")

# --- Mutable: dangerous ----------------------------------------------------
x = [1, 2]
y = x
y.append(3)     # modifies the list IN PLACE → both names see the change

print("\nMUTABLE (list)")
print(f"  x = {x}   ← changed, even though we only touched y!")
print(f"  y = {y}")

# %% [markdown]
# ### Why this matters in real data work
#
# The abstract example above looks avoidable. It is not, because the aliasing is
# usually hidden behind a function call.

# %%
def add_engineered_feature(records: list[dict]) -> list[dict]:
    """Add a `total` field to each record.

    BUG: this mutates the caller's data as a side effect.
    """
    for record in records:
        record["total"] = record["price"] * record["quantity"]
    return records


raw_orders = [
    {"price": 10.0, "quantity": 3},
    {"price": 5.5, "quantity": 2},
]

print(f"before : {raw_orders}")

enriched = add_engineered_feature(raw_orders)

print(f"after  : {raw_orders}")
print("\n↑ `raw_orders` was silently modified. Our 'immutable raw data' is gone.")
print(f"Worse, they are the same object: {enriched is raw_orders}")

# %% [markdown]
# ### The fix: copy before you mutate
#
# Python offers two kinds of copy, and the difference matters as soon as your data
# is nested — which, for real data, it always is.
#
# | Copy | Syntax | Copies the container? | Copies what is inside? |
# | :--- | :--- | :-: | :-: |
# | **Shallow** | `list(x)`, `x[:]`, `copy.copy(x)` | ✔ | ✘ — shares the inner objects |
# | **Deep** | `copy.deepcopy(x)` | ✔ | ✔ — recursively copies everything |

# %%
import copy

original = [{"price": 10.0, "quantity": 3}]

# --- Shallow copy: a new outer list, but the SAME inner dicts --------------
shallow = copy.copy(original)
shallow[0]["price"] = 999.0

print("After mutating a dict inside the SHALLOW copy:")
print(f"  original : {original}   ← also changed!")

# --- Deep copy: everything duplicated, all the way down -------------------
original = [{"price": 10.0, "quantity": 3}]      # reset
deep = copy.deepcopy(original)
deep[0]["price"] = 999.0

print("\nAfter mutating a dict inside the DEEP copy:")
print(f"  original : {original}   ← protected")
print(f"  deep     : {deep}")

# %% [markdown]
# ### The professional pattern
#
# Rather than copying defensively everywhere, prefer functions that **do not mutate
# their inputs at all**. Build and return something new:

# %%
def add_total(records: list[dict]) -> list[dict]:
    """Return new records with a `total` field added.

    The input is never modified — the caller's data stays trustworthy.
    """
    # `{**record, "total": ...}` builds a new dict from the old one plus a key.
    return [
        {**record, "total": record["price"] * record["quantity"]}
        for record in records
    ]


raw_orders = [{"price": 10.0, "quantity": 3}, {"price": 5.5, "quantity": 2}]
enriched = add_total(raw_orders)

print(f"raw_orders : {raw_orders}   ← untouched")
print(f"enriched   : {enriched}")

# %% [markdown]
# > [!TIP]
# > This is exactly why most pandas methods return a new object rather than editing
# > in place, and why `inplace=True` is discouraged in modern pandas. You will meet
# > this again in Module 03.

# %% [markdown]
# ## 4. Numbers
#
# ### Integers have no maximum
#
# Unlike most languages, Python integers grow to whatever size they need. There is
# no overflow, no `int64` ceiling.

# %%
big = 2**200
print(f"2**200 = {big}")
print(f"digits : {len(str(big))}")

# Handy for readability: underscores are ignored in numeric literals.
population = 8_100_000_000
print(f"\nworld population : {population:,}")

# %% [markdown]
# > [!WARNING]
# > **NumPy integers do NOT behave this way.** NumPy uses fixed-width types for
# > speed, so they overflow silently. This is a real source of wrong numbers in
# > production pipelines, and Module 02 covers it in detail.

# %%
import numpy as np

py_int = 2**62
np_int = np.int64(2**62)

print(f"int64 max : {np.iinfo(np.int64).max:,}")
print(f"\nPython : {py_int * 2:,}   ← grows as needed")

# Doubling 2**62 lands on 2**63, one past the int64 ceiling. Instead of raising,
# the value wraps around to the most negative representable number.
with np.errstate(over="ignore"):
    print(f"NumPy  : {np_int * 2:,}   ← silently wrapped to a NEGATIVE number")

print("\nA revenue total that silently turns negative is the kind of bug that")
print("reaches a dashboard before anyone notices. Module 02 covers dtype choice.")

# %% [markdown]
# ### Floats are approximations
#
# Floating-point numbers are stored in binary (IEEE 754, 64-bit). Just as 1/3
# cannot be written exactly in decimal, **0.1 cannot be written exactly in binary**.
# Every float operation therefore carries a tiny representation error.

# %%
result = 0.1 + 0.2

print(f"0.1 + 0.2            = {result}")
print(f"0.1 + 0.2 == 0.3     → {result == 0.3}")
print(f"\nwhat 0.1 really is   : {0.1:.20f}")
print(f"what 0.3 really is   : {0.3:.20f}")
print(f"the gap              : {abs(result - 0.3):.2e}")

# %% [markdown]
# #### How to compare floats correctly
#
# Never use `==` on computed floats. Compare within a tolerance instead.

# %%
import math

a, b = 0.1 + 0.2, 0.3

# Standard library: math.isclose (relative + absolute tolerance).
print(f"math.isclose(a, b)   : {math.isclose(a, b)}")

# NumPy equivalent, and its array-wide version — this is what you will use daily.
print(f"np.isclose(a, b)     : {np.isclose(a, b)}")

arr_1 = np.array([0.1 + 0.2, 1.0 + 2.0])
arr_2 = np.array([0.3, 3.0])
print(f"np.allclose(arrays)  : {np.allclose(arr_1, arr_2)}")

# %% [markdown]
# #### When exactness is mandatory: money
#
# For financial calculations, accumulated float error is unacceptable — regulators
# and auditors do not accept "it was off by 1e-16 per row, times 40 million rows".
# Use `decimal.Decimal`, which does exact base-10 arithmetic.

# %%
from decimal import Decimal, getcontext

getcontext().prec = 28

# Note: build Decimals from STRINGS. Decimal(0.1) would faithfully copy the
# float's error, defeating the point entirely.
price = Decimal("0.10")
tax = Decimal("0.20")

print(f"float   : {0.10 + 0.20!r}")
print(f"Decimal : {price + tax}")
print(f"exact?  : {price + tax == Decimal('0.30')}")

print(f"\nthe trap: Decimal(0.1) = {Decimal(0.1)}")

# %% [markdown]
# ### Integer division, floor division, and the modulo of negatives

# %%
print(f"7 / 2    = {7 / 2}      ← true division, always a float")
print(f"7 // 2   = {7 // 2}      ← floor division")
print(f"7 % 2    = {7 % 2}      ← remainder")
print(f"divmod(7, 2) = {divmod(7, 2)}")

# Python floors toward negative infinity, unlike C/Java which truncate toward 0.
# This surprises people, but it makes the modulo result's sign predictable —
# handy for cyclical features such as hour-of-day or day-of-week.
print(f"\n-7 // 2  = {-7 // 2}     ← floors DOWN, not toward zero")
print(f"-7 % 2   = {-7 % 2}      ← always has the sign of the divisor")
print(f"\nuseful:  hour 26 wraps to {26 % 24}, hour -3 wraps to {-3 % 24}")

# %% [markdown]
# ## 5. Strings
#
# Strings are **immutable sequences of Unicode characters**. Every "modification"
# actually produces a new string.

# %%
s = "data science"

print(f"original      : {s}")
print(f"upper()       : {s.upper()}      ← a NEW string")
print(f"original now  : {s}      ← unchanged")

# Indexing and slicing work as on any sequence: [start:stop:step], stop excluded.
print(f"\ns[0]          : {s[0]}")
print(f"s[-1]         : {s[-1]}      ← negative indexes count from the end")
print(f"s[0:4]        : {s[0:4]}")
print(f"s[::-1]       : {s[::-1]}      ← reversed")

# %% [markdown]
# ### The string methods you will actually use
#
# Cleaning text columns is a daily task. These handle most of it:

# %%
messy = "  Sao Paulo, BR \n"

print(f"repr(messy)        : {messy!r}")
print(f"strip()            : {messy.strip()!r}")
print(f"lower()            : {messy.strip().lower()!r}")
print(f"replace()          : {messy.strip().replace(',', ' —')!r}")
print(f"split(',')         : {messy.strip().split(',')}")
print(f"startswith('Sao')  : {messy.strip().startswith('Sao')}")

# `join` is the efficient way to concatenate many strings. Building a string with
# `+=` in a loop is O(n²) because each step allocates a whole new string.
parts = ["2024", "01", "15"]
print(f"\n'-'.join(parts)    : {'-'.join(parts)!r}")

# %% [markdown]
# ### f-strings: formatting you should use everywhere
#
# f-strings (Python 3.6+) are the fastest and most readable way to build strings.
# The format mini-language after `:` is worth learning — it removes a great deal of
# manual rounding code.

# %%
value = 1234567.8915
ratio = 0.0734
name = "revenue"

print(f"{value:,.2f}          ← thousands separator, 2 decimals")
print(f"{ratio:.1%}              ← as a percentage")
print(f"{value:.3e}         ← scientific notation")
print(f"{name:>15}    ← right-aligned in 15 chars")
print(f"{name:<15}|   ← left-aligned")
print(f"{name:^15}|   ← centred")
print(f"{42:05d}             ← zero-padded")

# `=` prints the expression AND its value — superb for debugging.
n_rows = 1500
print(f"\n{n_rows=}")
print(f"{value * 2=:,.2f}")

# %% [markdown]
# ### Encoding: the source of half your data-loading errors
#
# A `str` is abstract text. To write it to disk or send it over a network it must
# be **encoded** into `bytes`. UTF-8 is the correct default; legacy files are often
# Latin-1 or Windows-1252, and guessing wrong corrupts your data.

# %%
text = "café"

utf8_bytes = text.encode("utf-8")
latin1_bytes = text.encode("latin-1")

print(f"text            : {text}")
print(f"as UTF-8        : {utf8_bytes}   ({len(utf8_bytes)} bytes)")
print(f"as Latin-1      : {latin1_bytes}   ({len(latin1_bytes)} bytes)")

# Decoding with the wrong codec produces "mojibake" — or an outright error.
print(f"\nUTF-8 bytes read as Latin-1 : {utf8_bytes.decode('latin-1')}")

try:
    latin1_bytes.decode("utf-8")
except UnicodeDecodeError as exc:
    print(f"Latin-1 bytes read as UTF-8 : UnicodeDecodeError — {exc.reason}")

# %% [markdown]
# > [!TIP]
# > When `pd.read_csv` raises a `UnicodeDecodeError`, the file is not broken —
# > it is simply not UTF-8. Try `encoding="latin-1"` or `encoding="cp1252"`.
# > Notebook 06 covers this properly.

# %% [markdown]
# ## 6. Booleans, `None`, and truthiness
#
# Every Python object can be evaluated in a boolean context. The rules are
# short and worth memorising, because they make idiomatic Python much terser.

# %%
# These are all "falsy":
falsy = [False, None, 0, 0.0, "", [], {}, set(), ()]

for item in falsy:
    assert not item, f"{item!r} should be falsy"

print("Falsy  : False, None, 0, 0.0, '', [], {}, set(), ()")
print("Truthy : everything else — including 'False', [0], and -1")

# Two that surprise people:
print(f"\nbool('False') : {bool('False')}   ← a non-empty string is truthy")
print(f"bool([0])     : {bool([0])}   ← a non-empty list is truthy")
print(f"bool(-1)      : {bool(-1)}   ← only ZERO is falsy, not 'negative'")

# %% [markdown]
# ### Idiomatic emptiness checks

# %%
items: list[int] = []

# Idiomatic — reads naturally and works for any container.
if not items:
    print("✔  `if not items:`        — preferred")

# Works, but noisier.
if len(items) == 0:
    print("○  `if len(items) == 0:`  — unnecessary")

# %% [markdown]
# > [!WARNING]
# > This idiom is one of the few places where **pandas and NumPy deliberately
# > break from plain Python**. Calling `bool()` on an array with more than one
# > element is ambiguous — should it mean "any" or "all"? — so they raise instead.

# %%
arr = np.array([1, 2, 3])

try:
    if arr:
        pass
except ValueError as exc:
    print(f"ValueError: {exc}")

print(f"\nBe explicit instead:")
print(f"  arr.any()   : {arr.any()}")
print(f"  arr.all()   : {arr.all()}")
print(f"  arr.size    : {arr.size}   ← use this to test for emptiness")

# %% [markdown]
# ### `None` is not zero, and not missing-data
#
# `None` is Python's null object. In data science you will meet **three different
# kinds of "missing"**, and they are not interchangeable:

# %%
import pandas as pd

print(f"None        : {None!r}          ← Python's null; type {type(None).__name__}")
print(f"np.nan      : {np.nan!r}           ← IEEE 754 'not a number', a float")
print(f"pd.NaT      : {pd.NaT!r}           ← pandas' missing timestamp")

# NaN's defining, deeply surprising property: it is not equal to itself.
print(f"\nnp.nan == np.nan : {np.nan == np.nan}   ← ALWAYS False")
print(f"np.isnan(np.nan) : {np.isnan(np.nan)}    ← use this instead")
print(f"None == None     : {None == None}     ← None does compare equal")

# %% [markdown]
# > [!WARNING]
# > `nan != nan` is why `df[df["col"] == np.nan]` silently returns **zero rows**
# > instead of your missing values. The correct call is `df[df["col"].isna()]`.
# > This mistake makes it into production regularly.

# %% [markdown]
# ## 7. Control flow
#
# ### Conditionals and chained comparisons

# %%
def classify_spend(amount: float) -> str:
    """Bucket a customer by spend. Demonstrates chained comparisons."""
    # Python allows `a <= x < b`, which reads like maths and evaluates
    # `x` only once. Most languages require `x >= a and x < b`.
    if amount < 0:
        return "invalid"
    elif 0 <= amount < 50:
        return "low"
    elif 50 <= amount < 200:
        return "medium"
    else:
        return "high"


for amount in [-5, 12.5, 99.0, 5000.0]:
    print(f"{amount:>8.2f} → {classify_spend(amount)}")

# %% [markdown]
# ### Loops: iterate over objects, not indexes
#
# The most common giveaway of a non-Python background is looping over `range(len(x))`.

# %%
channels = ["organic", "paid", "referral"]

# ✘ C-style — works, but noisy and error-prone.
print("Non-idiomatic:")
for i in range(len(channels)):
    print(f"  {i}: {channels[i]}")

# ✔ Idiomatic — iterate the object directly.
print("\nIdiomatic:")
for channel in channels:
    print(f"  {channel}")

# ✔ When you genuinely need the index, use enumerate.
print("\nWith index (enumerate):")
for i, channel in enumerate(channels, start=1):
    print(f"  {i}: {channel}")

# ✔ To walk two sequences together, use zip.
# `strict=True` (Python 3.10+) raises if the lengths differ, rather than
# silently truncating — which has hidden many data-alignment bugs.
revenues = [45_000, 32_000, 12_000]
print("\nTwo sequences (zip):")
for channel, revenue in zip(channels, revenues, strict=True):
    print(f"  {channel:<10} ${revenue:>7,}")

# %% [markdown]
# ### `break`, `continue`, and the `for...else` clause
#
# `for...else` is a genuinely useful Python feature that almost nobody knows: the
# `else` block runs **only if the loop finished without hitting `break`**.

# %%
def find_first_negative(values: list[float]) -> float | None:
    """Return the first negative value, or None if the data is clean."""
    for value in values:
        if value < 0:
            print(f"  found a negative value: {value}")
            break
    else:
        # Reached only when `break` never fired.
        print("  no negative values — data passes validation")
        return None
    return value


print("Checking [10, 20, 30]:")
find_first_negative([10, 20, 30])

print("\nChecking [10, -5, 30]:")
find_first_negative([10, -5, 30])

# %% [markdown]
# ### Structural pattern matching (`match`)
#
# Added in Python 3.10, `match` is far more than a `switch`: it destructures data
# as it matches. It is excellent for handling heterogeneous API responses.

# %%
def describe_response(response: dict) -> str:
    """Classify an API response by shape, not just by value."""
    match response:
        # Match a specific key/value AND bind another key in one step.
        case {"status": "ok", "data": [*items]}:
            return f"success with {len(items)} records"
        case {"status": "ok"}:
            return "success, no data returned"
        case {"status": "error", "code": int(code)} if code >= 500:
            return f"server error ({code}) — safe to retry"
        case {"status": "error", "code": int(code)}:
            return f"client error ({code}) — do not retry"
        case _:
            return "unrecognised response shape"


examples = [
    {"status": "ok", "data": [1, 2, 3]},
    {"status": "ok"},
    {"status": "error", "code": 503},
    {"status": "error", "code": 404},
    {"unexpected": True},
]

for example in examples:
    print(f"{str(example):<42} → {describe_response(example)}")

# %% [markdown]
# ### The walrus operator `:=`
#
# Assigns *and* returns a value in one expression. Use it when you would otherwise
# compute something twice or add a throwaway line.

# %%
readings = [12.0, 45.5, 3.2, 88.1, 0.5]

# Without the walrus: you must either call the function twice, or add a line.
for reading in readings:
    if (scaled := reading * 1.8 + 32) > 100:
        print(f"  {reading:>5.1f} → {scaled:>6.1f} ⚠ above threshold")

# %% [markdown]
# ## 8. Exercises
#
# ### Exercise 2.1 — Predict before you run
#
# For each snippet, write down the output **before** executing it. Then check.
#
# ```python
# # (a)
# a = [1, 2, 3]
# b = a
# b = b + [4]          # note: NOT b += [4]
# print(a)
#
# # (b)
# a = [1, 2, 3]
# b = a
# b += [4]
# print(a)
#
# # (c)
# t = ([1, 2], 3)
# t[0].append(99)
# print(t)
# ```
#
# Explain *why* (a) and (b) differ, and why (c) works despite tuples being immutable.
#
# ### Exercise 2.2 — A float comparison bug
#
# This function is wrong. Find the bug and fix it.
#
# ```python
# def is_fully_allocated(weights: list[float]) -> bool:
#     """Return True when portfolio weights sum to 1."""
#     return sum(weights) == 1.0
#
# is_fully_allocated([0.1, 0.2, 0.3, 0.4])   # returns False!
# ```
#
# ### Exercise 2.3 — Safe enrichment
#
# Write `add_discount(orders, pct)` that returns new order dicts with a
# `discounted_price` key. Prove with an assertion that the input list is unmodified.
#
# ### Exercise 2.4 — Text cleaning
#
# Write `clean_city(raw)` that turns `"  sÃO   paulo\n"` into `"Sao Paulo"`:
# strip whitespace, collapse internal runs of spaces, and title-case the result.
#
# *Hint:* `" ".join(raw.split())` collapses whitespace in one step.
#
# ### Exercise 2.5 — Missing-value audit
#
# Given `values = [1.0, np.nan, 3.0, None, 5.0]`, count how many entries are
# "missing". Then explain why `values.count(np.nan)` gives the wrong answer.

# %% [markdown]
# ## Key takeaways
#
# - **Names are labels bound to objects.** Assignment never copies.
# - **Mutability decides everything.** Functions that mutate their arguments cause
#   bugs that are invisible at the call site — return new objects instead.
# - `copy.copy` is shallow; nested data needs `copy.deepcopy`.
# - **Floats are approximate.** Compare with `math.isclose` / `np.isclose`;
#   use `Decimal` for money.
# - Python ints are unbounded; **NumPy ints overflow silently**.
# - Text must be encoded to bytes. UTF-8 is the default; a `UnicodeDecodeError`
#   means "wrong codec", not "broken file".
# - **`nan != nan`** — always test missing values with `.isna()` / `np.isnan`.
# - Iterate over objects, not indexes: `enumerate`, `zip(..., strict=True)`.
#
# ---
#
# **Next:** [03 · Data Structures](03_data_structures.ipynb) — and why choosing a
# `set` over a `list` can turn 40 minutes into 2 seconds.
