# Numbers With Limits
@subtitle Dtypes, silent overflow, and the precision you can actually trust
@time About 75 minutes
@notebook 03_dtypes_and_precision.ipynb
@learn
- The dtype zoo, and how to choose deliberately rather than by accident.
- **Silent integer overflow**, Module 01's warning paid in full.
- Type promotion, including what changed in NumPy 2.0.
- `float32` versus `float64`: the memory, speed and accuracy trade-off, quantified.
- Machine epsilon and catastrophic cancellation.
- `NaN` and `inf`: how they spread, and how to control the warnings.
- Cutting a dataset's memory dramatically with correct, *verified* downcasting.
@end

@lead Python's own integers are generous: they grow as big as you need. NumPy's are thrifty: each one gets a fixed number of bytes, decided in advance. That thrift is the source of NumPy's speed, and of its sharpest edge. This chapter is about knowing exactly where the edge is.

## The dtype zoo

NumPy uses **fixed-width** types. Each has a hard minimum and maximum:

| dtype | bytes | range |
| :--- | :--- | :--- |
| `int8` | 1 | −128 to 127 |
| `int16` | 2 | −32,768 to 32,767 |
| `int32` | 4 | about ±2.1 billion |
| `int64` | 8 | about ±9.2 × 10¹⁸ |
| `uint8` … `uint64` | 1 … 8 | 0 to twice the signed maximum, roughly |

```python
np.iinfo(np.int32).max      # 2147483647
np.finfo(np.float32).precision, np.finfo(np.float64).precision   # 6, 15
```

![Precision is a budget: the number of decimal digits each float type can be trusted with.](assets/images/book/m02_float_digits.png){width=52}

`precision` is the number of **decimal digits you can trust**. `float32` gives about 7 (NumPy reports a conservative 6); `float64` about 15. That single number decides whether a computation is trustworthy, and it's worth memorising.

## Silent integer overflow

A fixed-width integer is an odometer. Push it past its largest value and it rolls over to the smallest:

![int8 holds 256 values. One step past 127 lands on −128, without a word.](assets/images/book/m02_int_wrap.png){width=100}

```python
arr = np.array([2**62], dtype=np.int64)
(arr * 2)[0]        # -9223372036854775808: wrapped to the most negative value
```

### The dangerous part: arrays wrap without a warning

```python
np.array([127], dtype=np.int8) + 1     # array([-128])   no warning at all
np.int8(127) + 1                       # -128, WITH a RuntimeWarning
```

The scalar path warns; the array path doesn't. That inconsistency makes the problem easy to miss in exploration and lethal in production.

> [!WARNING]
> **Array arithmetic overflows silently.** No exception, no warning, just a
> wrong number flowing into every downstream aggregate. In a revenue pipeline
> it shows up as a negative total; in an ID column it shows up as collisions.
> Neither is caught by a test that only checks for nulls.

> [!STORY] Thirty-seven seconds
> On 4 June 1996 the first Ariane 5 rocket lifted off from Kourou, French
> Guiana. About thirty-seven seconds later it veered off course and
> self-destructed. The inquiry traced the failure to a few lines of code
> reused from Ariane 4: a value related to the rocket's **horizontal
> velocity**, held as a 64-bit float, was converted into a **16-bit signed
> integer**. Ariane 5 was faster than Ariane 4 had ever been, the number no
> longer fitted, and the conversion raised an exception that shut down both
> inertial reference units in turn. The code had been correct for the old
> rocket. Nobody had re-checked that the *values* still fitted the *type*,
> exactly the check this chapter will teach you to automate.

### Where it bites, and where it doesn't

The intuitive answer here is wrong, so let's get it exactly right. Store daily page views as `int32` to save memory: 400 days of 8 million views each, a true total of 3.2 billion, well past the `int32` ceiling.

```python
daily_views = np.full(400, 8_000_000, dtype=np.int32)
daily_views.sum()                   # 3,200,000,000 ✔ (and its dtype is int64!)
daily_views.sum(dtype=np.int32)     # wrapped ✘
(daily_views * 400)[0]              # wrapped ✘
```

**Reductions are safer than you'd expect; element-wise arithmetic is not.** `sum`, `mean`, `prod` and `cumsum` promote their *accumulator* to the platform's default integer when the input is narrower, so `int32_array.sum()` is safe out of the box. Element-wise operations (`*`, `+`, `**`) get no such protection: the result keeps the input dtype and wraps in silence. **That's where the bug lives**: not in the aggregation you were worried about, but in the innocent `views * price` two lines earlier.

The defence for element-wise work is to widen *first*, on purpose:

```python
views.astype(np.int64) * 2          # correct
```

### Detecting overflow instead of hoping

`np.errstate` controls how NumPy reports numeric errors. Setting it to `"raise"` turns silence into an exception:

```python
with np.errstate(over="raise"):
    np.multiply(values, 4, dtype=np.int32)   # FloatingPointError
```

## Type promotion

When operands differ, NumPy promotes to a type that can hold both. NumPy 2.0 changed the rules materially (the change is called **NEP 50**), so examples written before 2023 may not match what you see.

| Expression | Result dtype |
| :--- | :--- |
| `int32` array + `int64` array | `int64` |
| `int32` array + `float32` array | `float64` |
| `float32` array + `float64` array | `float64` |
| `int` array + `bool` array | `int64` |
| `float32` array + Python `float` | **`float32`** |
| `int32` array + Python `int` | **`int32`** |

The last two rows are the NEP 50 change: **a Python scalar no longer upcasts an array.** `float32_array + 1.0` stays `float32`. Before 2.0 it became `float64`, silently doubling memory in deep-learning pipelines. "The array wins" is more predictable, and usually what you wanted.

You can ask what two types will produce without running anything: `np.promote_types(np.int64, np.float32)` is `float64`, because neither type can hold every value of the other.

### Division always produces a float

```python
a = np.array([7, 8, 9], dtype=np.int32)
a / 2       # float64: true division, and the memory just doubled
a // 2      # int32:   floor division keeps the integer type
```

## `float32` versus `float64`

`float64` is NumPy's default and the right choice most of the time. `float32` halves the memory and often runs faster, at the cost of about eight decimal digits. Deep learning uses `float32` (and even `float16`) everywhere; statistics usually shouldn't.

### Absorption

The starkest failure: add a small number to a big one, and the small one vanishes, because the result can't represent the difference.

```python
np.float32(1e8) + np.float32(1) == np.float32(1e8)   # True: the 1 is GONE
np.spacing(np.float32(1e8))                          # 8.0
```

Near 10⁸, consecutive `float32` values are 8 apart. There is simply no `float32` for 100,000,001.

### Accumulation

Sum ten million copies of 0.1 and compare with the true million:

- `float64`: off by a few billionths.
- `float32` via `.sum()`: off by about 0.1. Far better than you might fear, because **NumPy's `.sum()` uses pairwise summation**, which makes the error grow like √n instead of n. You're benefiting from a good algorithm, not from `float32` being accurate.

Write the naive running total yourself and that protection disappears:

```python
naive = np.float32(0.0)
for _ in range(1_000_000):
    naive = np.float32(naive + np.float32(0.1))
# about 1% off. From a loop that looks completely reasonable.
```

![Measured: relative error of a running float32 total versus NumPy's pairwise sum and float64, as more 0.1s are added. Both axes are logarithmic.](assets/images/book/m02_accumulation.png){width=100}

When you must sum `float32` data, widen the accumulator: `values_32.sum(dtype=np.float64)`.

> [!STORY] A third of a second, after a hundred hours
> During the Gulf War, a Patriot missile battery at Dhahran, Saudi Arabia,
> had been running continuously for about 100 hours when, on 25 February
> 1991, it failed to intercept an incoming Scud missile. The Scud hit a US
> Army barracks, killing 28 soldiers. The US General Accounting Office (today's
> Government Accountability Office) traced the cause to the system's clock: time was counted in tenths of a
> second, and **0.1 has no exact binary representation**. Stored in a
> 24-bit register, each tick was off by a tiny amount, and after 100 hours of
> accumulation the error was about 0.34 seconds. At the speed of a Scud, that
> shifted the tracking window by more than half a kilometre. The arithmetic
> on this page, small error times huge repetition, is not academic.

## Epsilon and catastrophic cancellation

**Machine epsilon** is the smallest number that still changes 1.0 when you add it. It sets the resolution of the whole type: about 1.2 × 10⁻⁷ for `float32` and 2.2 × 10⁻¹⁶ for `float64`. Add *half* of epsilon to 1.0 and it's lost entirely.

### Catastrophic cancellation

Subtracting two nearly equal numbers wipes out precision: the leading digits cancel, and what's left is mostly rounding error. The classic victim is the textbook variance formula:

```python
data = np.array([1e8 + 1, 1e8 + 2, 1e8 + 3, 1e8 + 4])
(data**2).mean() - data.mean()**2     # E[x²] − E[x]²: nonsense
((data - data.mean())**2).mean()      # two-pass formula: 1.25 ✔
np.var(data)                          # 1.25 ✔
```

The naive formula subtracts two numbers around 10¹⁶ to get an answer around 1. Sixteen significant digits cancel, and `float64` only has about fifteen, so the result is pure noise. That's why you call `np.var` instead of typing the formula: **the library implements the stable version.** The lesson generalises: for standard numerical operations, prefer the library function to the textbook expression.

> [!NOTE]
> The rules that make all of this predictable, on every computer, come from
> the **IEEE 754** standard of 1985, largely shaped by the Berkeley
> mathematician William Kahan, often called "the father of floating point".
> Before it, different machines rounded differently and the same program
> could give different answers on different hardware. Kahan received the
> Turing Award in 1989 for this work.

## `NaN` and `inf`

```python
special = np.array([1.0, np.nan, np.inf, -np.inf, 0.0])
np.isnan(special)       # [F  T  F  F  F]
np.isinf(special)       # [F  F  T  T  F]
np.isfinite(special)    # [T  F  F  F  T]  ← usually the one you want
```

### NaN spreads through everything

One NaN poisons a sum or a mean. The `nan*` family skips missing values instead: `np.nansum`, `np.nanmean`, `np.nanstd`, `np.nanmax`.

> [!WARNING]
> The `nan*` functions skip silently, which is convenient and sometimes
> wrong. `np.nanmean([1, nan, nan, nan])` is `1.0`, computed from a single
> observation, and it won't tell you. **Always report the count alongside**:
> `np.isfinite(x).sum()`.

### Controlling the warnings

Division by zero warns by default and produces `inf`, `-inf` or `nan`. While debugging, make it raise instead; it's far easier than hunting a NaN backwards through ten transformations. For deliberate division with a guard, `np.divide(..., where=...)` is the clean pattern:

```python
with np.errstate(divide="raise"):
    1.0 / a                                    # FloatingPointError

safe = np.divide(1.0, a, out=np.full_like(a, np.nan), where=(a != 0))
```

## Cutting memory with correct downcasting

Choosing the narrowest **safe** dtype is one of the highest-leverage optimisations in data work, but "safe" must be **verified**, not guessed. The notebook's `safe_downcast` checks each integer column's actual minimum and maximum against each candidate type's limits, and demotes a float column to `float32` only if the round trip is lossless for *this* data:

```python
if np.issubdtype(arr.dtype, np.integer):
    low, high = arr.min(), arr.max()
    candidates = ([np.uint8, np.uint16, np.uint32] if low >= 0
                  else [np.int8, np.int16, np.int32])
    for candidate in candidates:
        info = np.iinfo(candidate)
        if low >= info.min and high <= info.max:
            return arr.astype(candidate)
```

On a million-row dataset (user IDs, ages, session counts, scores, a boolean), this removes the large majority of the memory. The notebook then **checks every column survived** before celebrating. A downcast that loses data isn't an optimisation.

> [!WARNING]
> **Downcast IDs with care.** An `int32` holds up to about 2.1 billion: fine
> for a user ID today, a silent collision the day the table grows. When a
> column is a *key* rather than a *measurement*, keep it wide, or store it as
> a string.

## `datetime64` and `timedelta64`

```python
dates = np.array(["2024-01-15", "2024-06-30", "2024-12-31"], dtype="datetime64[D]")
np.diff(dates)                                   # gaps in days
dates + np.timedelta64(7, "D")                   # one week later
(dates - np.datetime64("2024-01-01")).astype(int)   # days since New Year
```

The `[D]` is the resolution: day. Change it with `.astype("datetime64[h]")`.

> [!NOTE]
> `datetime64` has no time zone; it represents a naive instant. For anything
> involving zones or calendars, pandas' `Timestamp` (Module 03) is the right
> tool: it wraps `datetime64` and adds the missing semantics.

## Try it yourself

Exercises 3.1 to 3.5 in notebook 03; solutions in notebook 10.

1. **Predict the dtype** of: `int8 + int16` arrays; `int32 / 2`; `float32 array + 1.0`; `float32 array + np.float64(1.0)`; `bool + int` arrays; `np.arange(3) ** 2`.
2. **Trigger, then prevent, an overflow.** Build an `int16` array whose sum overflows; show the wrong answer, then produce the right one two different ways.
3. **When is float32 not enough?** Find the size at which summing `float32` 0.1s drifts more than 0.1% from the truth, and plot relative error against size for both float types.
4. **Build a memory report.** `memory_report(arrays)` prints each column's dtype, size, narrowest safe dtype and potential saving, and refuses to suggest downcasting any column whose name ends in `_id`.
5. **Cancellation in the wild.** Compute the standard error of a proportion, `sqrt(p*(1-p)/n)`, for `p = 1 - 1e-17` and `n = 1000`, directly and via a rearranged form. Explain the difference.

> [!KEY]
> - NumPy dtypes are **fixed-width**. Integers wrap; they don't grow.
> - **Array overflow is silent.** Only the scalar path warns.
> - **Reductions promote their accumulator**; **element-wise arithmetic doesn't**. The bug is almost always in the multiplication, not the sum.
> - `np.errstate(over="raise")` turns silence into an exception while you debug.
> - Since NumPy 2.0 (**NEP 50**), a Python scalar no longer upcasts an array.
> - **`float32` ≈ 7 digits, `float64` ≈ 15.** Near 10⁸, `float32` can't even represent +1.
> - `.sum()` uses **pairwise summation**; your hand-written loop doesn't.
> - Never write `E[x²] − E[x]²`. Use `np.var`.
> - `nan*` functions skip NaN: **always report the count too**.
> - **Downcasting must be verified**, and ID columns should stay wide.
