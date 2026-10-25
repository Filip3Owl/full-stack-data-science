# Names, Numbers and Other Surprises
@subtitle The object model, and why b = a never copies anything
@time About 90 minutes
@notebook 02_python_fundamentals.ipynb
@learn
- Python's object model, and why `b = a` does not copy anything.
- Mutability and aliasing, the silent bug behind most data-pipeline mysteries.
- Integers that never overflow, floats that never quite add up, and `Decimal` for money.
- Strings, f-strings and text encodings.
- Truthiness, `None`, and the three different kinds of "missing".
- Control flow, including `for...else` and structural pattern matching.
@end

@lead If you only read one chapter of this module carefully, make it this one. Nearly every subtle bug you'll ever write in pandas traces back to something on these pages, usually to the very first idea: what a variable actually *is*.

## Everything is an object

Python has no "primitive types" that live by different rules. An integer, a string, a function, a class, even a module: all of them are objects, and every object carries exactly three things.

| Property | How you see it | Can it change? |
| :--- | :--- | :--- |
| **Identity**, its address in memory | `id(obj)` | Never |
| **Type**, what kind of thing it is | `type(obj)` | Never |
| **Value**, what it holds | the object itself | *Only if the type is mutable* |

Functions are objects too, which is why you can hand one to a new name without calling it:

```python
def greet(name: str) -> str:
    return f"Hello, {name}"

say_hi = greet          # no call, no parentheses: a second name, same function
say_hi("again")         # 'Hello, again'
```

## Names are labels, not boxes

Here is the most important idea in the chapter.

If you learned C or Java first, you probably picture a variable as a **box that holds a value**. Python doesn't work like that. A name is a **label tied to an object**, and assignment moves the label. It never copies the object.

![Two mental models. In Python, assignment ties another label to the same object.](assets/images/names_and_objects.png){width=88}

Think of a cloakroom: you hand in your coat and get a numbered ticket. The ticket isn't the coat. Give a friend a photocopy of your ticket and you now have two tickets for *one* coat. If your friend takes the coat and sews a patch on it, you'll find the patch too.

```python
a = [1, 2, 3]
b = a               # a second ticket for the SAME list
print(a is b)       # True: same object
```

### `==` versus `is`

These two operators ask genuinely different questions:

- `a == b` asks: do these have the same **value**?
- `a is b` asks: are these the **same object**?

```python
list_1 = [1, 2, 3]
list_2 = [1, 2, 3]      # a separate list that happens to hold equal values
list_1 == list_2        # True: same contents
list_1 is list_2        # False: two different objects
```

> [!WARNING]
> **Use `is` only for `None`, `True` and `False`.** For everything else, use
> `==`. CPython pre-builds small integers (−5 to 256) and shares them, so
> `100 is 100` happens to be `True` while `1000 is 1000` may be `False`. That
> is an implementation detail you must never rely on.

## Mutability is where the bugs live

Every type is either **mutable** (it can change in place) or **immutable** (it can't). That one property decides whether `b = a` is harmless or a time bomb.

| Immutable | Mutable |
| :--- | :--- |
| `int`, `float`, `bool`, `complex` | `list` |
| `str` | `dict` |
| `tuple` | `set` |
| `frozenset`, `bytes` | `bytearray`, **`numpy.ndarray`**, **`pandas.DataFrame`** |

![Mutable versus immutable: "changing" an immutable value really creates a new object.](assets/images/mutable_vs_immutable.png){width=100}

```python
a = 10
b = a
b += 1          # ints can't change in place → a NEW int; b's label moves
                # a is still 10

x = [1, 2]
y = x
y.append(3)     # lists CAN change in place → both labels see it
                # x is now [1, 2, 3], even though we only touched y
```

### Why this matters in real data work

The toy example looks easy to avoid. It isn't, because in real code the aliasing hides behind a function call:

```python
def add_engineered_feature(records: list[dict]) -> list[dict]:
    """BUG: mutates the caller's data as a side effect."""
    for record in records:
        record["total"] = record["price"] * record["quantity"]
    return records

enriched = add_engineered_feature(raw_orders)
# raw_orders has silently gained a "total" key: our "raw" data is gone,
# and enriched is raw_orders → True
```

### The fix: copy, or better, don't mutate

Python has two kinds of copy, and the difference shows up as soon as your data is nested, which real data always is.

| Copy | Syntax | New container? | New contents? |
| :--- | :--- | :---: | :---: |
| **Shallow** | `list(x)`, `x[:]`, `copy.copy(x)` | ✔ | ✘, shares the inner objects |
| **Deep** | `copy.deepcopy(x)` | ✔ | ✔, copies all the way down |

![A shallow copy duplicates only the outer container. A deep copy duplicates everything inside it too.](assets/images/book/m01_shallow_deep.png){width=100}

Copying defensively everywhere gets tiring, though. The professional habit is to write functions that **don't mutate their inputs at all**. Build and return something new:

```python
def add_total(records: list[dict]) -> list[dict]:
    """Return new records with a `total` field. The input is never touched."""
    return [
        {**record, "total": record["price"] * record["quantity"]}
        for record in records
    ]
```

`{**record, "total": ...}` builds a brand-new dict from the old one plus one key.

> [!TIP]
> This is exactly why most pandas methods return a new object rather than
> editing in place, and why `inplace=True` is discouraged in modern pandas.
> You'll meet the idea again in Module 03.

## Numbers

### Integers never overflow (in Python)

Python integers grow to whatever size they need. There's no ceiling:

```python
2**200          # a 61-digit number, no problem
population = 8_100_000_000    # underscores are just for readability
```

> [!WARNING]
> **NumPy integers do not behave like this.** For speed they use fixed widths
> such as `int64`, and they wrap around silently when they run out of room.
> Doubling `np.int64(2**62)` lands one step past the ceiling and comes back as
> the most negative number the type can hold. A revenue total that silently
> turns negative is the kind of bug that reaches a dashboard before anyone
> notices. Module 02 takes this apart properly.

> [!STORY] The video that broke a counter
> In December 2014, the music video for *Gangnam Style* became the first
> YouTube video to pass 2,147,483,647 views, the largest number a signed
> 32-bit integer can hold. YouTube had already switched its view counter to
> 64 bits, and said so with a joke on its own page. It's a light-hearted
> example of a very serious rule: every fixed-width number has a ceiling, and
> **someone has to decide in advance what happens when you reach it**.

### Floats are approximations

Floats are stored in binary, following the IEEE 754 standard. Just as 1/3 can't be written exactly in decimal (0.3333…), **0.1 can't be written exactly in binary**. Every float therefore carries a tiny representation error, and arithmetic can nudge it one representable step sideways.

```python
0.1 + 0.2              # 0.30000000000000004
0.1 + 0.2 == 0.3       # False
f"{0.1:.20f}"          # '0.10000000000000000555'
```

![The number line near 0.3, magnified about 10¹⁷ times. Only the tick marks exist as float64 values; 0.1 + 0.2 lands one tick away from the float you get by typing 0.3.](assets/images/book/m01_float_gap.png){width=100}

#### Comparing floats correctly

Never use `==` on floats you computed. Compare within a tolerance:

```python
import math
import numpy as np

math.isclose(0.1 + 0.2, 0.3)                 # True
np.isclose(0.1 + 0.2, 0.3)                   # True, and it works on arrays
np.allclose(np.array([0.1 + 0.2, 3.0]), np.array([0.3, 3.0]))   # True
```

#### When exactness is mandatory: money

For financial calculations, accumulated float error is unacceptable. Auditors don't accept "it was off by 10⁻¹⁶ per row, times forty million rows". Use `decimal.Decimal`, which does exact base-10 arithmetic:

```python
from decimal import Decimal

Decimal("0.10") + Decimal("0.20") == Decimal("0.30")   # True
Decimal(0.1)     # 0.1000000000000000055511151231257827…  ← the trap!
```

Always build a `Decimal` from a **string**. `Decimal(0.1)` faithfully copies the float's error, which defeats the whole point.

> [!STORY] The stock index that lost half its value to rounding
> In January 1982 the Vancouver Stock Exchange launched a new index at
> 1000.000. Twenty-two months later it stood at about 520, even though the
> market it tracked had broadly been rising. The culprit: the index was
> recalculated thousands of times a day, and each time the result was
> **truncated** (chopped) to three decimal places instead of rounded. Each
> chop shaved off a fraction of a thousandth, always downwards. When the
> exchange recomputed the index properly in November 1983, it jumped
> overnight from roughly 524 to roughly 1099. Tiny, systematic numeric errors
> don't average out. They **accumulate**.

### Division, floor division and negative modulo

```python
7 / 2      # 3.5   true division, always a float
7 // 2     # 3     floor division
7 % 2      # 1     remainder
-7 // 2    # -4    floors DOWN, toward minus infinity (C and Java give -3)
-7 % 2     # 1     the sign follows the divisor
```

That last rule surprises people, but it makes cyclical features behave: hour 26 wraps to `26 % 24 == 2`, and hour −3 wraps to `-3 % 24 == 21`.

## Strings

Strings are **immutable sequences of Unicode characters**. Every "change" makes a new string:

```python
s = "data science"
s.upper()      # 'DATA SCIENCE', a NEW string; s is unchanged
s[0], s[-1]    # 'd', 'e'   negative indexes count from the end
s[0:4]         # 'data'     [start:stop], stop is excluded
s[::-1]        # 'ecneics atad'
```

### The cleaning toolkit

Cleaning text columns is a daily chore, and a handful of methods cover most of it:

```python
messy = "  Sao Paulo, BR \n"
messy.strip()                     # 'Sao Paulo, BR'
messy.strip().lower()             # 'sao paulo, br'
messy.strip().split(",")          # ['Sao Paulo', ' BR']
messy.strip().startswith("Sao")   # True
"-".join(["2024", "01", "15"])    # '2024-01-15'
```

> [!TIP]
> Use `"".join(parts)` to glue many strings together. Building a string with
> `+=` in a loop is O(n²), because every step allocates a whole new string.

### f-strings: formatting you should use everywhere

The mini-language after the colon removes a surprising amount of manual rounding code:

```python
value, ratio, name = 1234567.8915, 0.0734, "revenue"
f"{value:,.2f}"     # '1,234,567.89'   thousands separator, 2 decimals
f"{ratio:.1%}"      # '7.3%'
f"{value:.3e}"      # '1.235e+06'
f"{name:>15}"       # right-aligned in 15 characters
f"{42:05d}"         # '00042'
n_rows = 1500
f"{n_rows=}"        # 'n_rows=1500'  ← brilliant for debugging
```

### Encoding: half of your data-loading errors

A `str` is abstract text. To write it to disk or send it over a network it must be **encoded** into bytes. UTF-8 is the right default; legacy files are often Latin-1 or Windows-1252, and decoding with the wrong one corrupts your data.

```python
"café".encode("utf-8")      # b'caf\xc3\xa9'   (5 bytes)
"café".encode("latin-1")    # b'caf\xe9'       (4 bytes)
"café".encode("utf-8").decode("latin-1")   # 'cafÃ©'  ← mojibake
```

> [!NOTE]
> That garbled `cafÃ©` has a name: **mojibake**, from the Japanese for
> "character transformation". Japanese computer users met it constantly in the
> days of several competing encodings, so they named it first.

> [!TIP]
> When `pd.read_csv` raises a `UnicodeDecodeError`, the file isn't broken. It
> simply isn't UTF-8. Try `encoding="latin-1"` or `encoding="cp1252"`.
> Chapter 6 covers this properly.

## Booleans, `None` and truthiness

Every object can be used where Python expects true or false. The rules are short:

- **Falsy:** `False`, `None`, `0`, `0.0`, `""`, `[]`, `{}`, `set()`, `()`.
- **Truthy:** everything else, including `"False"`, `[0]` and `-1`.

So the idiomatic emptiness check is simply `if not items:`, with no `len(...) == 0` needed.

> [!WARNING]
> This idiom is one of the few places where **NumPy and pandas deliberately
> break from plain Python**. Asking whether a multi-element array is "true" is
> ambiguous (any element? all of them?), so `if arr:` raises a `ValueError`.
> Say what you mean: `arr.any()`, `arr.all()`, or `arr.size` to test for
> emptiness.

### Three kinds of "missing"

In data work you'll meet three different nulls, and they are not interchangeable:

| Value | What it is | Equal to itself? |
| :--- | :--- | :--- |
| `None` | Python's null object | `None == None` → `True` |
| `np.nan` | IEEE 754 "not a number", a **float** | `np.nan == np.nan` → `False`, always |
| `pd.NaT` | pandas' missing timestamp | behaves like `nan` |

NaN's defining property really is that it isn't equal to itself. Test for it with `np.isnan(x)` or `.isna()`, never with `==`.

> [!WARNING]
> `nan != nan` is why `df[df["col"] == np.nan]` silently returns **zero rows**
> instead of your missing values. The correct call is `df[df["col"].isna()]`.
> This mistake makes it into production regularly.

## Control flow

### Chained comparisons

Python lets you write range checks the way a mathematician would, and evaluates the middle term only once:

```python
def classify_spend(amount: float) -> str:
    if amount < 0:
        return "invalid"
    elif 0 <= amount < 50:
        return "low"
    elif 50 <= amount < 200:
        return "medium"
    return "high"
```

### Loop over things, not indexes

The surest sign of someone arriving from another language is `for i in range(len(x))`. Python can iterate the object directly, and gives you helpers for the two common extras:

```python
for channel in channels:                        # just the items
    ...
for i, channel in enumerate(channels, start=1): # items plus a counter
    ...
for channel, revenue in zip(channels, revenues, strict=True):
    ...                                          # two sequences in step
```

`strict=True` (Python 3.10+) raises if the lengths differ, instead of silently dropping the extras. Silent truncation has hidden a great many data-alignment bugs.

### `for...else`

A genuinely useful feature almost nobody knows: the `else` block of a loop runs **only if the loop finished without hitting `break`**. It reads naturally as "if we never found it":

```python
for value in values:
    if value < 0:
        print(f"found a negative value: {value}")
        break
else:
    print("no negative values: data passes validation")
```

### Structural pattern matching

Added in Python 3.10, `match` is much more than a `switch` statement: it checks the **shape** of data and pulls pieces out in the same breath. It's excellent for messy API responses:

```python
match response:
    case {"status": "ok", "data": [*items]}:
        return f"success with {len(items)} records"
    case {"status": "ok"}:
        return "success, no data returned"
    case {"status": "error", "code": int(code)} if code >= 500:
        return f"server error ({code}): safe to retry"
    case {"status": "error", "code": int(code)}:
        return f"client error ({code}): do not retry"
    case _:
        return "unrecognised response shape"
```

### The walrus operator `:=`

`:=` assigns *and* returns a value inside an expression, which saves computing something twice:

```python
for reading in readings:
    if (scaled := reading * 1.8 + 32) > 100:
        print(f"{reading} → {scaled:.1f} above threshold")
```

> [!STORY] Why it's called Python
> Over the Christmas holidays of 1989, the Dutch programmer Guido van Rossum
> started a hobby project to keep himself busy: a scripting language that
> would be pleasant to read. He named it after the BBC comedy *Monty Python's
> Flying Circus*, not the snake. Version 0.9.0 appeared in February 1991.
> Readability was the point from day one, which is why so much of this chapter
> is about writing code that says what it means.

## Try it yourself

Exercises 2.1 to 2.5 in notebook 02; worked solutions are in notebook 09.

1. **Predict before you run.** Write down the output first, then check. Why do (a) and (b) differ, and why does (c) work even though tuples are immutable?

```python
a = [1, 2, 3]; b = a; b = b + [4]; print(a)    # (a)
a = [1, 2, 3]; b = a; b += [4];     print(a)    # (b)
t = ([1, 2], 3); t[0].append(99);   print(t)    # (c)
```

2. **A float comparison bug.** `is_fully_allocated([0.1, 0.2, 0.3, 0.4])` should be `True` but returns `False` because it checks `sum(weights) == 1.0`. Fix it.
3. **Safe enrichment.** Write `add_discount(orders, pct)` that returns new dicts with a `discounted_price` key, and prove with an assertion that the input list is unchanged.
4. **Text cleaning.** Write `clean_city(raw)` that turns `"  sÃO   paulo\n"` into `"Sao Paulo"`. Hint: `" ".join(raw.split())` collapses whitespace in one step.
5. **Missing-value audit.** Given `[1.0, np.nan, 3.0, None, 5.0]`, count the missing entries, and explain why `values.count(np.nan)` gives the wrong answer.

> [!KEY]
> - **Names are labels tied to objects.** Assignment never copies.
> - **Mutability decides everything.** Functions that mutate their arguments create bugs invisible at the call site, so return new objects instead.
> - `copy.copy` is shallow; nested data needs `copy.deepcopy`.
> - **Floats are approximate.** Compare with `math.isclose` / `np.isclose`; use `Decimal` (built from strings) for money.
> - Python ints are unbounded; **NumPy ints overflow silently**.
> - A `UnicodeDecodeError` means "wrong codec", not "broken file". UTF-8 is the default.
> - **`nan != nan`**: test missing values with `.isna()` or `np.isnan`.
> - Iterate over objects, not indexes: `enumerate`, `zip(..., strict=True)`.
