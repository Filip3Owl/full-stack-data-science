# Choosing the Right Container
@subtitle Pick by complexity, not by habit
@time About 90 minutes
@notebook 03_data_structures.ipynb
@learn
- The four built-in collections and how to choose between them **by complexity**.
- Why `x in my_set` can be thousands of times faster than `x in my_list`, measured rather than asserted.
- The `collections` module: `Counter`, `defaultdict`, `deque`, `namedtuple`.
- Comprehensions, the idiom that replaces most of your `for` loops.
- Sorting with key functions, and why a **stable** sort matters.
- Unpacking and star expressions.
@end

@lead A kitchen has drawers, shelves, a spice rack and a fridge. You could keep everything in one big drawer, and for a while it would even work. Then one day you need the cumin in a hurry. Python's built-in collections are the same: each one is shaped for a particular kind of reaching-in, and choosing the wrong one is the commonest cause of slow Python.

## The map

Here's the summary to carry around. Almost every everyday performance problem in Python comes from misreading the "membership" row.

![Choosing a built-in collection: ordering, mutability, duplicates and, above all, the cost of asking "is x in here?".](assets/images/data_structures_comparison.png){width=95}

## Lists: the ordered workhorse

A `list` is a mutable, ordered sequence. Internally it's an **array of pointers** to objects, and that one fact explains its whole personality: indexing by position is instant, but finding a *value* means walking the list.

```python
revenues = [1200.50, 980.00, 1540.75, 620.25, 2100.00]
len(revenues), sum(revenues), min(revenues), max(revenues)
```

### Slicing: `[start:stop:step]`

The `stop` index is always **excluded**. That isn't an accident. It makes `len(x[a:b]) == b - a`, and it makes `x[:n] + x[n:] == x` true for every `n`.

```python
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
data[2:5]     # [2, 3, 4]
data[:3]      # [0, 1, 2]       the first three
data[-3:]     # [7, 8, 9]       the last three
data[::2]     # [0, 2, 4, 6, 8] every second element
data[::-1]    # reversed
data[:]       # a NEW list: the standard shallow-copy idiom
```

> [!STORY] Why counting starts at zero
> In 1982 the computer scientist Edsger Dijkstra wrote a short handwritten
> note, *Why numbering should start at zero* (EWD831). His argument was
> about exactly this slicing convention: describing a range as "from a,
> up to but not including b" makes lengths a simple subtraction and lets
> adjacent ranges meet without overlapping or leaving gaps. Start counting at
> zero and the first n items are simply `[0, n)`. Python, NumPy and pandas all
> follow his advice.

### The methods worth knowing

```python
demo = [1, 2, 3]
demo.append(4)        # add one item at the end           O(1)
demo.extend([5, 6])   # add many                          O(k)
demo.insert(0, 0)     # insert at a position              O(n), shifts everything
demo.remove(3)        # delete the first matching VALUE   O(n)
demo.pop()            # remove and return the last item   O(1)
```

> [!WARNING]
> Mutating methods return `None`, not the list. `sorted_list = my_list.sort()`
> stores `None`, a slip almost everyone makes once. Use `sorted(my_list)` when
> you want a new sorted list and the original left alone.

## Tuples: records, and the hashable sequence

Tuples serve two different purposes, and mixing them up causes confusion:

1. **Records**: a fixed set of fields where *position has meaning*, like `(lat, lon)`.
2. **Immutability**: a sequence that can be a `dict` key or a `set` member.

```python
latitude, longitude = (-23.55, -46.63)       # unpacking reads a record

city_populations = {
    ("Brazil", "Sao Paulo"): 12_400_000,      # a composite key
    ("Japan", "Tokyo"): 13_960_000,
}
{["Brazil", "Sao Paulo"]: 1}   # TypeError: unhashable type: 'list'
```

> [!NOTE]
> **Immutable doesn't mean deep-frozen.** A tuple promises its *references*
> never change; the objects they point to may still be mutable. So
> `t = ([1, 2], "fixed"); t[0].append(3)` works, while `t[0] = [9, 9]` raises.

## Dictionaries: the most important structure in Python

A `dict` maps **hashable keys** to values, with **O(1)** average lookup, and since Python 3.7 it remembers insertion order. Loop over pairs with `.items()`:

```python
customer = {"id": 1042, "name": "Ana Silva", "channel": "organic"}
for key, value in customer.items():
    print(key, value)
```

### Safe access

Indexing a missing key raises `KeyError`. When a field may simply be absent from some records, ask politely instead:

```python
record = {"id": 7, "spend": 100.0}
record.get("channel")                 # None
record.get("channel", "unknown")      # 'unknown'
record.setdefault("channel", "unknown")   # inserts the default if absent
```

### Merging

The `|` operator (Python 3.9+) merges two dicts; keys on the right win. That makes it perfect for layering settings over defaults, and neither input is modified:

```python
defaults = {"model": "linear", "alpha": 0.1, "max_iter": 100}
config = defaults | {"alpha": 0.5}    # {'model': 'linear', 'alpha': 0.5, ...}
{**defaults, "alpha": 0.5}            # the same, as unpacking syntax
```

## Sets: membership and deduplication

A `set` is an unordered collection of unique, hashable elements with **O(1)** membership tests. It's the right answer far more often than people realise:

```python
set([101, 102, 101, 103, 102, 104])     # {101, 102, 103, 104}
```

### Set algebra

Set operators replace a lot of loop-and-compare code, and they map directly onto the SQL joins you'll meet in Module 05:

```python
last_month = {"ana", "bruno", "carla", "diego"}
this_month = {"bruno", "carla", "elena", "felipe"}

last_month & this_month   # retained   {'bruno', 'carla'}
last_month - this_month   # churned    {'ana', 'diego'}
this_month - last_month   # new        {'elena', 'felipe'}
last_month | this_month   # everyone
last_month ^ this_month   # in exactly one month
```

## Why complexity is not academic

Here's the heart of the chapter. To answer `x in container`, a list has to look at elements one by one until it finds a match or runs out: that's **O(n)**. A set runs the value through a **hash function**, which points straight at the one bucket where it would live: that's **O(1)**.

![A list checks every element in turn. A set hashes the value and jumps to the right bucket, however big the set is.](assets/images/book/m01_hash_table.png){width=100}

> [!STORY] A memo at IBM, 1953
> The trick behind every set and dict in Python is older than most
> programming languages. In 1953 Hans Peter Luhn, an engineer at IBM, wrote
> an internal memo proposing to turn a key into a number and use that number
> to choose a "bucket" to look in, rather than searching everything. Luhn
> also gave his name to the check-digit formula still used to validate credit
> card numbers. Seventy years later, his idea is what makes
> `if user_id in seen:` instant.

The notebook times a batch of lookups against containers of growing size, choosing values that *aren't* present (the worst case for a list, which must check everything before saying no):

![Measured on the machine that built this book. Both axes are logarithmic: the list's cost climbs with every extra element while the set's stays flat.](assets/images/book/m01_set_vs_list.png){width=64}

> [!TIP]
> **The practical rule:** the moment you write `if x in some_list` *inside a
> loop*, convert the list to a set first. One extra line routinely turns a
> multi-hour job into a multi-second one.
>
> ```python
> valid = set(load_valid_ids())          # O(n + m) instead of O(n × m)
> clean = [r for r in records if r["id"] in valid]
> ```

### The trade-off: memory

Sets and dicts buy speed with space: their hash tables are deliberately kept roughly a third empty so collisions stay rare. For 100,000 integers, a set takes several times the memory of the equivalent list. Usually a great trade; worth knowing when memory is the limit.

## Comprehensions

A comprehension builds a collection from an iterable in a single expression. It's more readable than the equivalent loop, and faster too, because the looping happens inside the interpreter's C code.

![The three parts of a comprehension, read left to right.](assets/images/book/m01_comprehension.png){width=90}

```python
temperatures_c = [18.5, 22.0, 31.2, 15.8, 28.4, 35.1]
warm_f = [t * 9 / 5 + 32 for t in temperatures_c if t > 20]
```

Dict and set comprehensions follow the same pattern:

```python
catalogue = {name: price for name, price in zip(products, prices, strict=True)}
premium = {n: p * 0.9 for n, p in catalogue.items() if p > 100}
initials = {name[0] for name in products}        # deduplicates for free
```

### Filter or transform?

Position matters. An `if` at the **end** *removes* items; an `if/else` at the **front** *transforms* every item:

```python
values = [10, -5, 8, -3, 0]
[v for v in values if v > 0]          # [10, 8]          drops items
[v if v > 0 else 0 for v in values]   # [10, 0, 8, 0, 0] keeps all, clips them
```

> [!WARNING]
> **Don't nest more than two levels.** If a comprehension needs a comment to be
> understood, write the loop instead: clarity beats cleverness in code someone
> will debug at 2 a.m. Two levels is fine, read left to right exactly like the
> loops it replaces: `[v for batch in batches for v in batch]`.

## The `collections` module

Four specialised containers that delete whole categories of boilerplate.

### `Counter`: frequency counting in one line

```python
from collections import Counter

counts = Counter(["organic", "paid", "organic", "referral", "paid", "organic"])
counts.most_common(2)     # [('organic', 3), ('paid', 2)]
counts["unknown"]         # 0: a missing key never raises
counts.total()            # 6
```

> [!STORY] Counting words to settle a 175-year-old argument
> Of the 85 *Federalist Papers*, written in 1787–88 to persuade New Yorkers
> to ratify the US Constitution, twelve were claimed by both Alexander
> Hamilton and James Madison. In 1964 the statisticians Frederick Mosteller
> and David Wallace settled it by counting. Not grand ideas, but humble
> "function words": Hamilton wrote *upon* constantly, Madison hardly ever;
> Madison preferred *whilst* where Hamilton wrote *while*. Feeding those
> frequencies into a careful Bayesian analysis, they attributed all twelve
> disputed papers to Madison. It was one of the founding moments of
> statistical text analysis, and at its core is a `Counter`.

### `defaultdict`: no more "if key not in dict"

```python
from collections import defaultdict

grouped = defaultdict(list)            # a missing key starts as []
for name, amount in transactions:
    grouped[name].append(amount)

totals = defaultdict(float)            # a missing key starts as 0.0
for name, amount in transactions:
    totals[name] += amount
```

### `deque`: a queue with fast ends

`list.insert(0, x)` and `list.pop(0)` are O(n), because every remaining element has to shuffle along. A `deque` makes both ends O(1), and `maxlen` gives you a rolling window that forgets old items for free:

```python
from collections import deque

window = deque(maxlen=3)
for reading in [10, 20, 30, 40, 50]:
    window.append(reading)      # ends as deque([30, 40, 50])
moving_average = sum(window) / len(window)
```

### `namedtuple`: records with names

```python
from collections import namedtuple

Measurement = namedtuple("Measurement", ["timestamp", "sensor_id", "value"])
reading = Measurement("2024-01-15T10:30", "S-001", 23.4)
reading.value              # 23.4 by name; reading[2] still works
reading._replace(value=24.0)   # a modified copy; the original is untouched
```

> [!TIP]
> For anything beyond a trivial record, prefer `@dataclass`: it supports type
> hints, defaults, methods and mutability. Chapter 5 covers it.

## Sorting

`sorted()` returns a new list; `list.sort()` sorts in place. Both take a `key` function that computes *what to sort by*:

```python
by_salary = sorted(employees, key=lambda e: e["salary"], reverse=True)

# Several fields: return a tuple. Negating a number flips its direction.
by_dept_then_salary = sorted(employees, key=lambda e: (e["dept"], -e["salary"]))

from operator import itemgetter
by_dept = sorted(employees, key=itemgetter("dept", "salary"))   # faster, clearer
```

### Stability

Python's sort is **stable**: items that compare equal keep their original relative order. That's a guarantee, and it enables a neat technique: sort by the least important key first, then by the most important one, and the earlier ordering survives inside each group.

```python
step_1 = sorted(employees, key=itemgetter("salary"), reverse=True)
step_2 = sorted(step_1, key=itemgetter("dept"))   # salary order kept per dept
```

> [!NOTE]
> Python's sorting algorithm is called **Timsort**, written by Tim Peters for
> Python in 2002. It hunts for already-sorted runs in your data, which real
> data is full of, and it's stable by design. It was good enough that Java
> and Android later adopted it too.

## Unpacking and star expressions

```python
first, second, third = [1, 2, 3]
head, *rest = [1, 2, 3, 4, 5]          # head=1, rest=[2, 3, 4, 5]
first, *middle, last = [1, 2, 3, 4, 5]
a, b = b, a                            # swap, no temporary variable

describe(**{"name": "Ana", "dept": "data", "salary": 95_000})   # dict → keywords
describe(*["Bruno", "eng", 105_000])                            # list → positions
```

## Try it yourself

Exercises 3.1 to 3.5 in notebook 03; solutions in notebook 09.

1. **Pick the structure.** Name the best collection, with a one-line justification, for: 50 million already-processed user IDs; the last 100 sensor readings; country code → country name; an RGB colour that must never change; word counts across 10,000 reviews.
2. **Fix the slow function.** `[i for i in all_ids if i not in known_ids]` takes minutes with 100,000 IDs on each side. Make it run in well under a second, and prove it with `%timeit`.
3. **Comprehension practice.** From a list of sales dicts, produce in single comprehensions: a list of revenues, a dict product → revenue, and the set of products with revenue above 100.
4. **Group without pandas.** Using `defaultdict`, write `group_by_key(records, key)`. This is `pandas.groupby` in eight lines, and writing it once makes Module 03 much clearer.
5. **Multi-key sorting.** Sort `employees` by department ascending, salary descending, name ascending, in a **single** `sorted()` call.

> [!KEY]
> - **Choose containers by complexity.** `list` membership is O(n); `set` and `dict` are O(1).
> - The habit that matters: **`set()` anything you'll repeatedly test membership against.**
> - Mutating list methods return `None`. Use `sorted()` when you want a new list.
> - Tuples are records *and* the hashable sequence, so they can be dict keys.
> - **Comprehensions** are the default way to build collections; keep them to two levels.
> - `Counter`, `defaultdict`, `deque` and `namedtuple` delete whole categories of boilerplate.
> - Python's sort is **stable**, which makes multi-pass sorting reliable.
