# Functions That Behave
@subtitle Signatures, scope, decorators and lazy generators
@time About 90 minutes
@notebook 04_functions_and_functional_programming.ipynb
@learn
- How to design a function signature that is hard to misuse.
- The mutable-default-argument trap, a bug that survives in production for years.
- Type hints and NumPy-style docstrings.
- Scope (the LEGB rule), `global`, `nonlocal` and closures.
- Decorators: how they work, and how to write useful ones.
- **Generators**: processing a file bigger than your RAM, one row at a time.
- `functools` and `itertools`: caching, partial application and lazy iteration.
@end

@lead A function is a promise. Give it these inputs and it will hand you back that output, and, ideally, it will do nothing else behind your back. This chapter is about writing functions that keep their promises, are hard to call wrongly, and stay fast and frugal when the data gets big.

## Anatomy of a good function

Every function in this course's `src/` folder follows one shape. Each part earns its place:

```python
def normalise(
    values: list[float],
    *,                         # everything after this is keyword-only
    method: str = "zscore",
    ddof: int = 1,
) -> list[float]:
    """Scale a sequence of numbers to a common range.

    Parameters
    ----------
    values : list of float
        The raw values. Must contain at least two elements.
    method : {"zscore", "minmax"}, default "zscore"
    ddof : int, default 1
        Delta degrees of freedom; 1 gives the sample standard deviation.

    Returns
    -------
    list of float

    Raises
    ------
    ValueError
        Fewer than two values, an unknown method, or zero spread.
    """
    if len(values) < 2:
        raise ValueError(f"need at least 2 values, got {len(values)}")
    ...
```

- **Type hints** (`list[float]`, `-> list[float]`) document intent and let tools catch mistakes.
- **A NumPy-style docstring**, the convention of the whole scientific Python world, explains parameters, return value, errors and examples.
- **Validation up front.** Fail early and loudly: a function that returns nonsense for bad input is far more expensive to debug than one that refuses it.
- **Helpful errors.** `f"unknown method {method!r}; expected 'zscore' or 'minmax'"` saves the caller a trip to the docs.

### Why the lonely `*`?

Everything after a bare `*` is **keyword-only**. That turns a cryptic call into a self-documenting one, and it lets you reorder or add parameters later without breaking a single caller:

```python
normalise(data, "minmax", 0)                  # ✘ now a TypeError, and good riddance
normalise(data, method="minmax", ddof=0)      # ✔ obvious at a glance
```

> [!TIP]
> A rule worth adopting: **every boolean or "mode" parameter should be
> keyword-only.** A call like `f(df, True, False)` is unreadable in a code
> review.

## Parameters in full

Python's parameter system is unusually expressive. Here's the whole picture in one signature:

```python
def demo(pos_only, /, standard, *args, kw_only, **kwargs):
    ...
demo(1, 2, 3, 4, kw_only=5, extra="x")
# pos_only=1  standard=2  args=(3, 4)  kw_only=5  kwargs={'extra': 'x'}
```

Everything before `/` is positional-only; everything after `*` is keyword-only. `*args` collects extra positional arguments into a tuple, `**kwargs` collects extra keywords into a dict. Their most common honest use is **forwarding** arguments to another function, for example a plotting wrapper that supplies house defaults the caller can still override:

```python
def plot_with_defaults(x, y, **kwargs):
    settings = {"linewidth": 2, "marker": "o"} | kwargs   # caller wins
    ...
```

## The mutable default argument trap

The most notorious gotcha in the language, and it follows directly from the object model in Chapter 2. **Default values are evaluated once, when the `def` line runs**, not each time you call the function. A mutable default is therefore shared by every call, forever.

```python
def append_reading_broken(value, readings=[]):
    readings.append(value)
    return readings

append_reading_broken(1.0)   # [1.0]
append_reading_broken(2.0)   # [1.0, 2.0]   ← expected [2.0]
append_reading_broken(3.0)   # [1.0, 2.0, 3.0]
```

![The default list is created once, when the function is defined, and every call that omits the argument shares and grows it.](assets/images/book/m01_mutable_default.png){width=100}

The fix is to use `None` as a sentinel and build a fresh object inside:

```python
def append_reading(value: float, readings: list[float] | None = None) -> list[float]:
    if readings is None:
        readings = []
    readings.append(value)
    return readings
```

> [!WARNING]
> The same trap applies to `{}`, `set()`, `pd.DataFrame()` and even
> `datetime.now()`, which gets frozen at definition time. The rule:
> **never put a mutable or time-dependent value in a default.**

## Scope: the LEGB rule

When Python meets a name, it searches four namespaces in a fixed order and stops at the first match: **L**ocal, **E**nclosing, **G**lobal, **B**uilt-in.

![The LEGB rule: Python looks for a name from the innermost scope outwards.](assets/images/legb_scope.png){width=78}

### Assignment makes a name local

One consequence trips everyone up once. If a function *assigns* to a name anywhere in its body, Python treats that name as local for the **whole** body:

```python
counter = 0
def increment_broken():
    counter += 1      # UnboundLocalError: assignment made `counter` local
```

You *can* fix it with `global counter`, but a function that reads and writes module state is hard to test. The better fix is to take state in and hand new state back:

```python
def increment(current: int) -> int:
    return current + 1
```

### Closures: functions that remember

An inner function keeps a live link to the variables of the function that created it, even after that outer function has returned. That captured environment is called a **closure**:

```python
def make_threshold_filter(threshold: float):
    def is_above(value: float) -> bool:
        return value > threshold        # remembered from the enclosing scope
    return is_above

above_100 = make_threshold_filter(100)
above_500 = make_threshold_filter(500)
[r for r in [50, 150, 600, 90, 1200] if above_100(r)]   # [150, 600, 1200]
```

To *rebind* a variable from the enclosing scope (not just read it), declare it `nonlocal`, the closure-sized cousin of `global`.

## Decorators

A decorator is a function that takes a function and returns a replacement. The `@` syntax is pure shorthand: `@timed` above `def slow_sum` means exactly `slow_sum = timed(slow_sum)`.

![A decorator wraps the original function, adding behaviour before and after each call.](assets/images/book/m01_decorator.png){width=100}

```python
import functools, time

def timed(func):
    @functools.wraps(func)        # keep the original's name and docstring
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"[{func.__name__}] took {time.perf_counter() - start:.4f}s")
        return result
    return wrapper

@timed
def slow_sum(n: int) -> int:
    return sum(i**2 for i in range(n))
```

You'll meet decorators everywhere: `@property`, `@pytest.fixture`, `@app.get` in FastAPI (Module 22), `@task` in Airflow.

> [!TIP]
> Always apply `functools.wraps`. Without it, debuggers, documentation tools
> and test reports all see a function called `wrapper`.

### A decorator with arguments

That needs one more layer: a function that *returns* a decorator. A retry decorator is a realistic example, since flaky network calls to data APIs are a daily reality:

```python
def retry(times: int = 3, delay: float = 0.01):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == times:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(times=3)
def flaky_api_call(): ...
```

### `lru_cache`: memoisation for free

`functools.lru_cache` remembers results by argument. For an expensive, deterministic function called repeatedly it's transformative. Naive recursive Fibonacci makes tens of millions of calls for `fibonacci(35)`; with the cache, the body runs just once for each of the 36 distinct values from 0 to 35.

```python
@functools.lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)
```

> [!NOTE]
> The idea has a lovely name. In 1968 the British AI researcher Donald Michie
> described "memo functions" in the journal *Nature*: functions that keep a
> memo of answers they've already worked out. The word **memoisation** comes
> straight from his paper.

> [!WARNING]
> `lru_cache` needs **hashable** arguments, so it can't cache a function that
> takes a list or a DataFrame. It also keeps results in memory: set `maxsize`
> deliberately rather than `None` on a hot path.

## Generators: the memory tool

A generator produces values **one at a time, on demand**, instead of building the whole collection up front. For data science this is the difference between processing a 50 GB file and getting a `MemoryError`. Any function containing `yield` is a generator function:

```python
def read_records_lazy(n: int):
    for i in range(n):
        yield {"id": i, "value": i * 1.5}
```

A list of a million records holds a million records. A generator holds… one recipe, and a bookmark:

![The list's memory grows with every element; the generator's stays at about two hundred bytes no matter how many items it will eventually produce.](assets/images/book/m01_generator_memory.png){width=64}

### Streaming a file

Chain generators together and you get a pipeline in which **nothing runs until the final loop asks for the next item**. A real file object iterates line by line in exactly the same way:

```python
def parse_lines(lines):
    for line in lines:
        name, amount = line.strip().split(",")
        yield {"name": name, "amount": float(amount)}

def filter_large(records, minimum: float):
    for record in records:
        if record["amount"] >= minimum:
            yield record

pipeline = filter_large(parse_lines(open("big.csv")), minimum=100)
for record in pipeline:     # data starts flowing only now
    ...
```

![A lazy pipeline pulls one record at a time through every stage.](assets/images/book/m01_lazy_pipeline.png){width=100}

### Generator expressions

Swap a list comprehension's square brackets for round ones and you get a generator expression. Prefer it whenever the result goes straight into a consumer like `sum`, `max`, `any` or `all`:

```python
total = sum(v**2 for v in range(1_000_000))   # never builds the million-item list
any(v > 999_998 for v in values)              # stops at the first True
```

> [!WARNING]
> A generator is **single-use**. Once exhausted it yields nothing more, and it
> does so silently, without an error:
>
> ```python
> gen = (x for x in [1, 2, 3])
> list(gen)    # [1, 2, 3]
> list(gen)    # []   ← empty, not an error
> ```

## `map`, `filter`, `reduce` and `lambda`

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]
list(map(lambda x: x**2, numbers))            # [1, 4, 9, 16, 25]
list(filter(lambda x: x % 2 == 0, numbers))   # [2, 4]
reduce(lambda acc, x: acc * x, numbers)       # 120
```

`lambda` creates a one-expression anonymous function. Use it only for short callables passed to something else; anything longer deserves a `def` and a name. In everyday Python a comprehension usually reads better than `map` and `filter`.

> [!STORY] From a 1930s logic paper to Google's data centres
> The keyword `lambda` comes from the **lambda calculus**, a formal system
> invented by the logician Alonzo Church in the 1930s to study what it means
> to compute something at all, years before electronic computers existed.
> Lisp borrowed it in the late 1950s, along with the habit of "mapping" a
> function over a list and "reducing" the results. In 2004 two Google
> engineers, Jeffrey Dean and Sanjay Ghemawat, published *MapReduce*, a way
> to spread exactly that pattern across thousands of machines. It powered a
> generation of big-data tools (Module 20). Not bad for a logic paper.

### `functools.partial`: pre-filling arguments

```python
def scale(value: float, *, factor: float, offset: float = 0.0) -> float:
    return value * factor + offset

to_fahrenheit = functools.partial(scale, factor=9 / 5, offset=32)
to_fahrenheit(20)     # 68.0
```

## `itertools`: lazy iteration tools

```python
import itertools

itertools.chain([1, 2], [3, 4])                     # 1, 2, 3, 4
itertools.islice(itertools.count(10), 5)            # 10, 11, 12, 13, 14
itertools.combinations(["a", "b", "c"], 2)          # ('a','b'), ('a','c'), ('b','c')
itertools.product([0.1, 0.5], ["l1", "l2"])         # every pairing: a grid search
itertools.groupby(sorted_records, key=lambda r: r["dept"])
```

> [!WARNING]
> `itertools.groupby` groups only **consecutive** equal items, so sort by the
> key first. Unsorted input gives you the same group several times.

> [!TIP]
> `itertools.product` is exactly how a hyperparameter grid search enumerates
> its candidates. You'll see it again, wrapped inside `GridSearchCV`, in
> Module 13.

## Pure functions and side effects

A **pure** function returns the same output for the same input and changes nothing outside itself. Pure functions are trivially testable, safe to run in parallel, and cacheable.

```python
tax_rate = 0.2
def apply_tax_impure(order: dict) -> None:      # ✘ mutates input, reads a global
    order["total"] = order["amount"] * (1 + tax_rate)

def apply_tax(order: dict, rate: float) -> dict:  # ✔ everything in, everything out
    return {**order, "total": order["amount"] * (1 + rate)}
```

Not everything can be pure. Reading files, writing to databases and logging are side effects, and they're the whole point of the program. The professional pattern is to **push side effects to the edges**: a thin impure shell that does I/O, wrapped around a pure core that does the thinking. You'll build exactly that in this module's project.

## Try it yourself

Exercises 4.1 to 4.5 in notebook 04; solutions in notebook 09.

1. **Fix the signature.** Rewrite `def summarise(data, True, 0.95, None)` as a well-designed signature with keyword-only options, defaults, type hints and a NumPy-style docstring.
2. **Spot the trap.** Call `add_feature(df_columns, new_cols=[])` three times with only its first argument. Explain the output, then fix it.
3. **Write a decorator.** `@validate_positive` should raise `ValueError` if any numeric positional argument is negative, and preserve the wrapped function's metadata.
4. **Generator pipeline.** Chain `parse(lines)`, `only_errors(records)` and `take(records, n)` over log lines, and confirm with `sys.getsizeof` that memory doesn't grow with input size.
5. **Closure counter.** Write `make_counter()` returning a function that yields 1, 2, 3… on successive calls, without `global`. Hint: `nonlocal`.

> [!KEY]
> - Make "mode" and boolean parameters **keyword-only** with `*`.
> - **Never use a mutable default argument.** Use `None` as a sentinel.
> - Type hints plus NumPy-style docstrings are the ecosystem's conventions.
> - Names resolve by **LEGB**; assigning inside a function makes a name local.
> - A **decorator** wraps a function; always use `functools.wraps`.
> - `lru_cache` turns an expensive deterministic function into a lookup.
> - **Generators trade time for memory**: constant footprint whatever the input size, and single-use.
> - Prefer **pure functions**; push I/O and mutation to the edges.
