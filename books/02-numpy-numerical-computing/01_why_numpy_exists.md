# Why NumPy Exists
@subtitle Memory layout, vectorisation, and the honest limits of arrays
@time About 60 minutes
@notebook 01_why_numpy_exists.ipynb
@learn
- Why a Python list of numbers is slow: the answer is **memory layout**, not just interpreter speed.
- What "vectorised" really means underneath.
- How to measure the difference yourself instead of taking it on faith.
- Universal functions (ufuncs), the interface every NumPy operation shares.
- When NumPy is *not* the right answer.
@end

@lead Every numerical library in this course (pandas, scikit-learn, statsmodels, SciPy, PyTorch) is a layer on top of one object: the NumPy array. This module is where that layer stops being magic. We start with the most basic question of all: why does it exist? Python already has lists.

## The problem with a list of numbers

A Python `list` is a wonderful general-purpose container. It holds anything, in any mix, and grows on demand. That flexibility has a cost, and the cost is structural.

![A list stores pointers to scattered, fully fledged Python objects. An ndarray stores the raw numbers themselves, back to back, all of one type.](assets/images/numpy_memory_layout.png){width=95}

A list doesn't store your numbers. It stores **pointers** to full Python objects living elsewhere in memory. Adding two lists element by element therefore means, for *every single element*:

1. follow a pointer to object A,
2. follow a pointer to object B,
3. check both types at runtime,
4. dispatch to the right `__add__`,
5. allocate a brand-new Python object for the result,
6. store a pointer to it.

An ndarray stores the raw numbers, contiguously, all of one type. Adding two arrays runs a tight loop in compiled C with none of steps 1 to 4.

Think of a library. A list is a card catalogue: each card tells you which shelf, in which room, holds the book you want, and you walk there every time. An ndarray is a single shelf with the books already lined up in order. Reading every book is a stroll along one shelf.

### Seeing the overhead

```python
sys.getsizeof(10)     # 28 bytes for one Python int object
np.int64(10).itemsize # 8 bytes for the same number inside an array
```

> [!NOTE]
> Those extra bytes aren't waste. They hold the reference count, the type
> pointer and the arbitrary-precision digits that make Python integers
> unbounded (Module 01, Chapter 2). You're paying for real features. In
> numerical work you rarely need them, and NumPy lets you decline.

### `sys.getsizeof` lies about containers

Module 01 flagged this: `getsizeof` is **shallow**. For a list it counts the array of pointers and nothing they point to. The honest figure adds up every referenced object too:

```python
py_list = list(range(1_000_000))
np_array = np.arange(1_000_000, dtype=np.int64)

shallow = sys.getsizeof(py_list)                            # ~7.6 MiB
true_size = shallow + sum(sys.getsizeof(x) for x in py_list) # ~34 MiB
np_array.nbytes                                             # ~7.6 MiB, exactly
```

For an array there's nothing else to count: the data *is* the buffer, so `nbytes == size × itemsize`, exactly, always.

## Measuring the speed difference

Let's settle Module 01's claim with a measurement rather than an assertion: four ways to sum a million integers.

```python
def time_it(fn, *args, repeats: int = 3) -> float:
    """Best of `repeats` timings, in seconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - start)
    return best
```

Why the *minimum* and not the average? Timing noise is one-sided: background processes can only ever make a run slower, never faster. The fastest observed run is the closest estimate of the true cost.

![Summing a million integers four ways, measured on the machine that built this book. Note the logarithmic scale.](assets/images/book/m02_speed_bars.png){width=100}

Two results are worth pausing on.

**`sum()` on an ndarray is the slowest option of all.** Python's built-in `sum` walks the array one element at a time, and NumPy has to box each raw value back into a Python object to hand it over. You get all of the list's overhead *plus* the boxing.

**`arr.sum()` is dramatically faster**, because the whole loop runs in compiled code over one contiguous buffer: no boxing, no type checks, no pointer chasing.

> [!WARNING]
> **Never apply a Python built-in to an array.** `sum(arr)`, `max(arr)` and
> `min(arr)` are slower than even the list versions. Use `arr.sum()`,
> `arr.max()`, `np.min(arr)`.

## What "vectorised" really means

The word gets thrown around loosely. Precisely, it means: **the loop moves out of the Python interpreter and into compiled code.** Three consequences follow.

**1. Operations apply to the whole array at once.**

```python
a = np.array([1, 2, 3, 4]); b = np.array([10, 20, 30, 40])
a + b        # [11 22 33 44]
a * b        # [10 40 90 160]  element-wise, NOT matrix multiplication
a > 2        # [False False  True  True]  comparisons vectorise too
```

**2. There's no per-element type check.** The dtype is fixed for the whole array, so the C loop checks it once.

**3. Contiguity lets the CPU do several numbers per instruction.** Modern processors have SIMD registers that add four or eight `float64`s in one go, but only when the numbers sit side by side in memory. Summing the same five million values through a *strided view* (every second element of a bigger array) is noticeably slower than summing them from a contiguous copy.

> [!TIP]
> That's why `np.ascontiguousarray` exists, and why Chapter 8 spends time on
> strides. Slicing, transposing and reshaping can leave an array
> non-contiguous, and the cost is real.

> [!STORY] A notation that became a way of thinking
> The idea of operating on whole arrays at once is older than NumPy, older
> than Python, older than most programming languages still in use. In 1962
> Kenneth Iverson, then at Harvard and IBM, published *A Programming
> Language*, a mathematical notation in which `A + B` simply meant "add these
> two arrays, element by element". It became the language **APL**, beloved
> by actuaries and engineers for its terse power, and Iverson received the
> Turing Award in 1979 for a lecture titled *Notation as a Tool of Thought*.
> Every time you write `prices * quantities` instead of a loop, you're
> thinking in Iverson's notation.

## Universal functions

A **ufunc** is a function that works element by element on arrays, with broadcasting, type promotion and an optional output buffer all handled for you. Nearly everything in NumPy is one:

```python
values = np.array([1.0, 4.0, 9.0, 16.0])
np.sqrt(values)                     # [1. 2. 3. 4.]
np.maximum([1, 5, 3], [4, 2, 6])    # [4 5 6]  element-wise pairwise max
isinstance(np.sqrt, np.ufunc)       # True
```

### Why `math.sqrt` is no substitute

The `math` module works on one scalar at a time. Applying it to an array means looping in Python, which is exactly what we're trying to avoid. On a million values, `np.sqrt(arr)` beats `[math.sqrt(v) for v in arr]` by a large factor.

### `out=`: skip the allocation

Every ufunc can write into an array you already have instead of allocating a new one. On large arrays inside a loop, this matters:

```python
buffer = np.empty(5_000_000)
np.sqrt(x, out=buffer)       # writes in place, allocates nothing
```

## How the advantage scales

The gap isn't a fixed multiplier. It **widens with array size**, because Python's per-element overhead is constant while NumPy spreads its own fixed per-call cost over more and more data.

> [!NOTE]
> At very small sizes the advantage shrinks, and it can even invert. NumPy
> pays a fixed cost per *call* (parsing arguments, resolving dtypes,
> allocating the result) that a plain Python operation avoids. For a handful
> of values, plain Python can win. Don't "NumPy-ify" a three-element list.

## When NumPy is not the answer

Knowing a tool includes knowing its limits:

| Situation | Why NumPy struggles | Reach for |
| :--- | :--- | :--- |
| Mixed columns (strings + numbers + dates) | One dtype per array | **pandas** (Module 03) |
| Genuinely variable-length records | Rectangular arrays only | lists of dicts, or Arrow |
| Data larger than RAM | Everything lives in memory | **Dask / Spark** (Module 20) |
| Heavy text processing | No string vectorisation worth the name | Python `str`, pandas `.str` |
| An algorithm that can't avoid a loop | Python loop overhead returns | **Numba**, **Cython**, or C |
| Fewer than ~100 elements | Per-call overhead dominates | plain Python |

NumPy *can* hold mixed data, but watch what happens to the dtype:

```python
mixed = np.array([1, "two", 3.0])
mixed.dtype           # <U32: everything became a string
mixed[0] + mixed[0]   # '11'   ← string concatenation!
```

That silently destroys your numbers. It's exactly the problem pandas solves.

> [!STORY] From three projects to one, and on to the stars
> NumPy wasn't designed in one go. In 1995 Jim Hugunin, then a graduate
> student, wrote **Numeric**, the first serious array package for Python.
> In the early 2000s a team at the Space Telescope Science Institute built
> **Numarray** to handle very large astronomical images, and for a while the
> community was split between the two. In 2005 Travis Oliphant merged the
> best of both into a single package, and **NumPy 1.0** shipped in 2006.
> Underneath, it leaned on decades-old Fortran libraries, BLAS (1979) and
> LAPACK (1992), for its heaviest linear algebra.
>
> The result became the foundation of scientific Python. When the LIGO
> observatories announced the first detection of gravitational waves in 2016,
> and when the Event Horizon Telescope released the first image of a black
> hole in 2019, NumPy was in the software stack. Both feats are described in
> the 2020 *Nature* paper on NumPy written by its maintainers.

![The milestones in this book's stories, from early Monte Carlo to NumPy 2.0.](assets/images/book/m02_timeline.png){width=100}

## Try it yourself

Exercises 1.1 to 1.5 in notebook 01; solutions in notebook 10.

1. **Measure it yourself.** Compute the element-wise product of two sequences of a million floats four ways: a `for` loop, a list comprehension, `zip` + comprehension, and `arr_a * arr_b`. Report each time and the speed-up.
2. **The honest memory question.** Write `true_list_size(lst)` counting the pointer array plus every referenced object, compare it with the equivalent `int64` array for 10, 1,000 and 100,000 elements, and plot the ratio.
3. **Find the break-even point.** For `x * 2 + 1`, find the size at which NumPy overtakes a list comprehension, searching from 1 to 10,000. Why isn't it 1?
4. **Spot the de-vectorisation.** Explain why each of these drops back to Python speed, and rewrite it: `sum(arr)`; `np.array([np.sqrt(v) for v in arr])`; `max(arr)`; `np.array(list(map(lambda x: x * 2, arr)))`.
5. **`out=` in a loop.** Run 100 rounds of `y = sqrt(x) + log1p(x)` over a million elements, once allocating fresh arrays and once reusing preallocated buffers. Measure both.

> [!KEY]
> - A list stores **pointers to objects**; an ndarray stores **raw values, back to back, all one dtype**. Every performance difference follows from that.
> - **`sys.getsizeof` is shallow** and badly understates a list. `ndarray.nbytes` is exact.
> - **Vectorised** means the loop runs in compiled C, not in the interpreter.
> - **Never apply a Python built-in to an array**: `sum(arr)` is slower than the list version.
> - Ufuncs accept **`out=`** to reuse a buffer instead of allocating.
> - The advantage **grows with size** and can invert below ~100 elements.
> - NumPy needs one dtype and a rectangular shape. When your data isn't like that, the answer is pandas, not a cleverer array.
