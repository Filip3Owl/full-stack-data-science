# Under the Hood
@subtitle Strides, memory order, temporaries, and knowing when to change tools
@time About 75 minutes
@notebook 08_performance_and_memory.ipynb
@learn
- **Strides**: how one flat buffer pretends to be an N-dimensional array.
- C order versus Fortran order, and why walking the "wrong" way is slower.
- In-place operations, and how to stop allocating temporaries.
- Measuring memory honestly, hidden intermediates included.
- `np.memmap` for arrays bigger than your RAM.
- Knowing when NumPy has run out of road, and what comes next.
@end

@lead So far we've treated an array as a grid of numbers. This chapter lifts the lid and shows what's really there: one long, flat strip of bytes, plus a tiny recipe for reading it as a grid. Once you can see the strip, a dozen performance puzzles solve themselves.

## Strides: the trick behind everything

An ndarray is a **flat buffer** plus a rule for interpreting it. That rule is the `strides` tuple: how many **bytes** to step to move one position along each axis.

![What you see is a 3 × 4 grid; what's stored is twelve int64 values in a row. Moving one column steps 8 bytes; moving one row steps 32.](assets/images/book/m02_strides.png){width=100}

```python
arr = np.arange(12, dtype=np.int64).reshape(3, 4)
arr.strides      # (32, 8): next row = skip 32 bytes (4 elements), next column = 8
arr.T.strides    # (8, 32): swapped. Same buffer, no data moved
arr[::2].strides # (64, 8): skipping rows just doubles the row stride
```

This is why reshaping and transposing are **free**: they only rewrite the strides, never the data. It's also why so many operations return views (Chapter 2): a view is just a new set of strides over the same strip of memory.

### C order versus Fortran order

There are two natural ways to lay a grid out in a strip. **C order** (row-major, NumPy's default) stores row 0, then row 1, and so on. **Fortran order** (column-major) stores column 0, then column 1.

![The same 3 × 4 array in both layouts, and the path memory takes through it.](assets/images/book/m02_c_vs_f.png){width=100}

> [!STORY] Why two orders exist
> The split goes back to the 1950s. **Fortran**, developed at IBM by John
> Backus's team and released in 1957, was the first widely used high-level
> language, built for scientists and engineers, and it stored arrays column
> by column. **C**, created at Bell Labs in the early 1970s, chose row by row.
> Both choices are perfectly sensible; they just disagree. Because NumPy's
> heaviest linear algebra still runs on battle-tested Fortran libraries such
> as LAPACK, NumPy has to speak both dialects, and it's why
> `np.asfortranarray` exists at all.

### Why the order matters: cache locality

CPUs don't fetch single numbers from memory; they fetch **cache lines** of 64 bytes at a time. When you read data that sits contiguously, every fetch delivers eight useful `float64`s. When you stride across memory, most of each fetch is thrown away.

On a 4000 × 4000 matrix, the notebook times `sum(axis=0)` and `sum(axis=1)` on a C-ordered and on a Fortran-ordered copy. The pattern is consistent: **each layout is faster along the axis it stores contiguously.**

> [!NOTE]
> Grace Hopper, a pioneer of programming languages, used to hand audiences a
> piece of wire about 30 cm (11.8 inches) long: the distance light travels in
> one nanosecond. Her point was that distance costs time, even inside a
> computer. Memory far from the processor is slow to reach, which is exactly
> why caches exist, and why reading memory in order matters so much.

> [!TIP]
> You rarely choose the order on purpose, but you inherit it. `arr.T`, some
> SciPy routines, and data from column-oriented sources can all hand you
> non-C-contiguous arrays. If a downstream step is unexpectedly slow, check
> `.flags` and consider `np.ascontiguousarray`. It pays a copy up front, so
> it's worth it only if you'll use the result several times; the notebook
> measures the break-even point.

## Temporaries: the hidden allocations

Every intermediate in an expression allocates a whole new array. A chain of operations on a big array can use several times the memory you'd expect:

![`y = (x * 2 + 1) ** 2` allocates two throw-away arrays as big as `x`. With `out=` and one reused buffer, there are none.](assets/images/book/m02_temporaries.png){width=100}

```python
def with_temporaries(arr):
    return (arr * 2 + 1) ** 2

def in_place(arr, out):
    np.multiply(arr, 2, out=out)     # no temporaries at all
    np.add(out, 1, out=out)
    np.power(out, 2, out=out)
    return out
```

On 20 million values the in-place version is somewhat faster, but **memory is the real prize**: in-place work is what lets a job fit in RAM at all. When you own the array, the in-place operators say the same thing more readably:

```python
y = x.copy()
y *= 2; y += 1; y **= 2
```

> [!WARNING]
> `y *= 2` mutates in place, so if `y` is a **view** of something else, you've
> just modified the original. Chapter 2's lesson meets the performance advice:
> in-place is efficient *and* dangerous, and you must know which arrays you
> own. `base = np.arange(10.0); view = base[2:5]; view *= 100` quietly
> rewrites `base`.

## Measuring memory honestly

`arr.base is None` tells you whether an array owns its memory. A small profiler that adds up only the arrays that *own* their data gives the true footprint: a slice and a transpose cost nothing; a `.copy()` and an `.astype(np.float32)` each allocate a fresh buffer.

## Arrays larger than RAM: `np.memmap`

A memory-mapped array lives **on disk** and is paged into memory on demand. It behaves like an ndarray but is limited by disk space, not RAM:

```python
mapped = np.memmap(path, dtype=np.float64, mode="w+", shape=(10_000, 500))
for start in range(0, 10_000, 1_000):          # write it in chunks
    mapped[start:start + 1_000] = rng.random((1_000, 500))
mapped.flush()

mapped = np.memmap(path, dtype=np.float64, mode="r", shape=(10_000, 500))
mapped.mean(axis=0)                             # computed without loading it all
```

A **chunked reduction**, accumulating a running sum and count block by block, is the pattern that generalises to genuinely huge files.

## Choosing the right tool

NumPy is fast when the work can be written as whole-array operations. When it can't, the answer is a different tool, not a cleverer array.

Exponential smoothing is a good example: `s[t] = α·x[t] + (1 − α)·s[t−1]`. Each step depends on the previous one, so it can't be vectorised directly, and a Python loop over 200,000 values is slow. But SciPy already implements this recurrence in C, as a digital filter:

```python
from scipy.signal import lfilter

def exponential_smoothing(values, alpha):
    zi = [(1 - alpha) * values[0]]      # start exactly where the loop starts
    smoothed, _ = lfilter([alpha], [1, -(1 - alpha)], values, zi=zi)
    return smoothed
```

It matches the loop exactly and runs vastly faster. Note the `zi` argument: leave the filter's initial state at its default of zero and the two definitions disagree from the first step, leaving a transient that decays but never fully disappears. Small details matter when you swap implementations.

![The escalation ladder: work down it before reaching for a bigger machine.](assets/images/book/m02_escalation.png){width=100}

### Sparse matrices

When almost every entry is zero, storing the zeros is pure waste. A 5000 × 5000 matrix with 0.1% non-zero entries takes 200 MB dense and well under 1 MB as a `scipy.sparse.csr_matrix`.

> [!NOTE]
> One-hot encoded categories and text TF-IDF matrices are almost always more
> than 99% zeros. `scipy.sparse` is why a bag-of-words model over a
> 50,000-word vocabulary fits in memory at all (Module 18).

## A performance checklist

| Check | Action |
| :--- | :--- |
| Is there a Python loop over elements? | Vectorise it |
| Calling built-in `sum`/`max`/`min` on an array? | Use `np.sum`, `arr.max()` |
| Is the dtype wider than needed? | `float32` halves memory, often time too |
| Allocating temporaries in a hot loop? | `out=` or in-place operators |
| Is the array C-contiguous? | Check `.flags`; `np.ascontiguousarray` if reused |
| Reducing across the slow axis? | Prefer `axis=-1` on C-ordered data |
| Sorting just to get the top *k*? | `np.argpartition` |
| Building a huge broadcast intermediate? | `einsum`, algebra, or chunking |
| Is the matrix mostly zeros? | `scipy.sparse` |
| Does it still not fit? | `memmap`, then Dask or Spark |

## Try it yourself

Exercises 8.1 to 8.5 in notebook 08; solutions in notebook 10.

1. **Predict the strides** for `a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)`, `a.T`, `a[::2]`, `a[:, ::2, :]` and `a.reshape(6, 4)`. Then verify.
2. **Row-major versus column-major.** Time `sum(axis=0)` and `sum(axis=1)` on a 3000 × 3000 array in both orders. Predict the fastest combination first, and explain it with cache lines.
3. **Eliminate the temporaries.** Rewrite `np.sqrt(a**2 + b**2) / (c + 1)` with `out=` and at most two buffers.
4. **Chunked statistics.** Compute the mean and standard deviation of a memmapped array **in one pass** by accumulating count, sum and sum of squares, and mind the catastrophic cancellation from Chapter 3.
5. **Know when to stop.** Implement a cumulative maximum three ways (Python loop, `np.maximum.accumulate`, chunked) on 10 million elements, and explain the ranking.

> [!KEY]
> - An ndarray is a **flat buffer plus strides**. Reshape and transpose only rewrite the strides, which is why they're free.
> - **Contiguity governs cache behaviour.** Reduce along the axis the array stores contiguously; check `.flags` when something is mysteriously slow.
> - Every intermediate **allocates a full array**. Use `out=` or in-place operators when memory is tight.
> - **In-place operations on a view mutate the original.**
> - `.base is None` tells you whether an array owns its memory.
> - **`np.memmap`** handles arrays bigger than RAM; chunked reductions generalise the pattern.
> - For a genuinely sequential loop, look for a **SciPy routine** before reaching for Numba.
> - **`scipy.sparse`** for matrices that are mostly zeros.
