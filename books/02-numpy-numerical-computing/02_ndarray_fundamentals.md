# The ndarray, Up Close
@subtitle Creation, shape, indexing, and the views-versus-copies rule
@time About 90 minutes
@notebook 02_ndarray_fundamentals.ipynb
@learn
- Every way to create an array, and which one to reach for.
- `shape`, `ndim`, `size`, `dtype`: the four attributes you'll check constantly.
- Indexing and slicing in N dimensions.
- **Views versus copies**, the rule that *reverses* what you learned about lists, and the biggest source of silent NumPy bugs.
- Reshaping, transposing, stacking and splitting.
@end

@lead Learning a new tool is a bit like moving into a new kitchen. Most things are where you'd expect, and then one drawer opens the other way. In NumPy, that drawer is slicing: the same `[1:3]` you used on lists all through Module 01 behaves completely differently on an array. We'll get there. First, the basics.

## Creating arrays

### From Python data

```python
np.array([1, 2, 3, 4])               # dtype int64, shape (4,)
np.array([1.0, 2.0, 3.0])            # dtype float64
np.array([[1, 2, 3], [4, 5, 6]])     # nested lists → 2-D, shape (2, 3)
np.array([1, 2, 3], dtype=np.float64)   # explicit dtype: a good habit
```

Being explicit about the dtype documents intent and prevents the surprise of getting integers when you needed floats.

### From scratch

```python
np.zeros(3)            # [0. 0. 0.]
np.ones((2, 3))        # a 2×3 block of ones
np.full((2, 2), 7)     # filled with 7
np.eye(3)              # the identity matrix
np.empty(3)            # UNINITIALISED memory, see below
```

> [!WARNING]
> `np.empty` does **not** zero its memory. It hands you whatever bytes happened
> to be there already. It's marginally faster than `zeros` and is the right
> choice only when you're about to overwrite every element. Forget to fill it
> and you get garbage that looks plausible.

### Ranges and grids

```python
np.arange(0, 10, 2)      # [0 2 4 6 8]   like range(), but an array
np.linspace(0, 1, 5)     # [0. 0.25 0.5 0.75 1.]   N points, BOTH ends included
np.logspace(-3, 0, 4)    # [0.001 0.01 0.1 1.]   ideal for hyperparameter grids
```

> [!TIP]
> **Prefer `linspace` to `arange` for floats.** `arange` steps by repeated
> addition, so floating-point error accumulates and the endpoint becomes
> unpredictable. `np.arange(0, 0.3, 0.1)` gives *three* values, `0.3`
> sneaking in at the end because `0.1 + 0.1 + 0.1` is slightly less than
> `0.3` in binary. `linspace` asks for a count and gets exact ends.

### Random arrays

Always use the modern **`Generator`** API. The legacy `np.random.seed` / `np.random.rand` functions share one global state, which makes reproducibility fragile as soon as two pieces of code draw numbers (Chapter 7 shows exactly how).

```python
rng = np.random.default_rng(seed=42)
rng.random(3)                  # uniform on [0, 1)
rng.normal(0, 1, 3)            # standard normal
rng.integers(0, 10, 5)         # integers in [0, 10)
rng.choice(["a", "b"], 4)
```

Same seed, same numbers, on every machine.

## The four attributes to check first

When something goes wrong in NumPy code, the answer is almost always in one of these. Print them before you debug anything else:

```python
arr = rng.normal(size=(3, 4, 2))
arr.shape      # (3, 4, 2)  size along each dimension
arr.ndim       # 3          number of dimensions
arr.size       # 24         total elements
arr.dtype      # float64    type of every element
arr.nbytes     # 192        size × itemsize
```

### 1-D is not 2-D

`(3,)`, `(3, 1)` and `(1, 3)` hold the same three numbers and behave completely differently under broadcasting (Chapter 4) and matrix multiplication (Chapter 6).

![Three shapes, three behaviours. A 1-D array has no rows and no columns; a column and a row are both 2-D.](assets/images/book/m02_shapes.png){width=100}

A taste of the trouble ahead: `np.array([1, 2, 3]) + np.array([[1], [2], [3]])` doesn't give three numbers. It gives a 3×3 matrix. Chapter 4 explains exactly why.

## Indexing and slicing

```python
arr = np.arange(1, 13).reshape(3, 4)
arr[0, 0]      # 1    one pair of brackets: [row, col]
arr[2, 3]      # 12
arr[-1, -1]    # 12   negative indexes work on every axis
```

`arr[1][2]` also works, but it's slower: it builds an intermediate row array first, then indexes that. Write `arr[1, 2]`.

Slicing works per axis, separated by commas:

![Four common selections on the same 3×4 array. Highlighted cells are the ones returned.](assets/images/book/m02_indexing.png){width=100}

```python
arr[0]         # first row
arr[:, 0]      # first column
arr[1:, 2:]    # bottom-right block
arr[::2]       # every other row
arr[:, ::-1]   # columns reversed
```

### `...`: "all the remaining axes"

The Ellipsis saves typing when you only care about the first or last dimension: on a `(2, 3, 4)` array, `cube[0, ...]` is `cube[0, :, :]` and `cube[..., 0]` is `cube[:, :, 0]`.

## Views versus copies: read this twice

![A slice of an array is a view onto the same memory. Writing through it changes the original.](assets/images/views_vs_copies.png){width=92}

In Module 01 you learned that slicing a list **copies**. In NumPy, **basic slicing returns a view**: a second window onto the *same* memory. Write through the view, and the original changes.

```python
lst = [1, 2, 3, 4]
part = lst[1:3]; part[0] = 99
lst                      # [1, 2, 3, 4]   unchanged: a copy was made

arr = np.array([1, 2, 3, 4])
part = arr[1:3]; part[0] = 99
arr                      # [ 1 99  3  4]  CHANGED: part was a view
```

It's a deliberate design decision. It makes slicing free no matter how big the array is: no data is copied, only a new description of where to look. It is also the source of a whole category of silent bugs.

Think of a view as a window cut into a wall. Looking through a different window doesn't give you a different room. Paint the wall you see through it, and it's painted for everyone.

### How to tell

```python
original = np.arange(10)
view = original[2:5]
copy = original[2:5].copy()

view.base is original              # True: a view borrows its memory
copy.base                          # None: a copy owns its data
np.shares_memory(original, view)   # True
np.shares_memory(original, copy)   # False
```

### What gives a view, what gives a copy

| Operation | Result |
| :--- | :--- |
| `arr[1:5]`, `arr[:, 0]` (basic slicing) | **view** |
| `arr.reshape(...)`, `arr.ravel()` | **view** *when possible* |
| `arr.T`, `arr.transpose()` | **view** |
| `arr[[0, 2, 4]]` (fancy indexing) | **copy** |
| `arr[arr > 3]` (boolean mask) | **copy** |
| `arr.flatten()`, `arr.copy()` | **copy**, always |
| `arr + 1`, `np.sqrt(arr)` | **new array** |

### The bug this causes in practice

```python
def normalise_first_column_broken(data: np.ndarray) -> np.ndarray:
    column = data[:, 0]          # a view
    column -= column.mean()      # in-place → writes into `data`
    return data

def normalise_first_column(data: np.ndarray) -> np.ndarray:
    out = data.copy()            # work on our own copy
    out[:, 0] = out[:, 0] - out[:, 0].mean()
    return out
```

The broken version silently rewrites the caller's array.

> [!WARNING]
> Notice *why* it bites: `column -= ...` is an **in-place** operation on a
> view. Had it been `column = column - column.mean()`, it would merely have
> rebound a local name and changed nothing at all. It's the `+=` versus
> `= ... +` distinction from Module 01's Chapter 2, with higher stakes.

## Reshaping

```python
flat = np.arange(12)
flat.reshape(3, 4)       # 3 rows, 4 columns
flat.reshape(2, -1)      # -1 means "work it out": (2, 6)
```

Use `-1` so the code survives a change in the number of rows.

The most common reshape in machine learning turns a 1-D array into a single-column 2-D array, because scikit-learn demands 2-D feature matrices. Three spellings, all views:

```python
feature = np.array([1.0, 2.0, 3.0, 4.0])
feature.reshape(-1, 1).shape     # (4, 1)
feature[:, np.newaxis].shape     # (4, 1)
feature[:, None].shape           # (4, 1), the terse version
```

### `ravel` versus `flatten`

Both give you a 1-D array. `ravel` returns a view when it can (cheap, but writes go through to the original); `flatten` always copies (safe).

### Transposing

`m.T` swaps the axes and costs nothing: it only reinterprets the strides (Chapter 8). But the result is no longer C-contiguous, which can slow later operations.

## Combining and splitting

```python
a = np.array([[1, 2], [3, 4]]); b = np.array([[5, 6], [7, 8]])
np.vstack([a, b])                    # stack rows → (4, 2)
np.hstack([a, b])                    # stack columns → (2, 4)
np.concatenate([a, b], axis=0)       # the general form
np.stack([a, b]).shape               # (2, 2, 2): a NEW leading axis
```

`stack` differs from `concatenate`: it adds a new axis rather than extending an existing one. That's how you build a batch out of individual samples.

Splitting is the inverse. Splitting at explicit indices is how a train/validation/test split works before scikit-learn does it for you:

```python
data = np.arange(12).reshape(6, 2)
train, validate, test = np.split(data, [4, 5])    # 4, 1 and 1 rows
```

## A practical checklist

When an array misbehaves, run through the same questions every time. The notebook wraps them in a `describe(arr)` helper worth keeping in your back pocket: shape, ndim, dtype, nbytes, whether it's a view (`arr.base is not None`), contiguity (`arr.flags["C_CONTIGUOUS"]`), whether it's writeable, whether every value is finite, and its range.

> [!STORY] What's in a name?
> "ndarray" stands for *N-dimensional array*, and the word "matrix" for a
> rectangular block of numbers is itself only about 170 years old. The
> English mathematician James Joseph Sylvester coined it in 1850, from the
> Latin for "womb": a matrix was the thing from which smaller determinants
> are *born*. A few years later his friend Arthur Cayley worked out the
> algebra of matrices: how to add them, multiply them and invert them. Every
> `reshape` and `@` you write rests on their Victorian groundwork.

## Try it yourself

Exercises 2.1 to 2.5 in notebook 02; solutions in notebook 10.

1. **Predict the shapes.** For `a = np.arange(24).reshape(2, 3, 4)`, give the shapes of `a[0]`, `a[:, 1]`, `a[..., 2]`, `a[0, :, ::2]`, `a.T` and `a.reshape(4, -1)` without running anything.
2. **View or copy?** For `a = np.arange(20).reshape(4, 5)`, decide whether each shares memory with `a`, then check with `np.shares_memory`: `a[1:3]`, `a[:, [0, 1]]`, `a.T[0]`, `a[a % 2 == 0]`, `a.reshape(5, 4)`, `a.ravel()[::2]`.
3. **Fix the mutation bug.** A `scale_features(X, factor)` loops over columns doing `col = X[:, j]; col *= factor`. Show it corrupts the caller's array, rewrite it, and add an assertion that would have caught it.
4. **Build a checkerboard.** Make an 8×8 array of alternating 0s and 1s with no Python loop, then a second way. Hint: slicing with a step, or `np.indices`.
5. **Reshape for scikit-learn.** Turn `y = np.array([1.0, 2.0, 3.0])` into shape `(n, 1)` three ways and confirm all three share memory with `y`.

> [!KEY]
> - Check **`shape`, `ndim`, `dtype`, `size`** first when debugging. Most NumPy bugs are shape bugs.
> - `(3,)`, `(3, 1)` and `(1, 3)` are three different things.
> - Prefer **`linspace` over `arange`** for float ranges.
> - Use **`np.random.default_rng(seed)`**, never the legacy global functions.
> - **Basic slicing returns a view**, the opposite of list slicing. Fancy indexing and boolean masks return copies.
> - Check with **`.base`** or **`np.shares_memory`**; force independence with **`.copy()`**.
> - In-place operators (`-=`, `*=`) on a view write through to the original.
> - `reshape(-1, 1)` makes the column vector scikit-learn wants.
