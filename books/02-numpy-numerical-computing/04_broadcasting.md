# Broadcasting
@subtitle Two rules for combining arrays of different shapes
@time About 75 minutes
@notebook 04_broadcasting.ipynb
@learn
- The two rules that completely decide whether two shapes are compatible.
- How to read, and fix, a broadcasting error.
- `np.newaxis` and `reshape` to make your intent explicit.
- The silent bug: shapes that broadcast happily into the wrong answer.
- Real applications: standardising features, pairwise distances, one-hot encoding.
- The memory trap, and how algebra, `einsum` and chunking avoid it.
@end

@lead Imagine handing out name badges at a conference. You don't print a separate copy of the event logo for each badge; you print the logo once and stamp it onto every one. Broadcasting is NumPy's version of that stamp: a smaller array is *virtually* stretched to match a bigger one, without ever being copied.

## The two rules

Broadcasting lets NumPy combine arrays of different shapes without you writing a loop or materialising a copy. There are exactly two rules.

![The two broadcasting rules: align shapes from the right, then compare each pair of dimensions.](assets/images/broadcasting_rules.png){width=92}

1. **Align the shapes from the right.** If one is shorter, pad it on the **left** with 1s.
2. **Two dimensions are compatible if they're equal, or if one of them is 1.** A dimension of size 1 is *stretched* to match the other.

If any pair fails rule 2, the operation raises. That's the whole thing.

```python
arr = np.array([1, 2, 3, 4])
arr + 10        # a scalar is shape (), padded to (1,), stretched to (4,)

matrix = np.arange(12).reshape(3, 4)
row = np.array([10, 20, 30, 40])
matrix + row    # (4,) → (1, 4) → stretched down all 3 rows
```

![A column of shape (4, 1) times a row of shape (1, 5). The dashed cells are never stored; NumPy reads the same values again.](assets/images/book/m02_broadcast_grid.png){width=100}

### Checking compatibility without computing anything

`np.broadcast_shapes` answers "will these combine, and into what?". It allocates nothing, so it's perfect for debugging:

| Shape A | Shape B | Result |
| :--- | :--- | :--- |
| `(3, 4)` | `(4,)` | `(3, 4)` |
| `(3, 4)` | `(3, 1)` | `(3, 4)` |
| `(3, 1)` | `(1, 4)` | `(3, 4)` |
| `(2, 3, 4)` | `(3, 4)` | `(2, 3, 4)` |
| `(5, 1, 3)` | `(4, 3)` | `(5, 4, 3)` |
| `(3, 4)` | `(3,)` | **incompatible** |
| `(2, 3)` | `(3, 2)` | **incompatible** |

## Reading the error message

```python
np.arange(12).reshape(3, 4) + np.array([1, 2, 3])
# ValueError: operands could not be broadcast together with shapes (3,4) (3,)
```

Decode it by aligning from the right, just as NumPy does:

```text
(3, 4)
   (3)   →  padded to (1, 3)
─────────
 4 vs 3  →  neither is 1  →  fail
```

The intent was almost certainly "add one value per **row**", which means `b` needs to be a **column**: `a + b.reshape(-1, 1)` works, and each row gets its own value.

## Say what you mean: `np.newaxis`

`np.newaxis` (an alias for `None`) inserts a dimension of size 1. It's the clearest way to say "treat this as a row" or "treat this as a column":

```python
v = np.array([1, 2, 3])
v[np.newaxis, :].shape    # (1, 3)  a ROW
v[:, np.newaxis].shape    # (3, 1)  a COLUMN
v[:, None] * v[None, :]   # the outer product, (3, 3); same as np.outer(v, v)
```

## The silent bug

The dangerous case isn't the one that raises. It's the one that **broadcasts successfully into a shape you never intended**.

```python
predictions = np.array([1.0, 2.0, 3.0, 4.0])           # (4,)
actuals = np.array([[1.1], [2.1], [2.9], [4.2]])       # (4, 1): a column!

errors = predictions - actuals                          # (4, 4)  ← surprise
```

![Every prediction subtracted from every actual. Only the diagonal is meaningful, yet the mean squared error averages all sixteen cells.](assets/images/book/m02_silent_bug.png){width=100}

The "MSE" computed from that matrix is several times too large, and nothing raised.

> [!WARNING]
> This is the most common broadcasting bug in machine-learning code. It
> usually arrives as an `(n, 1)` column from `reshape(-1, 1)`, or from a
> DataFrame column meeting a 1-D prediction array. Nothing raises; the loss is
> simply wrong.

**The defence: assert the shape you expect.** A metric function that flattens both inputs and checks they match makes the bug impossible:

```python
def mean_squared_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(((y_true - y_pred) ** 2).mean())
```

## Real applications

### Standardising features: what `StandardScaler` does

```python
X = rng.normal(loc=[0, 100, -50, 1000], scale=[1, 25, 10, 200], size=(1000, 4))
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

`X.mean(axis=0)` has shape `(4,)`: one mean per column. It pads to `(1, 4)` and stretches down all 1,000 rows, which is exactly right. Every column ends up with mean 0 and standard deviation 1.

> [!TIP]
> `axis=0` is what makes this work: it collapses the **rows**, leaving one
> statistic per column. Use the wrong axis and you'd normalise each *sample*
> against itself, which on a square matrix broadcasts without complaint and
> means nothing. Chapter 5 is all about `axis=`.

### Pairwise distances without a loop

A staple of clustering, nearest neighbours and recommender systems:

```python
points = rng.random((5, 2)) * 10
diffs = points[:, None, :] - points[None, :, :]    # (5, 1, 2) − (1, 5, 2) → (5, 5, 2)
distances = np.sqrt((diffs**2).sum(axis=-1))        # (5, 5)
```

Two sanity checks should always hold for a distance matrix, and are worth asserting: the diagonal is zero, and the matrix is symmetric. The notebook also confirms the result matches `scipy.spatial.distance.cdist`.

> [!STORY] The nearest neighbour, 1951
> In 1951 two statisticians at the University of California, Berkeley,
> Evelyn Fix and Joseph Hodges, wrote a technical report for the US Air
> Force School of Aviation Medicine on a strikingly simple idea: to classify
> a new observation, look at the training points closest to it and take a
> vote. It needed no assumptions about the shape of the data, only a way to
> measure distance. The **k-nearest-neighbours** rule became one of the most
> widely used methods in machine learning, and its heart is the pairwise
> distance matrix you just computed in two lines.

### One-hot encoding in one line

```python
labels = np.array([0, 2, 1, 2, 0, 1])
one_hot = (labels[:, None] == np.arange(3)).astype(np.int8)    # (6, 3)
```

`labels[:, None]` is `(6, 1)`; `np.arange(3)` is `(3,)`. Broadcasting compares every label against every class index, giving exactly one 1 per row.

## The memory trap

Broadcasting doesn't materialise the *inputs*; that's the whole point. But the **result** is fully materialised, and it can be enormous:

![Inputs grow linearly; the result of combining every pair grows with the square. Two 10,000-element vectors make an 800 MB matrix; at 100,000 elements, 80 GB.](assets/images/book/m02_memory_trap.png){width=64}

### When you only need a reduction, don't build the intermediate

**Algebra first.** The sum of all pairwise products, `(a[:, None] * b[None, :]).sum()`, builds an `n × n` matrix only to collapse it. But Σᵢ Σⱼ aᵢbⱼ = (Σᵢ aᵢ)(Σⱼ bⱼ), so `a.sum() * b.sum()` gives the same number with no intermediate at all.

**`np.einsum`** (Einstein summation) expresses "multiply these axes and sum over that one" in a single pass:

```python
np.einsum("ij,jk->ik", A, B)     # matrix product, same as A @ B
np.einsum("ij,ij->i", P, Q)      # row-wise dot products, no (n, m) temporary
```

**Chunking**, when the big result is genuinely what you need: compute pairwise distances a block of rows at a time, so peak memory is `chunk_size × n` instead of `n²`:

```python
def pairwise_distances_chunked(X, chunk_size=500):
    n = X.shape[0]
    out = np.empty((n, n), dtype=X.dtype)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = X[start:stop, None, :] - X[None, :, :]
        out[start:stop] = np.sqrt((block**2).sum(axis=-1))
    return out
```

## Try it yourself

Exercises 4.1 to 4.5 in notebook 04; solutions in notebook 10.

1. **Compatible or not?** Give the result shape or "incompatible" for `(5,4)&(4,)`, `(5,4)&(5,)`, `(5,4)&(5,1)`, `(3,1,4)&(2,4)`, `(2,3,4)&(3,1)`, `(8,1,6)&(7,1)`. Check with `np.broadcast_shapes`.
2. **Fix the silent bug.** `y_true` has shape `(3, 1)` and `y_pred` shape `(3,)`. What shape is `y_true - y_pred`, and what should it be? Fix it and add an assertion.
3. **Normalise rows.** Scale each row of a `(100, 5)` matrix to sum to 1. Explain why `X / X.sum(axis=1)` fails and `X / X.sum(axis=1, keepdims=True)` works.
4. **Distance to centroids.** With `X` of shape `(1000, 3)` and `centroids` of shape `(8, 3)`, compute a `(1000, 8)` distance matrix by broadcasting, then each point's nearest centroid. That's one full iteration of k-means, which you'll reuse in the project.
5. **Avoid the intermediate.** `((a[:, None] - b[None, :])**2).sum()` needs 20 GB for 50,000-element vectors. Expand the algebra and compute the same number using only `sum`, `dot` and scalars.

> [!KEY]
> - **Align right, pad left with 1s; dimensions must match or be 1.** That's all of broadcasting.
> - **`np.broadcast_shapes`** checks compatibility without allocating.
> - **`arr[:, None]` makes a column; `arr[None, :]` makes a row.** Say which you mean.
> - The dangerous case **succeeds with the wrong shape**: `(n,)` meeting `(n, 1)` gives `(n, n)` and no error.
> - **Assert your shapes** in any function that computes a metric or a loss.
> - Broadcasting doesn't copy the inputs, but **it does materialise the result**.
> - When you only need a reduction, use **algebra**, **`einsum`** or **chunking**.
