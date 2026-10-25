# Summarise, Select, Decide
@subtitle axis= demystified, boolean masks, and loops you no longer need
@time About 90 minutes
@notebook 05_aggregations_and_masking.ipynb
@learn
- `axis=` demystified, with the one mental model that makes it obvious.
- `keepdims`, and why it exists.
- Boolean masks: the vectorised replacement for an `if` inside a loop.
- `np.where`, `np.select`, `np.clip`: conditional logic without branching.
- Fancy indexing, `argsort` and `argpartition`.
- Counting and grouping: `unique`, `bincount`, `histogram`.
@end

@lead Most of what an analyst does with numbers comes down to three verbs: *summarise* them (a total, an average, a spread), *select* some of them (only the paid customers, only the outliers), and *decide* something per value (clip it, label it, replace it). This chapter does all three without a single Python loop.

## `axis=`: the only rule you need

Almost everyone gets this backwards at first, because "axis 0" sounds like "do it to the rows". The correct reading is the opposite:

@lead **`axis` names the dimension that disappears.**

![arr.sum(axis=0) collapses axis 0, the rows, leaving one value per column. axis=1 collapses the columns, leaving one value per row.](assets/images/axis_semantics.png){width=92}

```python
arr = np.arange(1, 13).reshape(3, 4)
arr.sum()           # 78: everything collapses to a scalar
arr.sum(axis=0)     # [15 18 21 24]  shape (4,): the rows vanished
arr.sum(axis=1)     # [10 26 42]     shape (3,): the columns vanished
```

The shape arithmetic makes it mechanical: start from the shape tuple, **delete the entry at position `axis`**, and that's your result shape. On a `(2, 3, 4)` cube, `axis=0` gives `(3, 4)`, `axis=1` gives `(2, 4)`, `axis=2` gives `(2, 3)`, and `axis=(0, 1)` gives `(4,)`.

### The reductions worth knowing

`sum`, `mean`, `std`, `var`, `min`, `max`, `median`, `percentile`, and the less famous `ptp` ("peak to peak", `max − min`) all take `axis=`.

> [!TIP]
> A `ptp` of 0 tells you instantly that a feature is constant, and therefore
> useless to a model.

### `argmin` / `argmax`: *where*, not *what*

```python
scores = np.array([0.82, 0.91, 0.78, 0.95, 0.88])
models = np.array(["ridge", "lasso", "tree", "forest", "boosting"])
models[scores.argmax()]          # 'forest': argmax returns an INDEX
```

In 2-D, `argmax` flattens the array by default, a classic surprise. `np.unravel_index(m.argmax(), m.shape)` converts the flat index back to `(row, col)`, and `m.argmax(axis=1)` gives the best column per row.

### `keepdims`: the argument that saves your broadcasts

A reduction removes a dimension, which then breaks the broadcast you wanted to do next. `keepdims=True` leaves a length-1 dimension in its place:

```python
X = rng.random((4, 3))
X.sum(axis=1).shape                   # (4,)
X.sum(axis=1, keepdims=True).shape    # (4, 1)

X / X.sum(axis=1)                     # ValueError: (4,3) vs (4,)
X / X.sum(axis=1, keepdims=True)      # ✔ every row sums to 1
```

> [!WARNING]
> Without `keepdims`, `(4, 3) / (4,)` raises here, which is lucky. On a
> **square** matrix the same mistake broadcasts silently and divides by the
> wrong axis. Use `keepdims=True` whenever a reduction feeds a broadcast.

## Boolean masks

A comparison on an array produces an array of booleans. That **mask** can then select, count or replace, with no loop and no `if`:

![A comparison produces a mask; the mask selects. Because True counts as 1, the mask's sum is a count and its mean is a proportion.](assets/images/book/m02_mask.png){width=100}

```python
values = np.array([12, 45, 7, 88, 23, 61, 4, 39])
mask = values > 30
values[mask]       # [45 88 61 39]   a COPY, not a view
mask.sum()         # 4               True is 1, so sum counts
mask.mean()        # 0.5             and mean gives the share
```

### Combining conditions: mind the parentheses

NumPy uses `&` (and), `|` (or) and `~` (not) for element-wise logic. Python's `and`, `or` and `not` **don't work**: they try to treat the whole array as one truth value, raising the same ambiguity error you met in Module 01.

```python
values[(values > 20) & (values < 60)]     # [45 23 39]
values[(values < 10) | (values > 80)]     # [ 7 88  4]
values[~mask]                             # the complement
```

> [!WARNING]
> **`&` binds more tightly than `>` in Python.** Without parentheses,
> `values > 20 & values < 60` is parsed as `values > (20 & values) < 60`,
> which raises or, worse, computes something else entirely. Always wrap each
> comparison in parentheses.

### `any` and `all`

```python
readings = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
np.isnan(readings).any()            # is there any NaN at all?
bad_rows = np.isnan(readings).any(axis=1)
readings[~bad_rows]                 # keep only complete rows
```

### Assigning through a mask

```python
temperatures = np.array([18.0, -999.0, 22.5, -999.0, 25.1])
cleaned = temperatures.copy()
cleaned[cleaned == -999.0] = np.nan     # -999 = "sensor failed" sentinel
np.nanmean(cleaned)                     # 21.87, not the naive −386.48
```

Sentinel values like −999 are a real and common convention. Averaging them as if they were temperatures is catastrophically wrong, and the notebook shows by how much.

## Decisions without branching

### `np.where`: a vectorised if/else

```python
values = np.array([-5, 3, -2, 8, 0, -1])
np.where(values < 0, 0, values)                   # ReLU in one line
np.where(values > 0, "positive", "non-positive")  # all three arguments broadcast
np.where(values < 0)[0]                           # one argument: the INDICES
```

### `np.select`: more than two branches

```python
scores = np.array([95, 72, 58, 88, 41, 67])
np.select([scores >= 90, scores >= 70, scores >= 60], ["A", "B", "C"], default="F")
# ['A' 'B' 'F' 'B' 'F' 'C']
```

Conditions are checked **in order**; the first match wins.

### `np.clip`: keep values inside a range

```python
np.clip(raw, 0, 100)       # bound both ends
np.clip(raw, 0, None)      # lower bound only
```

Clipping at *percentiles* rather than fixed values is called **winsorising**, the technique behind the `OutlierClipper` estimator you wrote in Module 01:

```python
low, high = np.percentile(data, [1, 99])
winsorised = np.clip(data, low, high)
```

## Fancy indexing

Indexing with an array of **integers** picks arbitrary elements in any order, repeats allowed. Unlike slicing, it always returns a **copy**.

```python
arr = np.array([10, 20, 30, 40, 50])
arr[[0, 2, 4]]      # [10 30 50]
arr[[4, 4, 0]]      # [50 50 10]  repeats and any order

matrix = np.arange(1, 13).reshape(3, 4)
matrix[[0, 1, 2], [3, 2, 1]]    # pairs up: (0,3), (1,2), (2,1) → [4 7 10]
```

Selecting whole rows with a permutation is how you shuffle a dataset, and using the *same* permutation for features and labels keeps them aligned:

```python
perm = rng.permutation(len(y))
X_shuffled, y_shuffled = X[perm], y[perm]
```

### `argsort`: sort one array by another

```python
order = revenue.argsort()[::-1]        # indices, largest revenue first
for rank, i in enumerate(order, start=1):
    print(rank, names[i], revenue[i])
```

### `argpartition`: the top *k* without a full sort

When you only need the largest few, sorting everything is wasted work. `argpartition` is **O(n)** where `argsort` is **O(n log n)**. On two million values, picking the top 10 is many times faster:

```python
top_k = np.argpartition(large, -10)[-10:]   # the 10 largest, in no particular order
```

## Counting and grouping

```python
values, counts = np.unique(channels, return_counts=True)   # NumPy's Counter
np.bincount(channels)              # faster, for small non-negative integers

categories, encoded = np.unique(labels, return_inverse=True)
categories[encoded]                # label encoding that round-trips exactly
```

### A group-by from first principles

`bincount` with `weights` sums the weights that fall in each bin, which is a **group-by sum in one call**. Understanding this makes pandas' `groupby` (Module 03) far less mysterious:

```python
group_ids = np.array([0, 1, 0, 2, 1, 0, 2])
amounts = np.array([10.0, 20.0, 5.0, 8.0, 12.0, 3.0, 15.0])

group_sums = np.bincount(group_ids, weights=amounts)   # [18. 32. 23.]
group_counts = np.bincount(group_ids)                  # [3 2 2]
group_means = group_sums / group_counts
```

The notebook cross-checks the result against pandas, and it matches.

> [!STORY] The nurse who grouped by cause
> In the Crimean War of the 1850s, Florence Nightingale went to the British
> Army hospital at Scutari and kept meticulous records of how soldiers died.
> When she grouped the deaths **by cause and by month**, the picture was
> damning: far more men were dying of preventable diseases (cholera, typhus,
> dysentery) than of their wounds. She presented the aggregated figures to
> the government in 1858 with her now-famous "rose" diagrams, and they helped
> drive sweeping sanitary reforms. She later became the first woman elected
> a Fellow of the Royal Statistical Society. Her method was, at heart, a
> `groupby` followed by a well-chosen chart.

### Binning with `digitize` and `histogram`

```python
bin_edges = np.array([18, 30, 45, 60, 100])
counts, _ = np.histogram(ages, bins=bin_edges)      # how many per age band
np.digitize(ages, bin_edges)                        # which band each age is in
```

## Putting it together: a vectorised quality report

The notebook combines everything above into one function that profiles a numeric feature matrix with no loops at all. For each column it reports the share missing, mean, standard deviation, min, median, max, the share of outliers by the **1.5 × IQR rule**, and whether the column is constant:

```python
q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
iqr = q3 - q1
is_outlier = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
outlier_share = np.nansum(is_outlier, axis=0) / len(X) * 100
```

Run on a deliberately messy matrix (one well-behaved column, one skewed, one constant, one with injected missing values), it flags each problem immediately.

![The 1.5 × IQR rule on right-skewed data: a box from the first to the third quartile, whiskers out to 1.5 times its width, and everything beyond flagged.](assets/images/book/m02_tukey.png){width=100}

> [!NOTE]
> The 1.5 × IQR fences come from the American statistician **John Tukey**,
> who popularised the box plot in his 1977 book *Exploratory Data Analysis*.
> Tukey argued that looking at data should come *before* testing hypotheses
> about it, an idea so influential that "EDA" is now a stage of every data
> science project. He also coined the word "bit".

Notice what the chart shows, though: on skewed data the rule flags plenty of perfectly genuine values in the long tail. A fence is a prompt to look, not a licence to delete.

## Try it yourself

Exercises 5.1 to 5.5 in notebook 05; solutions in notebook 10.

1. **Axis drill.** For `a = np.arange(24).reshape(2, 3, 4)`, give the shapes of `a.sum(axis=0)`, `a.sum(axis=1)`, `a.sum(axis=2)`, `a.sum(axis=(0, 2))`, `a.mean(axis=1, keepdims=True)` and `a.max(axis=-1)` before running them.
2. **The keepdims trap.** For a `(5, 5)` matrix both `X / X.sum(axis=1)` and `X / X.sum(axis=1, keepdims=True)` run. Show the first is wrong and explain exactly what it computed.
3. **Masking practice.** For `rng.normal(100, 20, 10_000)`: count values beyond two standard deviations, replace them with the mean without a loop, and report the share inside `[80, 120]`.
4. **Group-by from scratch.** `group_stats(keys, values)` returns count, sum, mean and max per key using only NumPy, handling string keys via `np.unique(..., return_inverse=True)`. Check it against pandas.
5. **Top-k per row.** For `scores` of shape `(1000, 50)`, find each user's top 5 items with `argpartition`, returning a `(1000, 5)` array sorted by score. That's the core of a recommender's serving path.

> [!KEY]
> - **`axis` names the dimension that disappears.** Delete it from the shape tuple to get the result shape.
> - **`keepdims=True`** whenever a reduction feeds a broadcast.
> - `argmin`/`argmax` return **indices**, and flatten in 2-D unless you pass an axis.
> - Boolean masks replace `if` inside a loop. Use **`&`, `|`, `~`**, and **parenthesise** every comparison.
> - `mask.sum()` counts; `mask.mean()` gives the proportion.
> - **`np.where`** is vectorised if/else; **`np.select`** handles many branches; **`np.clip`** bounds a range.
> - **Fancy indexing returns a copy**, with repeats and any order allowed.
> - **`argpartition`**, not `argsort`, when you only need the top *k*.
> - **`np.bincount(ids, weights=values)`** is a group-by sum in one call.
