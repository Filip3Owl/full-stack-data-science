# Project: Two Algorithms From Scratch
@subtitle Linear regression and k-means, in nothing but NumPy
@time About 120 minutes
@notebook 09_module_project.ipynb
@learn
- Implementing **linear regression** three ways (SVD, normal equations, gradient descent) and adding Ridge.
- Seeing **multicollinearity** happen, and watching Ridge tame it.
- Implementing **k-means** with k-means++ initialisation and independent restarts.
- Following the **scikit-learn estimator contract**, and checking every result against scikit-learn itself.
- Why being a few times slower than scikit-learn is a *good* result.
@end

@lead There's a particular confidence that comes from having built something yourself. When `LinearRegression().fit()` returns a suspicious coefficient in Module 11, or `KMeans` gives different clusters on every run, you'll know why, because you will have written the loop. This project builds two of the most-used algorithms in data science with nothing but NumPy, and only brings in scikit-learn at the end, as the examiner.

| Chapter | What the project uses from it |
| :--- | :--- |
| 1 | Vectorisation instead of Python loops |
| 2 | Shapes, views versus copies, `reshape(-1, 1)` |
| 3 | `float64` for accumulation, condition numbers |
| 4 | Broadcasting for pairwise distances |
| 5 | `argmin`, boolean masks, `keepdims` |
| 6 | `lstsq`, the normal equations, the SVD |
| 7 | Seeded generators, reproducible initialisation, `SeedSequence.spawn` |
| 8 | Avoiding temporaries; choosing the cheap operation |

## Part 1: linear regression

### The problem

Given a design matrix `X` of shape `(n, p)` and targets `y` of shape `(n,)`, find the coefficients β that minimise the sum of squared residuals, ‖Xβ − y‖².

### The estimator

`LinearRegressionScratch` follows the scikit-learn contract from Module 01: the constructor only stores settings, `fit` learns `coef_` and `intercept_` (trailing underscores) and returns `self`, and `predict` and `score` use what was learned. It's written as a `@dataclass`, with the fitted attributes marked `field(init=False)` so they stay out of the constructor.

```python
@dataclass
class LinearRegressionScratch:
    fit_intercept: bool = True
    alpha: float = 0.0                 # 0 → ordinary least squares; > 0 → Ridge
    solver: str = "lstsq"              # "lstsq", "normal" or "gradient"
    learning_rate: float = 0.01
    n_iterations: int = 1000

    coef_: np.ndarray = field(init=False, default=None)
    intercept_: float = field(init=False, default=0.0)
```

A `_validate` step coerces inputs to `float64` (Chapter 3 showed what `float32` does to accumulation), turns a 1-D `X` into a column, and refuses mismatched row counts with a clear message.

### Three solvers

**`lstsq`, via the SVD.** Robust, and it survives rank deficiency. Ridge is added in an elegant way: stack `√α · I` underneath `X` and zeros underneath `y`, then solve ordinary least squares on the taller system. That's mathematically identical to Ridge and more stable than forming `XᵀX + αI`:

```python
penalty = np.sqrt(self.alpha) * np.eye(n_features)
if self.fit_intercept:
    penalty[0, 0] = 0.0                # never penalise the intercept
Xd = np.vstack([Xd, penalty])
y = np.concatenate([y, np.zeros(n_features)])
beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
```

**`normal`, the normal equations**, solved with `np.linalg.solve`, never with `inv` (Chapter 6).

**`gradient`, batch gradient descent**: start from zeros and repeatedly step downhill along the gradient of the mean squared error. It's how neural networks are trained (Module 17), so it's worth writing once by hand:

```python
for _ in range(self.n_iterations):
    residuals = Xd @ beta - y
    gradient = (2 / n_samples) * (Xd.T @ residuals)
    beta -= self.learning_rate * gradient
```

> [!NOTE]
> Following the slope downhill is an old idea: the French mathematician
> Augustin-Louis Cauchy described the method of steepest descent in 1847, for
> solving systems of equations from astronomy. A century and a half later it
> trains every deep-learning model on the planet.

### Testing it

On 500 simulated samples with known coefficients (intercept 10; slopes 3.0, −1.5, 0.7 and 2.2), all three solvers recover the truth, and the `lstsq` version matches scikit-learn's `LinearRegression` coefficients and R² exactly. Ridge with `alpha=5` matches `sklearn.linear_model.Ridge` too. Gradient descent's loss curve falls steadily onto the closed-form answer.

### Where the solvers differ: multicollinearity

On well-conditioned data the solvers agree. The differences appear exactly where Chapter 6 said they would. The notebook builds a design in which column 2 is column 0 plus a whisper of noise (standard deviation one millionth), and fits both solvers.

Both fit the data equally well, with almost identical R². But the coefficients are wild: the model can't tell columns 0 and 2 apart, so it assigns them enormous, offsetting weights. **This is what multicollinearity looks like**, and it's why the coefficients of an unregularised model on correlated features must never be interpreted.

Ridge fixes it. With `alpha=1.0` the coefficient vector's length becomes small and sensible, and R² is barely any worse.

> [!STORY] Why it's called "regression"
> The word comes from Francis Galton, who in 1886 studied the heights of
> parents and their adult children. Very tall parents tended to have
> children who were tall, but *less* extremely so; very short parents had
> children who were short, but closer to average. Galton called the effect
> "regression towards mediocrity", meaning towards the mean, and the line he
> drew through his data gave the whole method its name. Today we'd call that
> effect regression to the mean, and it's still one of the most common traps
> in interpreting data: an extreme measurement is usually followed by a less
> extreme one, with no cause required.

## Part 2: k-means clustering

### The algorithm

1. Choose `k` initial centroids.
2. **Assign** each point to its nearest centroid.
3. **Update** each centroid to the mean of its assigned points.
4. Repeat steps 2 and 3 until the centroids stop moving.

![Three snapshots of k-means on four blobs: a poor random start, the picture after a single update, and the converged solution. The X marks are the centroids.](assets/images/book/m02_kmeans.png){width=100}

Step 2 is Chapter 4's broadcasting exercise; step 3 is a masked mean per cluster (Chapter 5).

### Distances without a giant tensor

The naive way to compute every point-to-centroid distance builds an `(n, k, p)` difference tensor. For 100,000 points, 10 clusters and 50 features, that tensor alone is 400 MB. Expanding the square avoids it entirely:

‖x − c‖² = ‖x‖² − 2 x·c + ‖c‖²

```python
@staticmethod
def _squared_distances(X, centroids):
    x_norm = (X**2).sum(axis=1, keepdims=True)    # (n, 1)
    c_norm = (centroids**2).sum(axis=1)            # (k,)
    cross = X @ centroids.T                        # (n, k)
    distances = x_norm - 2 * cross + c_norm        # broadcasts to (n, k)
    return np.maximum(distances, 0.0)              # clip tiny negatives
```

The result needs just 8 MB. Broadcasting, `keepdims` and a matrix product, all in five lines. The final `np.maximum` is Chapter 3 in action: cancellation in the subtraction can leave tiny negative numbers where the true answer is zero, so we clip them.

### k-means++ initialisation

Where the centroids start matters enormously. **k-means++** picks the first centroid at random, then each next one with probability proportional to its squared distance from the nearest centroid already chosen. That spreads the seeds out and sharply reduces the chance of a bad local optimum.

```python
closest = self._squared_distances(X, centroids[:i]).min(axis=1)
centroids[i] = X[rng.choice(n_samples, p=closest / closest.sum())]
```

An **empty cluster** (a centroid nobody chose) is re-seeded at the worst-fitted point, which is what scikit-learn does too.

### Why `n_init` is not optional

k-means converges to a **local** optimum that depends entirely on where it started. Run it once and you're gambling. The notebook runs 30 single-restart fits and finds a wide spread between the best and worst final inertia. So `fit` runs `n_init` independent restarts, each with its own generator spawned from one parent seed (Chapter 7), and keeps the best:

```python
parent = np.random.SeedSequence(self.random_state)
generators = [np.random.default_rng(s) for s in parent.spawn(self.n_init)]
```

### Testing it

On 1,500 points in four blobs, our inertia matches scikit-learn's `KMeans` to within a tiny relative gap. An elbow plot of inertia against `k` bends clearly at `k = 4`.

> [!NOTE]
> Comparing `labels_` element by element would be wrong: cluster 0 in our run
> may be cluster 2 in scikit-learn's. The **Adjusted Rand Index** measures
> whether the same points ended up *together*, which is the only thing that
> matters. Ours agrees with scikit-learn's partition almost perfectly.

> [!STORY] A clustering algorithm that waited 25 years to be published
> The procedure in this chapter is often called **Lloyd's algorithm**. Stuart
> Lloyd devised it at Bell Labs in 1957, as a way of quantising signals for
> pulse-code modulation, but it circulated only as an internal memo and
> wasn't formally published until 1982. Meanwhile James MacQueen coined the
> name "k-means" in 1967. The k-means++ seeding you just implemented is much
> younger: David Arthur and Sergei Vassilvitskii published it in 2007, and
> scikit-learn uses it by default.

### Performance: ours versus scikit-learn

On 50,000 points with 10 features and `k = 8`, our version lands within a fraction of a percent of scikit-learn's inertia, and is a few times slower. That's a **good** result for pure NumPy. The gap comes from Cython inner loops, Elkan's algorithm (which uses the triangle inequality to skip distance computations that can't change an assignment) and multithreading, not from anything wrong with our approach.

The lesson is Chapter 1's: **use the library in production, but know what it's doing.**

## Verification

The notebook ends with a checklist, every item of which must pass: our linear regression and Ridge match scikit-learn; gradient descent converges to the closed-form answer; predictions have the right shape; R² lies between 0 and 1; our k-means partition matches scikit-learn's (ARI above 0.99) with inertia within 1%; every point is assigned; no cluster is empty; and no squared distance is negative.

## Your turn: extensions

| # | Extension | Difficulty |
| :--- | :--- | :--- |
| 1 | **Promote** both classes into `src/dstoolkit/models.py`, with tests for the shape guards and the scikit-learn agreement. | ★ |
| 2 | **Weighted least squares**: `β = (XᵀWX)⁻¹XᵀWy`, *without* forming the `(n, n)` matrix `W` (80 GB for 100,000 samples). | ★★ |
| 3 | **Mini-batch k-means**: update centroids from random subsets; compare speed and final inertia on 500,000 points. | ★★ |
| 4 | **Silhouette score** in vectorised NumPy, used to choose `k` instead of the elbow. Watch the `(n, n)` memory. | ★★★ |
| 5 | **Polynomial features** with a `degree` parameter. Show the condition number exploding, and Ridge rescuing degree 10. | ★★ |
| 6 | **Elkan's algorithm**: skip impossible distance computations; measure how much of the gap to scikit-learn it closes. | ★★★ |

## What you built

Two production-shaped estimators that follow the scikit-learn contract and drop into a `Pipeline`; validate their inputs and fail with useful messages; use `lstsq` rather than `inv` and stay correct on rank-deficient data; compute distances by expansion rather than a difference tensor; restart from independent, seeded streams and keep the best; and agree with scikit-learn to within floating-point tolerance.

The next module is pandas. Everything you've learned about shapes, dtypes, views and vectorisation carries straight over: a DataFrame is a collection of labelled NumPy arrays with a great deal of convenience on top.

> [!KEY]
> - **`lstsq` is the default solver**: robust, and it survives collinearity. Normal equations via `solve`, never `inv`.
> - Ridge can be written as ordinary least squares on an **augmented** system.
> - Gradient descent reaches the same answer by walking downhill; it's how neural networks learn.
> - **Collinear features make coefficients meaningless** even when the fit looks perfect. Regularise before you interpret.
> - k-means is **assign, update, repeat**; expand ‖x − c‖² to avoid a giant tensor.
> - k-means finds a **local** optimum: use **k-means++** and **several restarts**.
> - Compare clusterings with the **Adjusted Rand Index**, never label by label.
> - Use the library in production, but know what it's doing.
