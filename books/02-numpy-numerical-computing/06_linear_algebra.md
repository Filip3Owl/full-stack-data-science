# Linear Algebra Without Tears
@subtitle Dot products, solving systems, least squares, eigenvectors and SVD
@time About 90 minutes
@notebook 06_linear_algebra.ipynb
@learn
- Vectors: dot products, norms, and what "similarity" means geometrically.
- Matrix multiplication: `@` versus `*`, and why the difference matters.
- Solving linear systems, and why **you should never invert a matrix**.
- Least squares: deriving linear regression from scratch.
- Eigenvalues and SVD, the machinery behind PCA.
- Condition numbers: recognising a problem that is numerically hopeless.
@end

@lead Linear algebra has a fearsome reputation, mostly earned in lecture halls full of proofs. For a data scientist it's something friendlier: a small set of operations on tables of numbers, each with a clear geometric meaning, all of which NumPy performs for you. Every model in Modules 11 to 19 is linear algebra underneath, so this is the chapter that pays off longest.

## Vectors

### The dot product

`a · b = Σ aᵢbᵢ`. Geometrically it equals `|a| |b| cos θ`: it measures how much two vectors point the same way.

```python
a = np.array([3.0, 4.0]); b = np.array([1.0, 2.0])
a @ b              # 11.0, the preferred syntax
np.dot(a, b)       # the same
(a * b).sum()      # the same, spelled out
```

### Norms: measuring length

```python
v = np.array([3.0, 4.0])
np.linalg.norm(v)              # 5.0, Euclidean (L2): the 3-4-5 triangle
np.linalg.norm(v, ord=1)       # 7.0, Manhattan (L1)
np.linalg.norm(v, ord=np.inf)  # 4.0, the largest component
np.linalg.norm(X, axis=1)      # one length per row: the everyday case
```

> [!NOTE]
> L1 and L2 are exactly the penalties in Lasso and Ridge regression
> (Module 11). L1's sharp corners are what push coefficients to exactly zero,
> which gives you feature selection; L2's smooth ball merely shrinks them.

### Cosine similarity

Divide out the lengths and only the *direction* remains. This is how text embeddings, recommender systems and nearest-neighbour search compare items:

```python
def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    if denominator == 0:
        return 0.0               # the angle is undefined; 0 misleads least
    return float(u @ v / denominator)
```

![Cosine similarity ignores length: a vector five times longer in the same direction scores 1, a perpendicular one 0, an opposite one −1.](assets/images/book/m02_cosine.png){width=50}

## Matrix multiplication

### `@` is not `*`

This is the most common source of wrong answers for people arriving from MATLAB or from a maths background:

```python
A = np.array([[1, 2], [3, 4]]); B = np.array([[5, 6], [7, 8]])
A * B     # [[ 5 12] [21 32]]   element-wise
A @ B     # [[19 22] [43 50]]   the matrix product
```

### The shape rule

`(m, n) @ (n, p) → (m, p)`. The **inner dimensions must match**, and they vanish; the outer ones survive. `(3, 4) @ (4, 2)` gives `(3, 2)`; `(2, 3) @ (2, 3)` is an error.

### 1-D arrays adapt, which is convenient and risky

A 1-D array is treated as a column when it's on the right and as a row when it's on the left, and `v @ v` gives a plain scalar:

```python
M = np.array([[1, 2, 3], [4, 5, 6]])     # (2, 3)
M @ np.array([1, 1, 1])                  # [ 6 15]  shape (2,)
np.array([1, 1]) @ M                     # [5 7 9]  shape (3,)
```

> [!TIP]
> Because 1-D arrays bend to whichever orientation makes the product work, a
> genuine shape mistake can slip through. When shapes matter, be explicit
> with `reshape(-1, 1)`.

### Batched multiplication

`@` works on the **last two axes** and broadcasts everything before them, which is exactly how a neural network processes a batch: a `(32, 4, 3)` batch `@` a `(3, 5)` weight matrix gives `(32, 4, 5)`, thirty-two independent products in one call.

## Solving linear systems

### Never invert a matrix

To solve `Ax = b`, the algebra says `x = A⁻¹b`. **Don't compute it that way.** `np.linalg.solve` uses an LU decomposition: faster, and markedly more accurate.

```python
A = np.array([[3.0, 1.0], [1.0, 2.0]]); b = np.array([9.0, 8.0])
x = np.linalg.solve(A, b)       # [2. 3.]
np.allclose(A @ x, b)           # True
```

On a nice small system both methods agree. On an awkward one they don't. The notebook builds a 12 × 12 **Hilbert matrix** (entries 1/(i+j+1)), a textbook example of a nearly singular matrix, and solves a system whose true answer is all ones. `solve` gets close; `inv(A) @ b` is visibly worse.

### The condition number

`np.linalg.cond(A)` estimates how much a tiny change in the input can be amplified in the output. A rule of thumb worth remembering: **you lose about `log10(cond)` significant digits.**

![Digits lost to the condition number, for four matrices. float64 only has about sixteen to spend; the 12 × 12 Hilbert matrix spends all of them.](assets/images/book/m02_condition.png){width=100}

> [!WARNING]
> A high condition number is why you **standardise features** before fitting
> a linear model, and why perfectly correlated columns break a regression.
> It isn't a numerical curiosity: it's the reason Ridge regression adds
> `alpha` to the diagonal (Module 11).

> [!NOTE]
> The term "condition number" was introduced by **Alan Turing** in a 1948
> paper on rounding errors in matrix computations, written while he was
> thinking about how the first electronic computers would cope with real
> arithmetic. The codebreaker was also one of the founders of numerical
> analysis.

## Least squares: linear regression from first principles

With more rows than unknowns there's usually no exact solution. **Least squares** finds the coefficients that minimise the sum of squared residuals, `‖Xβ − y‖²`.

![Least squares picks the line that makes the red segments, the residuals, as small as possible in total squared length.](assets/images/book/m02_least_squares.png){width=50}

The **normal equations** give a closed form, β̂ = (XᵀX)⁻¹Xᵀy. Add a column of ones to `X` and the intercept becomes just another coefficient. The notebook simulates data with known coefficients (intercept 5, slopes 2.5, −1.3 and 0.8) and recovers them three ways:

```python
X_design = np.column_stack([np.ones(n_samples), X])

beta_inv = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y      # literally
beta_solve = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)   # better
beta_lstsq, *_ = np.linalg.lstsq(X_design, y, rcond=None)             # use this
```

All three agree here, because the problem is well conditioned. `lstsq` is still the one to use: it works through the SVD, so it survives **rank deficiency**, two perfectly collinear features, where `solve` raises a `LinAlgError` ("Singular matrix") and `lstsq` returns the minimum-norm solution, splitting the shared effect between the twins. The notebook confirms our coefficients match scikit-learn's `LinearRegression` to many decimal places, because it's the same computation.

> [!STORY] Finding a lost planet with least squares
> On New Year's Day 1801 the astronomer Giuseppe Piazzi spotted a faint new
> object, later named Ceres. He tracked it for about six weeks before it
> vanished into the glare of the Sun, and astronomers had too few
> observations to work out where it would reappear. A 24-year-old Carl
> Friedrich Gauss took the scattered measurements, fitted an orbit with a
> method of his own, and predicted where to look. On the last night of 1801,
> Ceres was found almost exactly where he said. Gauss's method was **least
> squares**. Adrien-Marie Legendre published it first, in 1805; Gauss said
> he'd been using it since 1795, and the two argued about priority for years.
> Both would recognise `np.linalg.lstsq` at a glance.

## Eigenvalues and eigenvectors

An eigenvector of `A` is a direction that `A` only *stretches*, never rotates: `Av = λv`. The eigenvalue `λ` is the stretch factor.

```python
A = np.array([[4.0, 1.0], [2.0, 3.0]])
eigenvalues, eigenvectors = np.linalg.eig(A)     # vectors are the COLUMNS
lam, v = eigenvalues[0], eigenvectors[:, 0]
np.allclose(A @ v, lam * v)                       # True: the definition
```

### Why it matters: PCA

**Principal Component Analysis** is the eigendecomposition of the covariance matrix. The eigenvectors are the principal directions; the eigenvalues are the variance along each one.

```python
centred = points - points.mean(axis=0)
cov = (centred.T @ centred) / (len(centred) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)    # eigh: for symmetric matrices
order = eigenvalues.argsort()[::-1]                # eigh returns ascending order
projected = centred @ eigenvectors[:, order]
```

![Left: correlated data with its two principal axes, scaled by their spread. Right: the same points expressed in principal-component coordinates, where they are uncorrelated.](assets/images/book/m02_pca.png){width=100}

In the rotated coordinates the covariance matrix is diagonal: the components are uncorrelated, which is the whole point of PCA. The notebook checks the explained-variance ratios against scikit-learn's `PCA`, and they match.

> [!NOTE]
> PCA is older than computers. **Karl Pearson** described it in 1901 as
> finding "lines and planes of closest fit" to a cloud of points, and
> **Harold Hotelling** developed and named it in 1933. They computed by
> hand what `eigh` does in microseconds.

## Singular Value Decomposition

The SVD factors **any** matrix (square or not, invertible or not) as `A = U Σ Vᵀ`. It underpins PCA, recommender systems, image compression and `lstsq` itself.

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
np.allclose(A, U @ np.diag(s) @ Vt)     # True: an exact reconstruction
```

### Low-rank approximation

Keeping only the `k` largest singular values gives the **best possible** rank-`k` approximation of the matrix. That's compression, and it's also denoising. The notebook builds a 60 × 40 matrix that is secretly rank 3, plus a little noise:

![The singular values drop sharply after the third, the true rank, and keeping just three components already reconstructs the matrix almost perfectly.](assets/images/book/m02_svd.png){width=100}

Storing the rank-3 approximation takes `3 × (60 + 40 + 1)` numbers, about an eighth of the original 2,400.

> [!STORY] A million dollars for a better recommendation
> In October 2006 Netflix offered **$1,000,000** to anyone who could improve
> its film-recommendation accuracy by 10%. Thousands of teams competed. One of
> the most influential early ideas came from a programmer writing as "Simon
> Funk", who posted on his blog how to factorise the gigantic, mostly empty
> users × films ratings matrix into a handful of hidden "taste" factors, an
> SVD-style low-rank approximation like the one above. The prize was finally
> won in September 2009 by a merged team called *BellKor's Pragmatic Chaos*,
> whose winning blend still leaned heavily on matrix factorisation.

## Quick reference

| Call | Purpose |
| :--- | :--- |
| `a @ b` | dot product / matrix product |
| `np.linalg.norm(x, axis=1)` | row-wise vector lengths |
| `np.linalg.solve(A, b)` | solve `Ax = b`, never via `inv()` |
| `np.linalg.lstsq(A, b)` | least squares; survives rank deficiency |
| `np.linalg.inv(A)` | inverse: almost always the wrong tool |
| `np.linalg.det(A)`, `matrix_rank(A)` | determinant, numerical rank |
| `np.linalg.cond(A)` | condition number: check it before trusting a fit |
| `np.linalg.eig(A)` / `eigh(A)` | eigendecomposition; `eigh` for symmetric matrices |
| `np.linalg.svd(A)` | singular value decomposition |
| `np.trace(A)`, `A.T` | sum of the diagonal; transpose (a view) |

## Try it yourself

Exercises 6.1 to 6.5 in notebook 06; solutions in notebook 10.

1. **Shapes.** Result shape or "error": `(3,4)@(4,5)`, `(3,4)@(3,4)`, `(4,)@(4,)`, `(3,4)@(4,)`, `(10,3,4)@(4,2)`, `(3,)@(3,4)`.
2. **Cosine similarity matrix.** For `E` of shape `(1000, 128)`, compute the full `(1000, 1000)` cosine-similarity matrix without a loop. Hint: normalise the rows, then one product does the rest.
3. **Implement least squares.** `fit_linear(X, y, fit_intercept=True)` via `lstsq`, validated against scikit-learn. Then add `alpha` for Ridge, `β = (XᵀX + αI)⁻¹Xᵀy`, and match `sklearn.linear_model.Ridge`.
4. **Condition number and collinearity.** Build two columns with correlation 0.9999, report `cond(XᵀX)`, compare normal-equation and `lstsq` coefficients, then standardise and report the condition number again.
5. **Image compression with SVD.** Reconstruct a 200 × 200 structured "image" at ranks 5, 20 and 50 and plot all four versions with the storage each needs.

> [!KEY]
> - **`@` is matrix multiplication**; `*` is element-wise.
> - `(m, n) @ (n, p) → (m, p)`: inner dimensions match and disappear.
> - **Never use `inv` to solve a system.** `solve` is faster and more accurate; `lstsq` also survives rank deficiency.
> - Least squares *is* linear regression, and `lstsq` is what scikit-learn uses.
> - **`cond(A)`** tells you how many digits a problem will cost: that's why you standardise, and why Ridge exists.
> - **PCA is the eigendecomposition of the covariance matrix.** Use `eigh` for symmetric matrices.
> - **SVD works on any matrix** and gives the best low-rank approximation: compression, denoising, recommenders.
