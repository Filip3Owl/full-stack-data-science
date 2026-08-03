# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3.12 (Data Science Course)
#     language: python
#     name: dsc-py312
# ---

# %% [markdown]
# # 06 · Linear Algebra
#
# **Module 02 — NumPy & Numerical Computing**
#
# ---
#
# ## What you will learn
#
# 1. Vectors: dot products, norms, and what "similarity" means geometrically.
# 2. Matrix multiplication — `@` versus `*`, and why the distinction matters.
# 3. Solving linear systems, and why **you should never invert a matrix**.
# 4. Least squares — deriving linear regression from scratch.
# 5. Eigenvalues and SVD, the machinery behind PCA.
# 6. Condition numbers: recognising a problem that is numerically hopeless.
#
# **Estimated time:** 90 minutes
#
# > This is the mathematical core of Modules 11–19. Every model you will fit is
# > linear algebra underneath.

# %%
import matplotlib.pyplot as plt
import numpy as np

from dstoolkit import setup_plot_style
from dstoolkit.plotting import COLORS

setup_plot_style()
rng = np.random.default_rng(seed=42)

# %% [markdown]
# ## 1. Vectors
#
# ### The dot product
#
# `a · b = Σ aᵢbᵢ`. Geometrically it is `|a| |b| cos θ` — it measures how much two
# vectors point in the same direction.

# %%
a = np.array([3.0, 4.0])
b = np.array([1.0, 2.0])

print(f"a · b (np.dot)  : {np.dot(a, b)}")
print(f"a @ b           : {a @ b}   ← preferred syntax")
print(f"(a * b).sum()   : {(a * b).sum()}   ← same thing, spelled out")

# %% [markdown]
# ### Norms — measuring length

# %%
v = np.array([3.0, 4.0])

print(f"L2 (Euclidean)  : {np.linalg.norm(v)}          ← the 3-4-5 triangle")
print(f"L1 (Manhattan)  : {np.linalg.norm(v, ord=1)}")
print(f"L∞ (max)        : {np.linalg.norm(v, ord=np.inf)}")

# Row-wise norms of a matrix — the everyday case.
X = rng.normal(size=(4, 3))
print(f"\nper-row L2 norms : {np.linalg.norm(X, axis=1).round(3)}")

# %% [markdown]
# > [!NOTE]
# > L1 and L2 are exactly the penalties in Lasso and Ridge regression
# > (Module 11). L1's corners are what drive coefficients to exactly zero,
# > producing feature selection; L2's smooth ball merely shrinks them.

# %% [markdown]
# ### Cosine similarity
#
# Normalise away magnitude and only direction remains. This is how text
# embeddings, recommender systems, and nearest-neighbour search compare items.

# %%
def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine of the angle between two vectors, in [-1, 1].

    Returns 0.0 when either vector is the zero vector, since the angle is
    undefined there and 0 is the least misleading answer.
    """
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    if denominator == 0:
        return 0.0
    return float(u @ v / denominator)


same = np.array([1.0, 1.0])
scaled = np.array([5.0, 5.0])       # same direction, 5× the length
orthogonal = np.array([-1.0, 1.0])
opposite = np.array([-1.0, -1.0])

for name, other in [("identical", same), ("5× scaled", scaled),
                    ("orthogonal", orthogonal), ("opposite", opposite)]:
    print(f"{name:<12} cos = {cosine_similarity(same, other):+.3f}")

print("\n↑ magnitude is irrelevant — only the angle counts.")

# %%
fig, ax = plt.subplots(figsize=(5.5, 5.5))

vectors = [
    (same, "reference", COLORS["primary"]),
    (scaled, "5× scaled — cos 1.00", COLORS["teal"]),
    (orthogonal, "orthogonal — cos 0.00", COLORS["accent"]),
    (opposite, "opposite — cos −1.00", COLORS["danger"]),
]

for vec, label, color in vectors:
    ax.annotate("", xy=vec, xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 2.2})
    ax.plot([], [], color=color, linewidth=2.2, label=label)

ax.set_xlim(-2.5, 5.5)
ax.set_ylim(-2.5, 5.5)
ax.axhline(0, color=COLORS["muted"], linewidth=0.8)
ax.axvline(0, color=COLORS["muted"], linewidth=0.8)
ax.set_aspect("equal")
ax.set_title("Cosine similarity ignores magnitude")
ax.legend(loc="upper left", fontsize=8.5)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Matrix multiplication
#
# ### `@` is not `*`
#
# This is the most common source of wrong answers for people arriving from MATLAB
# or from a maths background.

# %%
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A =\n{A}\nB =\n{B}")
print(f"\nA * B  (element-wise, NOT matrix product):\n{A * B}")
print(f"\nA @ B  (matrix product):\n{A @ B}")

# %% [markdown]
# ### The shape rule
#
# `(m, n) @ (n, p) → (m, p)`. The **inner dimensions must match**, and they
# vanish; the outer dimensions survive.

# %%
combinations = [((3, 4), (4, 2)), ((5, 3), (3, 3)), ((2, 3), (2, 3))]

for shape_a, shape_b in combinations:
    M, N = rng.random(shape_a), rng.random(shape_b)
    try:
        result = (M @ N).shape
        print(f"{str(shape_a)} @ {str(shape_b)} → {result}")
    except ValueError as exc:
        print(f"{str(shape_a)} @ {str(shape_b)} → ERROR: {str(exc)[:58]}…")

# %% [markdown]
# ### 1-D arrays are treated specially

# %%
M = np.array([[1, 2, 3], [4, 5, 6]])       # (2, 3)
v = np.array([1, 1, 1])                     # (3,)

print(f"M @ v shape : {(M @ v).shape}   ← v treated as a column, result is 1-D")
print(f"M @ v       : {M @ v}")

u = np.array([1, 1])                        # (2,)
print(f"\nu @ M shape : {(u @ M).shape}   ← u treated as a row")
print(f"u @ M       : {u @ M}")

# vector @ vector gives a scalar, not a matrix.
print(f"\nv @ v       : {v @ v}   ← a scalar (the dot product)")

# %% [markdown]
# > [!TIP]
# > This convenience is also a trap: because 1-D arrays adapt to whichever
# > orientation makes the multiplication work, a genuine shape mistake can go
# > unnoticed. When shapes matter, be explicit with `reshape(-1, 1)`.

# %% [markdown]
# ### Batched multiplication
#
# `@` operates on the **last two axes** and broadcasts everything before them —
# which is exactly how neural networks process a batch of samples at once.

# %%
batch = rng.random((32, 4, 3))       # 32 matrices of shape (4, 3)
weights = rng.random((3, 5))          # one shared weight matrix

output = batch @ weights
print(f"batch   {batch.shape}")
print(f"weights {weights.shape}")
print(f"output  {output.shape}   ← 32 independent (4,3)@(3,5) products")

# %% [markdown]
# ## 3. Solving linear systems
#
# ### Never invert a matrix
#
# To solve `Ax = b`, the algebra says `x = A⁻¹b`. **Do not compute it that way.**
# `np.linalg.solve` uses LU decomposition: it is faster and markedly more
# accurate.

# %%
A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])

x_solve = np.linalg.solve(A, b)
x_inverse = np.linalg.inv(A) @ b

print(f"solve()     : {x_solve}")
print(f"inv() @ b   : {x_inverse}")
print(f"\nverify A @ x == b : {np.allclose(A @ x_solve, b)}")

# %%
# On an ill-conditioned system the difference stops being academic.
n = 12
# A Hilbert matrix is the textbook example of near-singularity.
hilbert = 1.0 / (np.arange(1, n + 1)[:, None] + np.arange(n) )
x_true = np.ones(n)
b = hilbert @ x_true

x_solve = np.linalg.solve(hilbert, b)
x_inverse = np.linalg.inv(hilbert) @ b

print(f"condition number : {np.linalg.cond(hilbert):.3e}   ← catastrophically high")
print(f"\nerror with solve()   : {np.abs(x_solve - x_true).max():.3e}")
print(f"error with inv() @ b : {np.abs(x_inverse - x_true).max():.3e}")
print(f"\n→ inv() is {np.abs(x_inverse - x_true).max() / np.abs(x_solve - x_true).max():.1f}× worse")

# %% [markdown]
# ### The condition number
#
# `cond(A)` estimates how much a small change in the input can be amplified in the
# output. As a rule of thumb, you lose about `log10(cond)` significant digits.

# %%
matrices = {
    "identity": np.eye(4),
    "well conditioned": np.array([[4.0, 1.0], [1.0, 3.0]]),
    "nearly singular": np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]]),
    "Hilbert (12×12)": hilbert,
}

print(f"{'matrix':<20} {'cond':>12} {'digits lost':>12}")
print("-" * 46)
for name, M in matrices.items():
    condition = np.linalg.cond(M)
    print(f"{name:<20} {condition:>12.3e} {np.log10(condition):>12.1f}")

print(f"\nfloat64 has ~15 digits — the Hilbert matrix consumes nearly all of them.")

# %% [markdown]
# > [!WARNING]
# > A high condition number is why you **standardise features** before fitting a
# > linear model, and why perfectly correlated columns break a regression. It is
# > not a numerical curiosity — it is the reason `Ridge` adds `alpha` to the
# > diagonal (Module 11).

# %% [markdown]
# ## 4. Least squares — linear regression from first principles
#
# For an overdetermined system (more rows than unknowns) there is usually no exact
# solution. Least squares finds the `x` minimising `‖Ax − b‖²`.
#
# The **normal equations** give the closed form:
#
# $$\hat{\beta} = (X^T X)^{-1} X^T y$$

# %%
# Simulate data with a known answer, so we can check our work.
n_samples = 200
true_coefficients = np.array([2.5, -1.3, 0.8])
true_intercept = 5.0

X = rng.normal(size=(n_samples, 3))
noise = rng.normal(0, 0.5, n_samples)
y = X @ true_coefficients + true_intercept + noise

# Add a column of ones so the intercept is just another coefficient.
X_design = np.column_stack([np.ones(n_samples), X])
print(f"design matrix shape : {X_design.shape}   ← first column is all 1s")

# %%
# --- Method 1: the normal equations, literally ------------------------------
beta_normal = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y

# --- Method 2: the same equations, solved rather than inverted --------------
beta_solve = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)

# --- Method 3: lstsq — what you should actually use -------------------------
beta_lstsq, residuals, rank, singular_values = np.linalg.lstsq(
    X_design, y, rcond=None
)

truth = np.concatenate([[true_intercept], true_coefficients])

print(f"{'':<14} {'intercept':>10} {'β₁':>8} {'β₂':>8} {'β₃':>8}")
print("-" * 50)
print(f"{'true':<14} {truth[0]:>10.4f} {truth[1]:>8.4f} "
      f"{truth[2]:>8.4f} {truth[3]:>8.4f}")
for name, beta in [("normal (inv)", beta_normal), ("solve", beta_solve),
                   ("lstsq", beta_lstsq)]:
    print(f"{name:<14} {beta[0]:>10.4f} {beta[1]:>8.4f} "
          f"{beta[2]:>8.4f} {beta[3]:>8.4f}")

# %% [markdown]
# All three agree here because the problem is well conditioned. `lstsq` is still
# the right choice: it uses SVD, so it handles rank-deficient matrices — the case
# where two features are perfectly collinear — instead of raising or returning
# garbage.

# %%
# Perfect collinearity: feature 3 is an exact copy of feature 1.
X_collinear = np.column_stack([np.ones(n_samples), X[:, 0], X[:, 1], X[:, 0]])

try:
    np.linalg.solve(X_collinear.T @ X_collinear, X_collinear.T @ y)
except np.linalg.LinAlgError as exc:
    print(f"solve()  → LinAlgError: {exc}")

beta_safe, _, rank, _ = np.linalg.lstsq(X_collinear, y, rcond=None)
print(f"lstsq()  → works. rank {rank} of {X_collinear.shape[1]} columns")
print(f"           coefficients {beta_safe.round(3)}")
print("\n↑ lstsq returns the minimum-norm solution, splitting the shared effect.")

# %%
# Cross-check against scikit-learn — the same computation it performs.
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, y)

print(f"our intercept    : {beta_lstsq[0]:.6f}")
print(f"sklearn intercept: {model.intercept_:.6f}")
print(f"our coefficients : {beta_lstsq[1:].round(6)}")
print(f"sklearn coefs    : {model.coef_.round(6)}")
print(f"\nidentical : {np.allclose(beta_lstsq[1:], model.coef_)}")

# %% [markdown]
# ## 5. Eigenvalues and eigenvectors
#
# An eigenvector of `A` is a direction that `A` only *scales*, never rotates:
# `Av = λv`. The eigenvalue `λ` is that scale factor.

# %%
A = np.array([[4.0, 1.0], [2.0, 3.0]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"A =\n{A}")
print(f"\neigenvalues  : {eigenvalues.round(4)}")
print(f"eigenvectors (columns):\n{eigenvectors.round(4)}")

# Verify the definition for the first eigenpair.
lam, v = eigenvalues[0], eigenvectors[:, 0]
print(f"\nA @ v   : {(A @ v).round(6)}")
print(f"λ * v   : {(lam * v).round(6)}")
print(f"equal   : {np.allclose(A @ v, lam * v)}")

# %% [markdown]
# ### Why this matters: PCA
#
# Principal Component Analysis is the eigendecomposition of the covariance
# matrix. The eigenvectors are the principal directions; the eigenvalues are the
# variance explained along each.

# %%
# Correlated 2-D data, so there is an obvious principal direction.
mean = [0, 0]
covariance = [[3.0, 2.4], [2.4, 2.5]]
points = rng.multivariate_normal(mean, covariance, size=500)

centred = points - points.mean(axis=0)
cov_matrix = (centred.T @ centred) / (len(centred) - 1)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)   # eigh: symmetric input
# eigh returns ascending order; reverse so component 1 is the largest.
order = eigenvalues.argsort()[::-1]
eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

print(f"covariance matrix:\n{cov_matrix.round(3)}")
print(f"\neigenvalues (variance per component) : {eigenvalues.round(3)}")
print(f"variance explained : "
      f"{(eigenvalues / eigenvalues.sum()).round(3)}")
print(f"→ the first component captures "
      f"{eigenvalues[0] / eigenvalues.sum():.1%} of the variance")

# %%
fig, (ax_data, ax_proj) = plt.subplots(1, 2, figsize=(11.5, 4.8))

# --- Left: the data with its principal axes -------------------------------
ax_data.scatter(centred[:, 0], centred[:, 1], s=12, alpha=0.4,
                color=COLORS["primary"], edgecolors="none")

for i, (value, color) in enumerate(
    zip(eigenvalues, [COLORS["danger"], COLORS["accent"]], strict=True)
):
    # Scale each axis by its standard deviation so the arrow length is meaningful.
    direction = eigenvectors[:, i] * np.sqrt(value) * 2.5
    ax_data.annotate("", xy=direction, xytext=(0, 0),
                     arrowprops={"arrowstyle": "-|>", "color": color,
                                 "linewidth": 2.6})
    ax_data.plot([], [], color=color, linewidth=2.6,
                 label=f"PC{i + 1} — {eigenvalues[i] / eigenvalues.sum():.0%} of variance")

ax_data.set_title("Principal components are the eigenvectors")
ax_data.set_xlabel("feature 1")
ax_data.set_ylabel("feature 2")
ax_data.set_aspect("equal")
ax_data.legend(fontsize=8.5)

# --- Right: the data rotated onto its principal axes ----------------------
projected = centred @ eigenvectors

ax_proj.scatter(projected[:, 0], projected[:, 1], s=12, alpha=0.4,
                color=COLORS["teal"], edgecolors="none")
ax_proj.set_title("The same data, expressed in PC coordinates")
ax_proj.set_xlabel("PC1")
ax_proj.set_ylabel("PC2")
ax_proj.set_aspect("equal")

fig.tight_layout()
plt.show()

# %%
# The projected components are uncorrelated — that is the point of PCA.
projected_cov = np.cov(projected.T)
print(f"covariance of the projected data:\n{projected_cov.round(6)}")
print(f"\noff-diagonal ≈ 0 : {np.isclose(projected_cov[0, 1], 0, atol=1e-10)}")

# Cross-check with scikit-learn.
from sklearn.decomposition import PCA

sk_pca = PCA(n_components=2).fit(centred)
print(f"\nour variance ratios     : {(eigenvalues / eigenvalues.sum()).round(6)}")
print(f"sklearn variance ratios : {sk_pca.explained_variance_ratio_.round(6)}")

# %% [markdown]
# ## 6. Singular Value Decomposition
#
# SVD factors **any** matrix — square or not, invertible or not — as
# `A = U Σ Vᵀ`. It underpins PCA, recommender systems, image compression, and
# `lstsq` itself.

# %%
A = rng.random((6, 4))
U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)

print(f"A  shape : {A.shape}")
print(f"U  shape : {U.shape}")
print(f"Σ        : {singular_values.round(4)}")
print(f"Vᵀ shape : {Vt.shape}")

reconstructed = U @ np.diag(singular_values) @ Vt
print(f"\nreconstruction exact : {np.allclose(A, reconstructed)}")

# %% [markdown]
# ### Low-rank approximation
#
# Keeping only the largest `k` singular values gives the best possible rank-`k`
# approximation. This is compression, and it is also denoising.

# %%
# A matrix that is genuinely low-rank plus a little noise.
true_rank = 3
left = rng.random((60, true_rank))
right = rng.random((true_rank, 40))
M = left @ right + rng.normal(0, 0.01, (60, 40))

U, s, Vt = np.linalg.svd(M, full_matrices=False)

print(f"first 8 singular values : {s[:8].round(3)}")
print(f"→ a sharp drop after {true_rank}, which is the true rank\n")

for k in [1, 2, 3, 5, 10]:
    approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
    error = np.linalg.norm(M - approx) / np.linalg.norm(M)
    stored = k * (M.shape[0] + M.shape[1] + 1)
    print(f"rank {k:>2} : relative error {error:6.2%}   "
          f"stores {stored:>5,} of {M.size:,} values "
          f"({stored / M.size:.0%})")

# %% [markdown]
# ## 7. Quick reference

# %%
reference = [
    ("a @ b", "dot product / matrix product"),
    ("np.linalg.norm(x, axis=1)", "row-wise vector lengths"),
    ("np.linalg.solve(A, b)", "solve Ax = b — never use inv()"),
    ("np.linalg.lstsq(A, b)", "least squares; handles rank deficiency"),
    ("np.linalg.inv(A)", "inverse — almost always the wrong tool"),
    ("np.linalg.det(A)", "determinant"),
    ("np.linalg.matrix_rank(A)", "numerical rank"),
    ("np.linalg.cond(A)", "condition number — check before trusting a fit"),
    ("np.linalg.eig(A)", "eigendecomposition, general matrix"),
    ("np.linalg.eigh(A)", "eigendecomposition, symmetric — faster, more stable"),
    ("np.linalg.svd(A)", "singular value decomposition"),
    ("np.trace(A)", "sum of the diagonal"),
    ("A.T", "transpose (a view)"),
]

print(f"{'call':<28} {'purpose'}")
print("-" * 74)
for call, purpose in reference:
    print(f"{call:<28} {purpose}")

# %% [markdown]
# ## 8. Exercises
#
# ### Exercise 6.1 — Shapes
#
# Give the result shape or "error":
#
# ```python
# (3, 4) @ (4, 5)      (3, 4) @ (3, 4)      (4,) @ (4,)
# (3, 4) @ (4,)        (10, 3, 4) @ (4, 2)  (3,) @ (3, 4)
# ```
#
# ### Exercise 6.2 — Cosine similarity matrix
#
# Given `E` of shape `(1000, 128)` (1000 documents, 128-dimensional embeddings),
# compute the full `(1000, 1000)` cosine similarity matrix **without a loop**.
# *Hint:* normalise the rows first, then one matrix product does the rest.
#
# ### Exercise 6.3 — Implement least squares
#
# Write `fit_linear(X, y, fit_intercept=True)` returning coefficients via
# `np.linalg.lstsq`. Validate against `sklearn.linear_model.LinearRegression` on
# random data. Then add an `alpha` parameter implementing Ridge:
# `β = (XᵀX + αI)⁻¹Xᵀy`, and confirm it matches `sklearn.linear_model.Ridge`.
#
# ### Exercise 6.4 — Condition number and collinearity
#
# Build `X` with two nearly identical columns (correlation 0.9999). Report
# `cond(XᵀX)`. Fit by normal equations and by `lstsq`, and compare the
# coefficients. Then standardise the columns and report the condition number
# again.
#
# ### Exercise 6.5 — Image compression with SVD
#
# Build a 200×200 grayscale "image" (any structured pattern). Reconstruct it at
# ranks 5, 20, and 50, and plot the four versions side by side with the storage
# each requires.

# %% [markdown]
# ## Key takeaways
#
# - **`@` is matrix multiplication; `*` is element-wise.** Confusing them is the
#   most common linear-algebra bug in Python.
# - `(m, n) @ (n, p) → (m, p)` — inner dimensions must match and disappear.
# - **Never use `np.linalg.inv` to solve a system.** `solve` is faster and much
#   more accurate; `lstsq` also survives rank deficiency.
# - Least squares *is* linear regression. `lstsq` is what scikit-learn calls.
# - **`cond(A)`** tells you how many digits a problem will cost you. High
#   condition numbers are why you standardise features and why Ridge exists.
# - **PCA is the eigendecomposition of the covariance matrix.** Use `eigh` for
#   symmetric matrices.
# - **SVD works on any matrix** and gives the best low-rank approximation —
#   compression, denoising, and recommenders all rest on it.
#
# ---
#
# **Next:** [07 · Randomness & Simulation](07_random_and_simulation.ipynb) —
# reproducible random numbers, and answering hard questions by simulating them.
