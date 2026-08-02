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
# # 05 · Object-Oriented Programming
#
# **Module 01 — Python Foundations for Data Science**
#
# ---
#
# ## What you will learn
#
# 1. When a class is the right tool — and when a function is better.
# 2. Classes, instances, and the difference between class and instance attributes.
# 3. Dunder methods: making your objects work with `print`, `len`, `==`, and `[]`.
# 4. `@dataclass` — most of the boilerplate, deleted.
# 5. `@property` for validated, computed attributes.
# 6. Inheritance vs. composition, and why the industry prefers composition.
# 7. **The scikit-learn estimator pattern** — the interface you will implement for real.
#
# **Estimated time:** 75 minutes

# %%
import numpy as np

from dstoolkit import setup_plot_style

setup_plot_style()

# %% [markdown]
# ## 1. When should a data scientist write a class?
#
# Honest answer: **less often than tutorials suggest.** Most analysis code should be
# functions operating on DataFrames. Reach for a class when you have **state and
# behaviour that belong together**:
#
# | Situation | Use |
# | :--- | :--- |
# | Transform data once, return a result | a **function** |
# | Group related constants | a **module** or `Enum` |
# | Carry a fixed set of fields around | a **`@dataclass`** |
# | Something *learns* parameters, then *applies* them | a **class** |
# | Several interchangeable implementations behind one interface | a **class hierarchy** or `Protocol` |
#
# The fourth row is the big one in data science: a scaler learns a mean and a
# standard deviation from training data, then applies them to new data. That
# fitted state is exactly what an object is for.

# %% [markdown]
# ## 2. Classes and instances

# %%
class Dataset:
    """A named collection of numeric observations.

    Attributes
    ----------
    name : str
        Human-readable identifier.
    values : list of float
        The observations.
    """

    # A CLASS attribute — shared by every instance. Use these for constants.
    version = "1.0"

    def __init__(self, name: str, values: list[float]) -> None:
        """Initialise a dataset.

        `__init__` is not a constructor — the object already exists by the time
        it runs. Its job is to set up INSTANCE attributes, which are unique to
        each object.
        """
        self.name = name          # instance attribute
        self.values = list(values)  # copy, so the caller's list stays independent

    def mean(self) -> float:
        """Return the arithmetic mean of the observations."""
        # `self` is the instance the method was called on. It is passed
        # automatically: `d.mean()` is really `Dataset.mean(d)`.
        return sum(self.values) / len(self.values)

    def summary(self) -> dict[str, float]:
        """Return basic descriptive statistics."""
        arr = np.asarray(self.values)
        return {
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }


sales = Dataset("q1_sales", [120.0, 340.5, 89.0, 560.25, 210.0])

print(f"name     : {sales.name}")
print(f"mean     : {sales.mean():.2f}")
print(f"version  : {sales.version}   ← from the class, not the instance")
print(f"summary  : { {k: round(v, 2) for k, v in sales.summary().items()} }")

# %% [markdown]
# ### Class vs. instance attributes — a trap
#
# Class attributes are shared. If a **mutable** one is modified through any
# instance, every instance sees the change. This is the object-model lesson from
# notebook 02 appearing in a new costume.

# %%
class BrokenLogger:
    entries = []        # BUG: one list, shared by every instance ever created

    def log(self, message: str) -> None:
        self.entries.append(message)


a, b = BrokenLogger(), BrokenLogger()
a.log("from a")
b.log("from b")

print(f"a.entries : {a.entries}")
print(f"b.entries : {b.entries}   ← the same list!")
print(f"same object : {a.entries is b.entries}")


class FixedLogger:
    def __init__(self) -> None:
        self.entries: list[str] = []     # a fresh list per instance

    def log(self, message: str) -> None:
        self.entries.append(message)


c, d = FixedLogger(), FixedLogger()
c.log("from c")
d.log("from d")
print(f"\nc.entries : {c.entries}")
print(f"d.entries : {d.entries}   ← independent")

# %% [markdown]
# ## 3. Dunder methods
#
# "Dunder" (double underscore) methods let your objects participate in Python's
# built-in syntax. Implementing a handful makes a class feel native.

# %%
class Measurement:
    """A value with a unit, demonstrating the most useful dunder methods."""

    def __init__(self, value: float, unit: str) -> None:
        self.value = value
        self.unit = unit

    def __repr__(self) -> str:
        """Unambiguous representation, aimed at developers.

        Convention: it should look like the code that would recreate the object.
        This is what you see in a debugger and inside containers.
        """
        return f"Measurement(value={self.value!r}, unit={self.unit!r})"

    def __str__(self) -> str:
        """Readable representation, aimed at end users. Used by print()."""
        return f"{self.value:g} {self.unit}"

    def __eq__(self, other: object) -> bool:
        """Value equality. Without this, == falls back to identity."""
        if not isinstance(other, Measurement):
            return NotImplemented        # let Python try the reflected operation
        return (self.value, self.unit) == (other.value, other.unit)

    def __hash__(self) -> int:
        """Defining __eq__ removes the default __hash__ — restore it explicitly,
        otherwise the object cannot go in a set or be used as a dict key."""
        return hash((self.value, self.unit))

    def __lt__(self, other: "Measurement") -> bool:
        """Enables <, and therefore sorted()."""
        if self.unit != other.unit:
            raise ValueError(f"cannot compare {self.unit} with {other.unit}")
        return self.value < other.value

    def __add__(self, other: "Measurement") -> "Measurement":
        """Enables the + operator."""
        if self.unit != other.unit:
            raise ValueError(f"cannot add {self.unit} to {other.unit}")
        return Measurement(self.value + other.value, self.unit)


m1 = Measurement(23.5, "°C")
m2 = Measurement(1.5, "°C")

print(f"str  (print)  : {m1}")
print(f"repr (debug)  : {m1!r}")
print(f"in a list     : {[m1, m2]}      ← containers use __repr__")
print(f"m1 + m2       : {m1 + m2}")
print(f"m1 == copy    : {m1 == Measurement(23.5, '°C')}")
print(f"sorted        : {sorted([m1, m2])}")
print(f"in a set      : {len({m1, m2, Measurement(23.5, '°C')})} unique")

# %% [markdown]
# ### Container dunders
#
# Implement `__len__`, `__getitem__`, and `__iter__` and your class works with
# `len()`, `[]`, `for`, `in`, and unpacking — for free.

# %%
class TimeSeries:
    """A sequence of readings that behaves like a built-in container."""

    def __init__(self, readings: list[float]) -> None:
        self._readings = list(readings)

    def __len__(self) -> int:
        return len(self._readings)

    def __getitem__(self, index):
        # Supporting slices as well as integers costs one line and makes the
        # class behave the way users expect a sequence to behave.
        if isinstance(index, slice):
            return TimeSeries(self._readings[index])
        return self._readings[index]

    def __iter__(self):
        return iter(self._readings)

    def __repr__(self) -> str:
        return f"TimeSeries({self._readings})"


ts = TimeSeries([10.0, 12.5, 9.8, 15.2, 11.1])

print(f"len(ts)     : {len(ts)}")
print(f"ts[0]       : {ts[0]}")
print(f"ts[1:3]     : {ts[1:3]}")
print(f"max(ts)     : {max(ts)}         ← works because __iter__ exists")
print(f"12.5 in ts  : {12.5 in ts}")
print(f"sum(ts)     : {sum(ts)}")

# %% [markdown]
# ## 4. `@dataclass` — delete the boilerplate
#
# Most classes in data work just hold fields. `@dataclass` generates `__init__`,
# `__repr__`, and `__eq__` from type-annotated attributes.

# %%
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Configuration for one model training run."""

    name: str
    learning_rate: float = 0.01
    max_epochs: int = 100
    # Mutable defaults need `field(default_factory=...)` — the dataclass
    # machinery enforces this, catching the notebook-04 trap at class-definition
    # time rather than letting it bite you at runtime.
    features: list[str] = field(default_factory=list)
    seed: int = 42


config = ExperimentConfig(name="baseline", learning_rate=0.001)

print(f"repr for free : {config}")
print(f"eq for free   : {config == ExperimentConfig('baseline', 0.001)}")
print(f"defaults      : epochs={config.max_epochs}, seed={config.seed}")

# Proof that the mutable-default trap is handled:
c1, c2 = ExperimentConfig("a"), ExperimentConfig("b")
c1.features.append("age")
print(f"\nc1.features   : {c1.features}")
print(f"c2.features   : {c2.features}   ← independent, as it must be")

# %% [markdown]
# ### Frozen dataclasses — immutable configuration
#
# `frozen=True` makes instances immutable and hashable. This is the right choice
# for configuration objects: it guarantees no code path quietly rewrites a
# hyperparameter halfway through a run.

# %%
@dataclass(frozen=True, slots=True)
class ModelParams:
    """Immutable, memory-efficient hyperparameters.

    `slots=True` stores attributes in a fixed array instead of a per-instance
    dict — meaningfully less memory when you create millions of these.
    """

    alpha: float
    l1_ratio: float = 0.5

    def __post_init__(self) -> None:
        """Runs after the generated __init__ — the place for validation."""
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError(f"l1_ratio must be in [0, 1], got {self.l1_ratio}")


params = ModelParams(alpha=0.1)
print(f"params : {params}")
print(f"hashable → usable as a dict key : {hash(params) is not None}")

try:
    params.alpha = 0.5
except Exception as exc:
    print(f"\nimmutable : {type(exc).__name__} — {exc}")

try:
    ModelParams(alpha=0.1, l1_ratio=1.5)
except ValueError as exc:
    print(f"validated : {exc}")

# %% [markdown]
# ## 5. `@property` — computed and validated attributes
#
# A property looks like an attribute at the call site but runs code. Use it to
# validate on assignment, or to expose a derived value that should always be
# consistent with its inputs.

# %%
class Portfolio:
    """A set of asset weights that must always sum to 1."""

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights      # goes through the setter below

    @property
    def weights(self) -> dict[str, float]:
        """The asset weights."""
        return self._weights

    @weights.setter
    def weights(self, value: dict[str, float]) -> None:
        """Validate on every assignment, including in __init__."""
        total = sum(value.values())
        if not np.isclose(total, 1.0):        # float-safe comparison, per notebook 02
            raise ValueError(f"weights must sum to 1.0, got {total:.4f}")
        if any(w < 0 for w in value.values()):
            raise ValueError("short positions are not supported")
        self._weights = dict(value)

    @property
    def n_assets(self) -> int:
        """Derived, read-only — can never fall out of sync with `weights`."""
        return len(self._weights)

    @property
    def concentration(self) -> float:
        """Herfindahl index: 1/n for an equal-weight portfolio, 1.0 if fully concentrated."""
        return sum(w**2 for w in self._weights.values())


portfolio = Portfolio({"AAPL": 0.4, "MSFT": 0.35, "GOOG": 0.25})

print(f"weights       : {portfolio.weights}")
print(f"n_assets      : {portfolio.n_assets}        ← looks like an attribute")
print(f"concentration : {portfolio.concentration:.4f}")

try:
    portfolio.weights = {"AAPL": 0.5, "MSFT": 0.2}
except ValueError as exc:
    print(f"\nrejected      : {exc}")

try:
    portfolio.n_assets = 10
except AttributeError as exc:
    print(f"read-only     : {exc}")

# %% [markdown]
# ## 6. Inheritance vs. composition
#
# Inheritance expresses **"is-a"**. Composition expresses **"has-a"**. Modern
# practice strongly favours composition, because deep inheritance hierarchies
# become impossible to reason about.

# %%
from abc import ABC, abstractmethod


class Transformer(ABC):
    """Abstract base class defining a shared interface.

    `ABC` + `@abstractmethod` means this class cannot be instantiated, and any
    subclass MUST implement `transform`. That turns a documentation convention
    into something the interpreter enforces.
    """

    @abstractmethod
    def transform(self, values: np.ndarray) -> np.ndarray:
        """Return the transformed values."""

    def describe(self) -> str:
        """Concrete method — shared by every subclass."""
        return f"{type(self).__name__} transformer"


class LogTransformer(Transformer):
    def transform(self, values: np.ndarray) -> np.ndarray:
        # log1p computes log(1 + x) accurately for small x, and tolerates zeros.
        return np.log1p(values)


class SqrtTransformer(Transformer):
    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.sqrt(values)


data = np.array([0.0, 1.0, 10.0, 100.0])

for transformer in (LogTransformer(), SqrtTransformer()):
    result = transformer.transform(data)
    print(f"{transformer.describe():<28} {np.round(result, 3)}")

try:
    Transformer()
except TypeError as exc:
    print(f"\nabstract class : TypeError — {str(exc)[:70]}…")

# %% [markdown]
# ### Composition — the pattern you will use more often

# %%
class Pipeline:
    """Applies a sequence of transformers in order.

    Note that Pipeline does not INHERIT from Transformer — it HOLDS a list of
    them. New behaviour comes from combining objects, not from extending a
    class. This is exactly how `sklearn.pipeline.Pipeline` works.
    """

    def __init__(self, steps: list[Transformer]) -> None:
        self.steps = steps

    def transform(self, values: np.ndarray) -> np.ndarray:
        for step in self.steps:
            values = step.transform(values)
        return values

    def __repr__(self) -> str:
        names = " → ".join(type(s).__name__ for s in self.steps)
        return f"Pipeline({names})"


pipeline = Pipeline([SqrtTransformer(), LogTransformer()])

print(f"{pipeline}")
print(f"input  : {data}")
print(f"output : {np.round(pipeline.transform(data), 4)}")

# %% [markdown]
# ## 7. The scikit-learn estimator pattern
#
# Every scikit-learn object follows one contract. Implement it and your custom
# code drops straight into a `Pipeline`, a `GridSearchCV`, or a
# `cross_val_score` — this is the single most valuable class-writing skill in
# applied data science.
#
# | Method | Contract |
# | :--- | :--- |
# | `fit(X, y=None)` | **Learn** parameters from training data. Store them on `self` with a trailing underscore (`self.mean_`). Return `self`. |
# | `transform(X)` | **Apply** the learned parameters. Never learn here. |
# | `fit_transform(X)` | Convenience — inherited free from `TransformerMixin`. |
#
# > [!WARNING]
# > The trailing underscore is not decoration: it is the convention that marks an
# > attribute as *learned from data*. `self.strategy` is a hyperparameter the user
# > chose; `self.median_` is something `fit` computed.

# %%
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip features to a percentile range learned from the training data.

    A genuinely useful preprocessing step: it limits the influence of extreme
    values without discarding rows.

    Parameters
    ----------
    lower_percentile : float, default 1.0
        Values below this training percentile are clipped up to it.
    upper_percentile : float, default 99.0
        Values above this training percentile are clipped down to it.

    Attributes
    ----------
    lower_bounds_ : ndarray of shape (n_features,)
        Learned lower clip points. The trailing underscore marks it as fitted.
    upper_bounds_ : ndarray of shape (n_features,)
        Learned upper clip points.
    """

    def __init__(self, lower_percentile: float = 1.0,
                 upper_percentile: float = 99.0) -> None:
        # scikit-learn requires __init__ to do nothing but store its arguments
        # unchanged, under the same names. `get_params`/`clone` depend on it.
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        """Learn the clip bounds from X."""
        X = np.asarray(X, dtype=float)

        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("require 0 <= lower_percentile < upper_percentile <= 100")

        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        self.n_features_in_ = X.shape[1]

        return self       # returning self is what allows `.fit(X).transform(X)`

    def transform(self, X):
        """Clip X using the bounds learned during fit."""
        # Guard against being used before fitting — a clear error beats a
        # confusing AttributeError deep inside numpy.
        if not hasattr(self, "lower_bounds_"):
            raise RuntimeError("call fit() before transform()")

        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this was fitted on "
                f"{self.n_features_in_}"
            )

        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


# --- Try it out -------------------------------------------------------------
rng = np.random.default_rng(seed=42)

X_train = rng.normal(loc=50, scale=10, size=(1_000, 2))
X_train[0] = [500, -300]        # inject extreme outliers

clipper = OutlierClipper(lower_percentile=1, upper_percentile=99)
X_clipped = clipper.fit_transform(X_train)      # free, from TransformerMixin

print(f"learned lower bounds : {np.round(clipper.lower_bounds_, 2)}")
print(f"learned upper bounds : {np.round(clipper.upper_bounds_, 2)}")
print(f"\nbefore  min/max : {X_train.min(axis=0).round(1)} / {X_train.max(axis=0).round(1)}")
print(f"after   min/max : {X_clipped.min(axis=0).round(1)} / {X_clipped.max(axis=0).round(1)}")

# %% [markdown]
# ### The payoff: it composes with the rest of scikit-learn

# %%
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

# Our custom class sits alongside built-in ones as an equal citizen.
model = SklearnPipeline([
    ("clip", OutlierClipper(lower_percentile=2, upper_percentile=98)),
    ("scale", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])

y_train = X_train @ np.array([1.5, -2.0]) + rng.normal(0, 1, size=1_000)
model.fit(X_train, y_train)

print(model)
print(f"\nR² on training data : {model.score(X_train, y_train):.4f}")

# get_params works because __init__ only stored its arguments — this is what
# lets GridSearchCV tune our custom step (Module 13).
print(f"\ntunable parameters : {[k for k in model.get_params() if 'clip__' in k]}")

# %% [markdown]
# ## 8. Exercises
#
# ### Exercise 5.1 — Class or function?
#
# For each task, decide whether you would write a function, a dataclass, or a full
# class, and justify it in one sentence:
#
# 1. Converting a temperature from Celsius to Fahrenheit.
# 2. Holding the connection settings for a database.
# 3. A tokeniser that builds a vocabulary from a training corpus, then encodes new text.
# 4. Computing the median of a list.
#
# ### Exercise 5.2 — Dunder practice
#
# Extend `Measurement` with `__sub__`, `__mul__` (by a plain number), and
# `__format__` so that `f"{m:.2f}"` works.
#
# ### Exercise 5.3 — Dataclass conversion
#
# Rewrite the `Dataset` class from section 2 as a `@dataclass`. Keep `mean()` and
# `summary()`. How many lines did you remove?
#
# ### Exercise 5.4 — Build an estimator
#
# Write `RareCategoryGrouper(BaseEstimator, TransformerMixin)` that:
#
# - in `fit`, learns which categories appear in fewer than `min_frequency` rows;
# - in `transform`, replaces those categories with `"__other__"`.
#
# Confirm it works inside a `SklearnPipeline`. This is a real preprocessing step
# used to stop high-cardinality categorical columns from exploding.
#
# ### Exercise 5.5 — Find the bug
#
# ```python
# class Model:
#     history = []
#     def fit(self, X, y):
#         self.history.append("fitted")
#         return self
# ```
#
# Create two `Model` instances and fit each once. Explain `len(m1.history)`.

# %% [markdown]
# ## Key takeaways
#
# - Use a class when **state and behaviour belong together** — most often, when
#   something is *fitted* and then *applied*.
# - **Class attributes are shared.** Mutable ones are a bug waiting to happen;
#   initialise per-instance state in `__init__`.
# - Implement `__repr__` on every class you write — your future self will be
#   reading it in a debugger.
# - `@dataclass` removes the boilerplate; `frozen=True` gives immutable, hashable
#   configuration objects.
# - `@property` validates on assignment and exposes derived values safely.
# - **Prefer composition to inheritance.**
# - Follow the **scikit-learn estimator contract** — `fit` learns and stores
#   `attribute_`, `transform` applies, `__init__` only stores its arguments — and
#   your code composes with the entire ecosystem.
#
# ---
#
# **Next:** [06 · Files & Data Formats](06_files_and_data_formats.ipynb) — reading
# and writing data without corrupting it.
