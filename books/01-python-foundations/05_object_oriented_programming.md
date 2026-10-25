# Objects With a Purpose
@subtitle Classes, dataclasses, and the scikit-learn estimator contract
@time About 75 minutes
@notebook 05_object_oriented_programming.ipynb
@learn
- When a class is the right tool, and when a plain function is better.
- Classes, instances, and the trap hiding in class attributes.
- Dunder methods: making your objects work with `print`, `len`, `==` and `[]`.
- `@dataclass`, which deletes most of the boilerplate.
- `@property` for validated, computed attributes.
- Inheritance versus composition, and why the industry prefers composition.
- **The scikit-learn estimator pattern**, the interface you'll implement for real.
@end

@lead Object-oriented programming has a reputation for ceremony: diagrams, hierarchies, and classes that exist to hold other classes. In data science you need very little of that. You need a clear answer to one question (*when does a class actually earn its keep?*) and one pattern you'll use for the rest of your career.

## When should a data scientist write a class?

Honest answer: **less often than tutorials suggest.** Most analysis code should be functions operating on DataFrames. Reach for a class when you have **state and behaviour that belong together**.

| Situation | Use |
| :--- | :--- |
| Transform data once, return a result | a **function** |
| Group related constants | a **module** or an `Enum` |
| Carry a fixed set of fields around | a **`@dataclass`** |
| Something *learns* parameters, then *applies* them | a **class** |
| Several interchangeable implementations behind one interface | a **class hierarchy** or a `Protocol` |

The fourth row is the big one. A scaler learns a mean and a standard deviation from training data, then applies them to new data. That *fitted state* is exactly what an object is for.

> [!STORY] Classes were invented to simulate ships and queues
> The first language with classes and objects was **Simula 67**, built in
> Oslo by Ole-Johan Dahl and Kristen Nygaard at the Norwegian Computing
> Center. They weren't trying to invent a programming philosophy; they were
> writing *simulations* (of ships, harbours, queues of customers) and wanted
> each simulated thing to bundle its own data with its own behaviour. Alan
> Kay later coined the term "object-oriented", and Dahl and Nygaard received
> the Turing Award in 2001. The original motive is still the right one:
> **a class is for something with state that does things.**

## Classes and instances

```python
class Dataset:
    """A named collection of numeric observations."""

    version = "1.0"                      # CLASS attribute: shared by all instances

    def __init__(self, name: str, values: list[float]) -> None:
        self.name = name                 # INSTANCE attributes: one per object
        self.values = list(values)       # a copy, so the caller's list stays independent

    def mean(self) -> float:
        return sum(self.values) / len(self.values)

sales = Dataset("q1_sales", [120.0, 340.5, 89.0, 560.25, 210.0])
sales.mean()
```

Two things worth knowing. `__init__` is not a constructor: the object already exists when it runs, and its job is to set up instance attributes. And `self` is just the instance the method was called on: `sales.mean()` is really `Dataset.mean(sales)`.

### The class-attribute trap

Class attributes are shared. If a **mutable** one is changed through any instance, every instance sees the change. It's Chapter 2's aliasing lesson in a new costume:

```python
class BrokenLogger:
    entries = []                         # BUG: one list for every instance, ever

    def log(self, message: str) -> None:
        self.entries.append(message)

a, b = BrokenLogger(), BrokenLogger()
a.log("from a"); b.log("from b")
b.entries        # ['from a', 'from b']   ← the same list!
```

The fix is to create per-instance state inside `__init__`: `self.entries: list[str] = []`.

## Dunder methods

"Dunder" (double underscore) methods let your objects join in with Python's built-in syntax. Implement a handful and a class feels native:

```python
class Measurement:
    def __init__(self, value: float, unit: str) -> None:
        self.value, self.unit = value, unit

    def __repr__(self) -> str:      # for developers: looks like code that rebuilds it
        return f"Measurement(value={self.value!r}, unit={self.unit!r})"

    def __str__(self) -> str:       # for people: used by print()
        return f"{self.value:g} {self.unit}"

    def __eq__(self, other) -> bool:          # value equality instead of identity
        if not isinstance(other, Measurement):
            return NotImplemented
        return (self.value, self.unit) == (other.value, other.unit)

    def __hash__(self) -> int:      # defining __eq__ removes the default hash
        return hash((self.value, self.unit))

    def __lt__(self, other) -> bool:          # enables <, and therefore sorted()
        return self.value < other.value

    def __add__(self, other) -> "Measurement":  # enables +
        return Measurement(self.value + other.value, self.unit)
```

> [!TIP]
> Implement `__repr__` on every class you write. Containers, debuggers and
> error messages all show it, and your future self will be reading it at
> some inconvenient hour.

### Container dunders

Add `__len__`, `__getitem__` and `__iter__` and your class works with `len()`, `[]`, `for`, `in`, `max()`, `sum()` and unpacking, all for free. Supporting slices in `__getitem__` costs one `isinstance(index, slice)` check and makes the class behave the way users expect a sequence to.

## `@dataclass`: delete the boilerplate

Most classes in data work just hold fields. `@dataclass` writes `__init__`, `__repr__` and `__eq__` for you from type-annotated attributes:

```python
from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    name: str
    learning_rate: float = 0.01
    max_epochs: int = 100
    features: list[str] = field(default_factory=list)   # a fresh list per instance
    seed: int = 42
```

Notice `field(default_factory=list)`. The dataclass machinery *refuses* a plain `= []` default, catching Chapter 4's mutable-default trap at class-definition time instead of letting it bite at runtime.

### Frozen dataclasses for configuration

`frozen=True` makes instances immutable and hashable. It's the right choice for configuration: no code path can quietly rewrite a hyperparameter halfway through a run. `slots=True` stores attributes in a fixed array instead of a per-instance dict, saving real memory when you create millions of objects. And `__post_init__` is the place for validation:

```python
@dataclass(frozen=True, slots=True)
class ModelParams:
    alpha: float
    l1_ratio: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError(f"l1_ratio must be in [0, 1], got {self.l1_ratio}")

params = ModelParams(alpha=0.1)
params.alpha = 0.5        # FrozenInstanceError
```

## `@property`: computed and validated attributes

A property looks like an attribute from outside but runs code inside. Use it to **validate on assignment**, or to expose a **derived value** that can never drift out of sync with its inputs:

```python
class Portfolio:
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights                     # runs the setter below

    @property
    def weights(self) -> dict[str, float]:
        return self._weights

    @weights.setter
    def weights(self, value: dict[str, float]) -> None:
        total = sum(value.values())
        if not np.isclose(total, 1.0):             # float-safe, per Chapter 2
            raise ValueError(f"weights must sum to 1.0, got {total:.4f}")
        self._weights = dict(value)

    @property
    def n_assets(self) -> int:                     # read-only and always in sync
        return len(self._weights)
```

## Inheritance versus composition

Inheritance expresses **"is a"**. Composition expresses **"has a"**. Modern practice strongly favours composition, because deep inheritance trees become impossible to reason about.

![Inheritance shares an interface down a tree. Composition builds behaviour by combining independent parts.](assets/images/book/m01_composition.png){width=100}

Inheritance still has a good use: defining a shared **interface**. An abstract base class turns "every transformer must have a `transform` method" from a polite convention into a rule the interpreter enforces:

```python
from abc import ABC, abstractmethod

class Transformer(ABC):
    @abstractmethod
    def transform(self, values: np.ndarray) -> np.ndarray: ...

class LogTransformer(Transformer):
    def transform(self, values):
        return np.log1p(values)        # log(1 + x): accurate near 0, tolerates zeros

Transformer()     # TypeError: can't instantiate an abstract class
```

Composition is the pattern you'll use more. A `Pipeline` doesn't *inherit* from `Transformer`; it *holds a list of them* and calls each in turn, which is exactly how scikit-learn's own `Pipeline` works:

```python
class Pipeline:
    def __init__(self, steps: list[Transformer]) -> None:
        self.steps = steps

    def transform(self, values):
        for step in self.steps:
            values = step.transform(values)
        return values
```

> [!NOTE]
> "Favour object composition over class inheritance" is one of the central
> pieces of advice in *Design Patterns* (1994), the so-called "Gang of Four"
> book by Gamma, Helm, Johnson and Vlissides. Thirty years on, it still holds.

## The scikit-learn estimator pattern

Every scikit-learn object follows one contract. Implement it and your own code drops straight into a `Pipeline`, a `GridSearchCV` or `cross_val_score`. It's the single most valuable class-writing skill in applied data science.

| Method | Contract |
| :--- | :--- |
| `__init__(...)` | **Only store** the arguments, unchanged, under the same names. No validation, no computation. |
| `fit(X, y=None)` | **Learn** from training data. Store results with a trailing underscore (`self.mean_`). Return `self`. |
| `transform(X)` | **Apply** what was learned. Never learn here. |
| `fit_transform(X)` | Free, inherited from `TransformerMixin`. |

![The estimator contract: fit learns state from training data; transform applies that state to any data and never learns.](assets/images/book/m01_estimator.png){width=100}

> [!WARNING]
> The trailing underscore isn't decoration: it marks an attribute as *learned
> from data*. `self.lower_percentile` is a hyperparameter the user chose;
> `self.lower_bounds_` is something `fit` computed. Mixing them up breaks
> cloning, grid search and your colleagues' expectations.

Here's a genuinely useful custom transformer: it learns per-column percentile bounds from the training data and clips extreme values to them, taming outliers without dropping rows.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile: float = 1.0,
                 upper_percentile: float = 99.0) -> None:
        self.lower_percentile = lower_percentile      # just store them
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        self.n_features_in_ = X.shape[1]
        return self                         # enables .fit(X).transform(X)

    def transform(self, X):
        if not hasattr(self, "lower_bounds_"):
            raise RuntimeError("call fit() before transform()")
        return np.clip(np.asarray(X, dtype=float),
                       self.lower_bounds_, self.upper_bounds_)
```

### The payoff: it composes

Because it honours the contract, our class sits alongside scikit-learn's own steps as an equal:

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

model = SklearnPipeline([
    ("clip", OutlierClipper(lower_percentile=2, upper_percentile=98)),
    ("scale", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
model.fit(X_train, y_train)
[k for k in model.get_params() if k.startswith("clip__")]
# ['clip__lower_percentile', 'clip__upper_percentile']  ← tunable by GridSearchCV
```

`get_params` works only because `__init__` did nothing but store its arguments. That's what lets `GridSearchCV` tune our custom step in Module 13.

> [!STORY] A summer project that became the industry's grammar
> scikit-learn began in 2007 as a Google Summer of Code project by David
> Cournapeau. A team at INRIA, France's national computing research
> institute (Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort and Vincent
> Michel), picked it up and made the first public release in 2010. In 2013
> its developers wrote a paper about their API design, arguing that a small,
> consistent interface (`fit`, `predict`, `transform`, plain parameters in
> `__init__`) matters more than any single algorithm. They were right: today
> dozens of other libraries copy that interface so that they can plug into
> scikit-learn's pipelines.

## Try it yourself

Exercises 5.1 to 5.5 in notebook 05; solutions in notebook 09.

1. **Class or function?** Decide, with one sentence each: Celsius → Fahrenheit; database connection settings; a tokeniser that builds a vocabulary from a corpus and then encodes new text; the median of a list.
2. **Dunder practice.** Give `Measurement` a `__sub__`, a `__mul__` by a plain number, and a `__format__` so that `f"{m:.2f}"` works.
3. **Dataclass conversion.** Rewrite `Dataset` as a `@dataclass`, keeping `mean()` and `summary()`. How many lines disappeared?
4. **Build an estimator.** `RareCategoryGrouper` learns in `fit` which categories appear fewer than `min_frequency` times, and in `transform` replaces them with `"__other__"`. Check it inside a scikit-learn `Pipeline`.
5. **Find the bug.** A `Model` class keeps `history = []` at class level and appends to it in `fit`. Fit two instances once each and explain `len(m1.history)`.

> [!KEY]
> - Use a class when **state and behaviour belong together**, most often when something is *fitted* and then *applied*.
> - **Class attributes are shared.** Initialise per-instance state in `__init__`.
> - Implement `__repr__` on every class you write.
> - `@dataclass` removes boilerplate; `frozen=True` gives immutable, hashable configuration.
> - `@property` validates on assignment and exposes derived values safely.
> - **Prefer composition to inheritance**; use abstract base classes to define interfaces.
> - Follow the **scikit-learn contract**: `__init__` stores, `fit` learns into `attribute_` and returns `self`, `transform` applies. Then your code composes with the whole ecosystem.
