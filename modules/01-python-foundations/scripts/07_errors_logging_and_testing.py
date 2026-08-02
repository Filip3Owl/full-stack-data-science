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
# # 07 · Errors, Logging & Testing
#
# **Module 01 — Python Foundations for Data Science**
#
# ---
#
# ## What you will learn
#
# 1. Exceptions: catching the right one, and why `except Exception:` hides bugs.
# 2. `try / except / else / finally`, and exception chaining.
# 3. Custom exceptions that make a pipeline's failures self-explaining.
# 4. **Logging** — and why `print` does not survive contact with production.
# 5. Defensive data validation: fail fast, fail loudly.
# 6. **`pytest`** — testing data transformations, including the tests that catch
#    the bugs unit tests normally miss.
#
# **Estimated time:** 90 minutes
#
# > This notebook is the difference between a script that *seems* to work and a
# > pipeline you would let run unattended.

# %%
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dstoolkit import setup_plot_style

setup_plot_style()

# %% [markdown]
# ## 1. The exception hierarchy
#
# Exceptions form a class tree. Catching a class catches every one of its
# subclasses — which is exactly why catching too high up is dangerous.
#
# ![Python exceptions you will meet weekly](../../../assets/images/exception_hierarchy.png)

# %%
# The relationships are real inheritance, and you can verify them:
print(f"ValueError is an Exception      : {issubclass(ValueError, Exception)}")
print(f"KeyError is a LookupError       : {issubclass(KeyError, LookupError)}")
print(f"FileNotFoundError is an OSError : {issubclass(FileNotFoundError, OSError)}")
print(f"KeyboardInterrupt is an Exception : "
      f"{issubclass(KeyboardInterrupt, Exception)}   ← note: it is NOT")

# %% [markdown]
# That last line matters. `KeyboardInterrupt` and `SystemExit` inherit from
# `BaseException`, *not* `Exception` — deliberately, so that `except Exception:`
# does not swallow your Ctrl-C. A bare `except:` catches `BaseException` and
# therefore makes a runaway job impossible to stop.

# %% [markdown]
# ## 2. Catch narrowly
#
# The rule: **catch the most specific exception you can actually do something
# about.** Everything else should propagate.

# %%
def parse_amount_bad(raw: str) -> float:
    """Catches everything, so every failure looks identical."""
    try:
        return float(raw)
    except Exception:
        return 0.0        # a typo and a null are now indistinguishable


def parse_amount_good(raw: str | None) -> float | None:
    """Handles the cases we understand; lets the rest surface."""
    if raw is None or raw.strip() == "":
        return None       # genuinely missing — a legitimate, expected state

    try:
        return float(raw)
    except ValueError:
        # A non-numeric string is a data-quality problem worth surfacing,
        # not something to paper over with a zero.
        raise ValueError(f"could not parse {raw!r} as a number") from None


for value in ["12.50", "", None, "N/A"]:
    bad = parse_amount_bad(value) if value is not None else "crashes"
    try:
        good = parse_amount_good(value)
    except ValueError as exc:
        good = f"ValueError: {exc}"
    print(f"  {str(value):<8} → bad: {str(bad):<8} good: {good}")

# %% [markdown]
# > [!WARNING]
# > `parse_amount_bad` turns `"N/A"` into `0.0`. Sum a column of those and you get
# > a revenue figure that is confidently, silently wrong. **Returning a plausible
# > default on failure is how bad numbers reach dashboards.**

# %% [markdown]
# ## 3. `try / except / else / finally`
#
# All four clauses have distinct jobs:
#
# | Clause | Runs when | Use it for |
# | :--- | :--- | :--- |
# | `try` | always | the operation that might fail |
# | `except` | an exception matched | handling that specific failure |
# | `else` | **no** exception was raised | the follow-on work, kept out of `try` |
# | `finally` | **always**, even on `return` or re-raise | cleanup |

# %%
def load_dataset(path: Path) -> pd.DataFrame | None:
    """Demonstrate all four clauses on a realistic operation."""
    print(f"  attempting {path.name}")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("    ✘ file not found — returning None")
        return None
    except pd.errors.EmptyDataError:
        print("    ✘ file is empty")
        return None
    else:
        # Only reached on success. Keeping this OUT of the `try` block means we
        # never accidentally catch an exception raised by the follow-on code.
        print(f"    ✔ loaded {len(df)} rows")
        return df
    finally:
        # Always runs — including after the `return` statements above.
        print("    (finally: cleanup would go here)")


tmp = Path("nonexistent_file.csv")
result = load_dataset(tmp)
print(f"  result : {result}")

# %% [markdown]
# ### Exception chaining: `raise ... from`
#
# When you convert a low-level error into a domain-level one, preserve the
# original. `from exc` keeps the full causal chain in the traceback.

# %%
def load_config(path: Path) -> dict:
    try:
        import json
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        # `from exc` records the underlying cause, so a reader of the traceback
        # sees both "config missing" AND which path failed.
        raise RuntimeError(f"configuration required at {path}") from exc


try:
    load_config(Path("missing_config.json"))
except RuntimeError as exc:
    print(f"raised  : {exc}")
    print(f"caused by : {type(exc.__cause__).__name__} — {exc.__cause__}")

# %% [markdown]
# ## 4. Custom exceptions
#
# Defining your own exception types lets callers handle *your* failures
# specifically. In a data pipeline this is how you distinguish "the data is bad"
# from "the code is broken" — two problems with very different owners.

# %%
class DataValidationError(Exception):
    """Base class for every data-quality failure in this project.

    A common base means callers can catch all of them with one clause while
    still being able to target a specific subtype.
    """


class MissingColumnsError(DataValidationError):
    """A required column is absent."""

    def __init__(self, missing: set[str], available: set[str]) -> None:
        self.missing = missing
        self.available = available
        # A good message states what was wrong AND what was found instead.
        super().__init__(
            f"missing required column(s): {sorted(missing)}. "
            f"Available: {sorted(available)}"
        )


class SchemaTypeError(DataValidationError):
    """A column has an unexpected dtype."""

    def __init__(self, column: str, expected: str, actual: str) -> None:
        self.column = column
        super().__init__(
            f"column {column!r} should be {expected}, found {actual}"
        )


def validate_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Check an orders DataFrame against the expected contract.

    Returns the DataFrame unchanged so this can sit inside a method chain.
    """
    required = {"order_id", "amount", "order_date"}
    missing = required - set(df.columns)
    if missing:
        raise MissingColumnsError(missing, set(df.columns))

    if not pd.api.types.is_numeric_dtype(df["amount"]):
        raise SchemaTypeError("amount", "numeric", str(df["amount"].dtype))

    if (df["amount"] < 0).any():
        n_bad = int((df["amount"] < 0).sum())
        raise DataValidationError(f"{n_bad} row(s) have a negative amount")

    return df


good = pd.DataFrame({
    "order_id": [1, 2],
    "amount": [10.0, 20.0],
    "order_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
})
print(f"valid frame passes : {len(validate_orders(good))} rows\n")

for name, frame in {
    "missing column": pd.DataFrame({"order_id": [1], "amount": [10.0]}),
    "wrong dtype": good.assign(amount=["10.0", "20.0"]),
    "negative amount": good.assign(amount=[10.0, -5.0]),
}.items():
    try:
        validate_orders(frame)
    except DataValidationError as exc:
        # One `except` clause catches all three subtypes.
        print(f"  {name:<16} → {type(exc).__name__}: {exc}")

# %% [markdown]
# ## 5. Logging
#
# `print` is fine in a notebook. It is inadequate the moment code runs unattended,
# because it carries no severity, no timestamp, no source, and cannot be filtered
# or redirected.
#
# | | `print` | `logging` |
# | :--- | :-: | :-: |
# | Severity levels | ✘ | ✔ |
# | Timestamps | ✘ | ✔ |
# | Which module emitted it | ✘ | ✔ |
# | Silence it without editing code | ✘ | ✔ |
# | Route to a file / service | ✘ | ✔ |

# %%
# Configure once, at the entry point of your program — never inside a library.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,      # so the output appears in the notebook
    force=True,             # override any configuration Jupyter already applied
)

# Convention: one logger per module, named after it. `__name__` gives you the
# module path for free, which is what makes log filtering possible later.
logger = logging.getLogger("pipeline.ingest")

logger.debug("Detailed diagnostics — hidden at INFO level")
logger.info("Loaded 200,000 rows from orders.csv")
logger.warning("3 rows had a null country; defaulted to 'unknown'")
logger.error("Failed to reach the currency API; using yesterday's rates")
logger.critical("Disk full — aborting the run")

# %% [markdown]
# ### Choosing a level
#
# | Level | Use it for | Example |
# | :--- | :--- | :--- |
# | `DEBUG` | Diagnostics you want only when investigating | shapes, intermediate counts |
# | `INFO` | Confirmation that things are proceeding | "loaded 200k rows" |
# | `WARNING` | Something unexpected, but recoverable | "12 nulls imputed" |
# | `ERROR` | An operation failed; the run may continue | "one partition failed" |
# | `CRITICAL` | The program cannot continue | "database unreachable" |

# %%
def process_batch(records: list[dict]) -> pd.DataFrame:
    """A realistic pipeline step, instrumented as it would be in production."""
    log = logging.getLogger("pipeline.transform")

    log.info("Starting batch of %d records", len(records))

    df = pd.DataFrame(records)

    n_missing = int(df["amount"].isna().sum())
    if n_missing:
        # Lazy %-formatting: the string is only built if the level is enabled.
        # With f-strings, formatting happens even for suppressed messages.
        log.warning("%d record(s) have a null amount; dropping them", n_missing)
        df = df.dropna(subset=["amount"])

    if df.empty:
        log.error("Batch is empty after cleaning — nothing to write")
        return df

    log.info("Batch complete: %d valid records, total %.2f",
             len(df), df["amount"].sum())
    return df


batch = [
    {"id": 1, "amount": 10.0},
    {"id": 2, "amount": None},
    {"id": 3, "amount": 25.5},
]
processed = process_batch(batch)

# %% [markdown]
# ### Logging an exception with its traceback
#
# `logger.exception` is only valid inside an `except` block, and it attaches the
# full traceback automatically. This is what you want in a scheduled job — without
# it, you get "something failed" and no way to find out what.

# %%
def risky_division(a: float, b: float) -> float | None:
    log = logging.getLogger("pipeline.compute")
    try:
        return a / b
    except ZeroDivisionError:
        log.exception("Division failed for a=%s b=%s", a, b)
        return None


print(f"result : {risky_division(10, 0)}")

# %% [markdown]
# > [!TIP]
# > Use lazy `%s` formatting (`log.info("got %d", n)`), not f-strings. At `INFO`
# > level a suppressed `DEBUG` message costs nothing; with an f-string you pay the
# > formatting cost on every call regardless.

# %% [markdown]
# ## 6. Defensive validation: fail fast
#
# The cheapest bug is the one caught at the boundary. Validate the shape of your
# data *as it enters* your pipeline, not three transformations later.

# %%
def summarise_revenue(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate revenue by a grouping column, with contract checks up front."""
    # 1. Structural checks — cheap, and catch the majority of real mistakes.
    if group_col not in df.columns:
        raise MissingColumnsError({group_col}, set(df.columns))
    if df.empty:
        raise DataValidationError("received an empty DataFrame")

    # 2. Semantic checks — assumptions the maths depends on.
    if df["revenue"].isna().any():
        raise DataValidationError(
            f"{int(df['revenue'].isna().sum())} null revenue value(s); "
            "impute or drop before aggregating"
        )

    result = (
        df.groupby(group_col, observed=True)["revenue"]
        .agg(["count", "sum", "mean"])
        .round(2)
        .sort_values("sum", ascending=False)
    )

    # 3. Postcondition — verify our own output before handing it back.
    # An assertion is right here: this fires only if THIS function has a bug,
    # which is exactly what assertions are for.
    assert len(result) <= df[group_col].nunique(), "produced more groups than exist"

    return result


sales = pd.DataFrame({
    "channel": ["organic", "paid", "organic", "email", "paid"],
    "revenue": [100.0, 250.0, 75.5, 40.0, 310.0],
})
summarise_revenue(sales, "channel")

# %% [markdown]
# > [!WARNING]
# > **`assert` is for bugs, not for data validation.** Python removes every
# > `assert` statement when run with `-O` (optimised mode), so an assertion is not
# > a guarantee. Use `raise` for anything that must always be checked — including
# > all validation of incoming data.

# %% [markdown]
# ## 7. Testing with `pytest`
#
# Data scientists test less than software engineers, and it is the main reason
# analysis bugs reach production. A transformation is a *function*: it takes input
# and returns output. That is the easiest thing in the world to test.
#
# ### Anatomy of a test
#
# ```python
# def test_<what>_<condition>_<expected>():
#     # Arrange — build the inputs
#     # Act     — call the thing
#     # Assert  — check the result
# ```

# %%
# The function under test — a realistic feature-engineering step.
def add_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add revenue and revenue-share columns.

    Parameters
    ----------
    df : DataFrame
        Must contain `amount` (float) and `quantity` (int).

    Returns
    -------
    DataFrame
        A **new** DataFrame with `revenue` and `revenue_share` added.
    """
    if not {"amount", "quantity"}.issubset(df.columns):
        raise MissingColumnsError(
            {"amount", "quantity"} - set(df.columns), set(df.columns)
        )

    out = df.copy()                       # never mutate the caller's frame
    out["revenue"] = out["amount"] * out["quantity"]

    total = out["revenue"].sum()
    # Guard the division: an all-zero frame is valid input, not an error.
    out["revenue_share"] = out["revenue"] / total if total else 0.0

    return out


demo = pd.DataFrame({"amount": [10.0, 20.0], "quantity": [2, 3]})
add_revenue_features(demo)

# %% [markdown]
# ### Writing the tests
#
# Normally these live in `tests/`. We define them inline here so you can see them
# run, then look at the real suite in the repository.

# %%
import pytest


class TestAddRevenueFeatures:
    """Tests for add_revenue_features, grouped by behaviour."""

    def test_computes_revenue_as_amount_times_quantity(self):
        df = pd.DataFrame({"amount": [10.0, 20.0], "quantity": [2, 3]})
        result = add_revenue_features(df)
        assert result["revenue"].tolist() == [20.0, 60.0]

    def test_revenue_share_sums_to_one(self):
        df = pd.DataFrame({"amount": [10.0, 20.0, 5.0], "quantity": [1, 1, 1]})
        result = add_revenue_features(df)
        # Float-safe comparison — the lesson from notebook 02, applied.
        assert np.isclose(result["revenue_share"].sum(), 1.0)

    def test_does_not_mutate_the_input(self):
        # THE test data scientists most often forget, and the bug that hurts most.
        df = pd.DataFrame({"amount": [10.0], "quantity": [2]})
        original_columns = list(df.columns)
        add_revenue_features(df)
        assert list(df.columns) == original_columns

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"amount": [10.0]})
        with pytest.raises(MissingColumnsError, match="quantity"):
            add_revenue_features(df)

    # --- Edge cases: where the real bugs hide -----------------------------
    def test_handles_all_zero_revenue_without_dividing_by_zero(self):
        df = pd.DataFrame({"amount": [0.0, 0.0], "quantity": [1, 2]})
        result = add_revenue_features(df)
        assert result["revenue_share"].tolist() == [0.0, 0.0]

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame({"amount": [], "quantity": []})
        result = add_revenue_features(df)
        assert len(result) == 0
        assert "revenue" in result.columns

    @pytest.mark.parametrize(
        ("amount", "quantity", "expected"),
        [
            (10.0, 1, 10.0),
            (0.0, 5, 0.0),
            (2.5, 4, 10.0),
            (-10.0, 2, -20.0),      # refunds are legitimately negative
        ],
    )
    def test_revenue_for_various_inputs(self, amount, quantity, expected):
        """One test definition, four independent test cases."""
        df = pd.DataFrame({"amount": [amount], "quantity": [quantity]})
        assert add_revenue_features(df)["revenue"].iloc[0] == expected


print("Tests defined. Running them…")

# %% [markdown]
# ### Running them
#
# In a real project you would simply type `pytest` in the terminal and it would
# discover every `test_*.py` file. To show the runner working from inside a
# notebook, we write the same code out to a temporary file and invoke pytest on it.

# %%
import subprocess
import tempfile

TEST_FILE_SOURCE = '''
"""A self-contained version of the tests above, for pytest to collect."""
import numpy as np
import pandas as pd
import pytest


class MissingColumnsError(Exception):
    def __init__(self, missing, available):
        self.missing = missing
        super().__init__(
            f"missing required column(s): {sorted(missing)}. "
            f"Available: {sorted(available)}"
        )


def add_revenue_features(df):
    if not {"amount", "quantity"}.issubset(df.columns):
        raise MissingColumnsError(
            {"amount", "quantity"} - set(df.columns), set(df.columns)
        )
    out = df.copy()
    out["revenue"] = out["amount"] * out["quantity"]
    total = out["revenue"].sum()
    out["revenue_share"] = out["revenue"] / total if total else 0.0
    return out


class TestAddRevenueFeatures:
    def test_computes_revenue_as_amount_times_quantity(self):
        df = pd.DataFrame({"amount": [10.0, 20.0], "quantity": [2, 3]})
        assert add_revenue_features(df)["revenue"].tolist() == [20.0, 60.0]

    def test_revenue_share_sums_to_one(self):
        df = pd.DataFrame({"amount": [10.0, 20.0, 5.0], "quantity": [1, 1, 1]})
        result = add_revenue_features(df)
        assert np.isclose(result["revenue_share"].sum(), 1.0)

    def test_does_not_mutate_the_input(self):
        df = pd.DataFrame({"amount": [10.0], "quantity": [2]})
        original_columns = list(df.columns)
        add_revenue_features(df)
        assert list(df.columns) == original_columns

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"amount": [10.0]})
        with pytest.raises(MissingColumnsError, match="quantity"):
            add_revenue_features(df)

    def test_handles_all_zero_revenue_without_dividing_by_zero(self):
        df = pd.DataFrame({"amount": [0.0, 0.0], "quantity": [1, 2]})
        assert add_revenue_features(df)["revenue_share"].tolist() == [0.0, 0.0]

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame({"amount": [], "quantity": []})
        result = add_revenue_features(df)
        assert len(result) == 0 and "revenue" in result.columns

    @pytest.mark.parametrize(
        ("amount", "quantity", "expected"),
        [(10.0, 1, 10.0), (0.0, 5, 0.0), (2.5, 4, 10.0), (-10.0, 2, -20.0)],
    )
    def test_revenue_for_various_inputs(self, amount, quantity, expected):
        df = pd.DataFrame({"amount": [amount], "quantity": [quantity]})
        assert add_revenue_features(df)["revenue"].iloc[0] == expected
'''

with tempfile.TemporaryDirectory() as tmpdir:
    test_file = Path(tmpdir) / "test_revenue_features.py"
    test_file.write_text(TEST_FILE_SOURCE)

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file),
         "-v", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
    )

# Print the per-test result lines, trimming pytest's boilerplate.
for line in completed.stdout.splitlines():
    if "::" in line or " passed" in line or " failed" in line:
        print(line.replace(str(test_file), "test_revenue_features.py"))

# %% [markdown]
# ### The project's real test suite
#
# The repository has a working suite in `tests/`. Run it any time with:
#
# ```bash
# .venv/bin/pytest
# ```

# %%
repo_root = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)

completed = subprocess.run(
    [sys.executable, "-m", "pytest", str(repo_root / "tests"), "-q", "--no-header"],
    capture_output=True,
    text=True,
    cwd=repo_root,
)
print(completed.stdout[-900:])

# %% [markdown]
# ### What to test in data science code
#
# | Test this | Because |
# | :--- | :--- |
# | **Shape** — row/column counts after a join | Silent row explosion from a many-to-many join is extremely common |
# | **No mutation** of inputs | The notebook-02 aliasing bug, caught automatically |
# | **Edge cases** — empty, single row, all-null, all-zero | Division by zero and empty aggregations |
# | **Boundaries** — negative, zero, very large | Refunds, free items, outliers |
# | **Invariants** — "shares sum to 1", "no nulls in a key" | Encodes the assumption your maths depends on |
# | **Known-answer cases** | A tiny hand-computed example pins the definition down |
#
# > [!TIP]
# > Do not chase 100% coverage. Test the transformations whose failure would be
# > **silent** — those are the ones that produce a plausible-looking wrong number.

# %% [markdown]
# ## 8. Exercises
#
# ### Exercise 7.1 — Narrow the catch
#
# ```python
# def load_and_clean(path):
#     try:
#         df = pd.read_csv(path)
#         df["date"] = pd.to_datetime(df["date"])
#         df["amount"] = df["amount"].astype(float)
#         return df
#     except:
#         return pd.DataFrame()
# ```
#
# List three distinct failures this hides. Rewrite it so each is handled or
# propagated appropriately.
#
# ### Exercise 7.2 — Custom exception hierarchy
#
# Design exceptions for an ETL pipeline covering: source unreachable, schema
# mismatch, validation failure, and destination write failure. They should share a
# base class so a caller can catch them all at once.
#
# ### Exercise 7.3 — Instrument a function
#
# Take `add_revenue_features` and add logging: `INFO` on entry with the row count,
# `WARNING` if more than 5% of amounts are null, `DEBUG` with the resulting shape.
#
# ### Exercise 7.4 — Write the tests first
#
# Before writing any implementation, write `pytest` tests for a function
# `pct_change(series)` returning period-over-period percentage change. Cover: a
# normal series, a zero in the denominator, a single element, an empty series, and
# a series containing nulls. *Then* implement it so the tests pass.
#
# ### Exercise 7.5 — Catch a real bug
#
# ```python
# def merge_customer_orders(customers, orders):
#     return customers.merge(orders, on="customer_id", how="left")
# ```
#
# Write a test that fails when `orders` contains duplicate `customer_id` values
# and the merge silently multiplies rows. This exact bug is one of the most common
# in production analytics.

# %% [markdown]
# ## Key takeaways
#
# - **Catch narrowly.** `except Exception:` hides bugs; a bare `except:` also
#   swallows Ctrl-C.
# - **Never return a plausible default on failure** — that is how wrong numbers
#   reach dashboards.
# - Use `raise ... from exc` to keep the causal chain intact.
# - **Custom exceptions** let callers distinguish "bad data" from "broken code".
# - Use **`logging`**, not `print`, with lazy `%s` formatting and one logger per
#   module. `logger.exception` inside `except` captures the traceback.
# - **`assert` is for bugs; `raise` is for data.** Assertions vanish under `-O`.
# - Test **shape, mutation, edge cases, and invariants** — the failures that are
#   silent are the ones worth testing.
#
# ---
#
# **Next:** [08 · Module Project](08_module_project.ipynb) — put all of it together
# in a tested, logged, end-to-end pipeline.
