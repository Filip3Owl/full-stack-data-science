# Failing Loudly
@subtitle Exceptions, logging and tests for pipelines you can trust
@time About 90 minutes
@notebook 07_errors_logging_and_testing.ipynb
@learn
- Exceptions: catching the right one, and why `except Exception:` hides bugs.
- `try / except / else / finally`, and exception chaining with `raise ... from`.
- Custom exceptions that make a pipeline's failures explain themselves.
- **Logging**, and why `print` doesn't survive contact with production.
- Defensive validation: fail fast, fail loudly.
- **`pytest`**: testing data transformations, including the tests that catch bugs unit tests usually miss.
@end

@lead There are two kinds of broken programs. One crashes, tells you where, and gets fixed by lunchtime. The other keeps running and produces a number that looks perfectly reasonable and is wrong. This chapter is about turning the second kind into the first. It's the difference between a script that *seems* to work and a pipeline you'd let run unattended.

## The exception hierarchy

Exceptions form a family tree of classes. Catching a class catches all of its descendants, which is exactly why catching too high up the tree is dangerous.

![The exceptions you'll meet weekly, and how they're related.](assets/images/exception_hierarchy.png){width=86}

You can verify the relationships yourself: `issubclass(KeyError, LookupError)` and `issubclass(FileNotFoundError, OSError)` are both `True`. One relationship is deliberately absent. `KeyboardInterrupt` and `SystemExit` inherit from `BaseException`, **not** `Exception`, so that `except Exception:` won't swallow your Ctrl-C. A bare `except:` catches `BaseException` too, which makes a runaway job impossible to stop.

> [!STORY] The first actual bug
> On 9 September 1947, operators of the Harvard Mark II computer traced a
> fault to a moth trapped in one of its relays. They taped it into the
> logbook with the note "First actual case of bug being found". Grace Hopper,
> who worked on the machine and loved telling the story, helped make it
> famous. Engineers had been calling glitches "bugs" since at least Thomas
> Edison's day, which is the joke: this time it was a *literal* one. The page,
> moth included, is now held by the Smithsonian.

## Catch narrowly

The rule: **catch the most specific exception you can actually do something about.** Let everything else propagate.

```python
def parse_amount_bad(raw: str) -> float:
    try:
        return float(raw)
    except Exception:
        return 0.0          # a typo and a null now look identical

def parse_amount_good(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None         # genuinely missing: an expected, legitimate state
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"could not parse {raw!r} as a number") from None
```

> [!WARNING]
> `parse_amount_bad` turns `"N/A"` into `0.0`. Sum a column of those and you
> get a revenue figure that is confidently, silently wrong. **Returning a
> plausible default on failure is how bad numbers reach dashboards.**

## `try / except / else / finally`

All four clauses have distinct jobs:

| Clause | Runs when | Use it for |
| :--- | :--- | :--- |
| `try` | always | the operation that might fail |
| `except` | a matching exception was raised | handling that specific failure |
| `else` | **no** exception was raised | follow-on work, kept out of `try` |
| `finally` | **always**, even after `return` or a re-raise | cleanup |

![The four clauses as a flow: finally runs on every path.](assets/images/book/m01_try_flow.png){width=100}

```python
def load_dataset(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None
    except pd.errors.EmptyDataError:
        return None
    else:
        print(f"loaded {len(df)} rows")   # outside `try`: its errors aren't masked
        return df
    finally:
        print("cleanup would go here")    # runs even after the returns above
```

Why the `else`? Code there runs only on success, and because it's outside the `try`, you can't accidentally catch an exception raised by the follow-on work.

### Exception chaining: `raise ... from`

When you translate a low-level error into a domain-level one, keep the original. `from exc` records the cause, so the traceback shows both the message you wrote *and* which file actually went missing:

```python
def load_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"configuration required at {path}") from exc
```

## Custom exceptions

Your own exception types let callers handle *your* failures specifically. In a pipeline, that's how you tell "the data is bad" from "the code is broken", two problems with very different owners. Give them a shared base class so that one `except` can catch them all:

```python
class DataValidationError(Exception):
    """Base class for every data-quality failure in this project."""

class MissingColumnsError(DataValidationError):
    def __init__(self, missing: set[str], available: set[str]) -> None:
        self.missing, self.available = missing, available
        super().__init__(f"missing required column(s): {sorted(missing)}. "
                         f"Available: {sorted(available)}")

class SchemaTypeError(DataValidationError):
    def __init__(self, column: str, expected: str, actual: str) -> None:
        super().__init__(f"column {column!r} should be {expected}, found {actual}")
```

A good error message says what was wrong **and** what was found instead. A validator raising these can sit inside a method chain because it returns the DataFrame unchanged when all is well:

```python
def validate_orders(df: pd.DataFrame) -> pd.DataFrame:
    required = {"order_id", "amount", "order_date"}
    if missing := required - set(df.columns):
        raise MissingColumnsError(missing, set(df.columns))
    if not pd.api.types.is_numeric_dtype(df["amount"]):
        raise SchemaTypeError("amount", "numeric", str(df["amount"].dtype))
    if (df["amount"] < 0).any():
        raise DataValidationError(f"{int((df['amount'] < 0).sum())} negative amount(s)")
    return df
```

> [!STORY] Pounds, newtons and a lost spacecraft
> On 23 September 1999, NASA's Mars Climate Orbiter fired its engine to slip
> into orbit around Mars and was never heard from again. The investigation
> found that ground software supplied by a contractor reported thruster
> impulse in **pound-force seconds**, while the navigation software that
> consumed those numbers expected **newton-seconds**, a factor of about 4.45.
> For months, small trajectory corrections had been computed with the wrong
> scale, and the probe flew far too close to the planet. Each team's code was
> fine on its own. What was missing was a check **at the boundary** between
> them: exactly the place this chapter says to validate.

## Logging

`print` is fine in a notebook. It's inadequate the moment code runs unattended, because it has no severity, no timestamp, no source, and can't be filtered or redirected without editing code.

| | `print` | `logging` |
| :--- | :---: | :---: |
| Severity levels | ✘ | ✔ |
| Timestamps | ✘ | ✔ |
| Which module said it | ✘ | ✔ |
| Silence it without editing code | ✘ | ✔ |
| Send it to a file or a service | ✘ | ✔ |

```python
import logging, sys

logging.basicConfig(                      # configure ONCE, at the entry point
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,                           # override Jupyter's own setup
)
logger = logging.getLogger("pipeline.ingest")   # one logger per module
logger.info("Loaded 200,000 rows from orders.csv")
logger.warning("3 rows had a null country; defaulted to 'unknown'")
```

> [!NOTE]
> Never call `basicConfig` inside a library. Configuration belongs to the
> application that *uses* the library, so it can decide where messages go.

### Choosing a level

![The five standard levels. Setting level=INFO hides DEBUG and keeps everything above it.](assets/images/book/m01_log_levels.png){width=100}

| Level | Use it for | Example |
| :--- | :--- | :--- |
| `DEBUG` | Diagnostics you want only when investigating | shapes, intermediate counts |
| `INFO` | Confirmation that things are proceeding | "loaded 200k rows" |
| `WARNING` | Something unexpected but recoverable | "12 nulls imputed" |
| `ERROR` | An operation failed; the run may continue | "one partition failed" |
| `CRITICAL` | The program cannot continue | "database unreachable" |

> [!TIP]
> Use lazy `%`-style arguments: `log.info("got %d rows", n)`, not an f-string.
> The message is only formatted if the level is enabled, so a suppressed
> `DEBUG` line in a hot loop costs almost nothing.

### Logging an exception with its traceback

Inside an `except` block, `logger.exception(...)` logs at `ERROR` level **and** attaches the full traceback. In a scheduled job, that's the difference between "something failed" and knowing what:

```python
try:
    result = a / b
except ZeroDivisionError:
    log.exception("Division failed for a=%s b=%s", a, b)
```

## Defensive validation: fail fast

The cheapest bug is the one caught at the boundary. Validate data *as it enters* a function, not three transformations later. A well-guarded function checks in three places:

```python
def summarise_revenue(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    # 1. Structural checks: cheap, and they catch most real mistakes
    if group_col not in df.columns:
        raise MissingColumnsError({group_col}, set(df.columns))
    if df.empty:
        raise DataValidationError("received an empty DataFrame")
    # 2. Semantic checks: assumptions the maths depends on
    if df["revenue"].isna().any():
        raise DataValidationError("null revenue values; impute or drop first")

    result = (df.groupby(group_col, observed=True)["revenue"]
                .agg(["count", "sum", "mean"]).round(2)
                .sort_values("sum", ascending=False))

    # 3. Postcondition: verify our OWN output
    assert len(result) <= df[group_col].nunique(), "more groups than exist"
    return result
```

> [!WARNING]
> **`assert` is for bugs, not for data validation.** Python deletes every
> `assert` when run with the `-O` (optimise) flag, so an assertion is not a
> guarantee. Use `raise` for anything that must always be checked, including
> every check on incoming data. Keep `assert` for "this can only fire if *my
> own* code is wrong".

## Testing with `pytest`

Data scientists test less than software engineers, and that's the main reason analysis bugs reach production. Yet a transformation is a *function*: input in, output out. It's the easiest thing in the world to test. Every test follows the same three beats:

```python
def test_<what>_<condition>_<expected>():
    # Arrange: build the inputs
    # Act:     call the thing
    # Assert:  check the result
```

Here's a realistic feature-engineering step and its tests:

```python
def add_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    if not {"amount", "quantity"}.issubset(df.columns):
        raise MissingColumnsError({"amount", "quantity"} - set(df.columns),
                                  set(df.columns))
    out = df.copy()                           # never mutate the caller's frame
    out["revenue"] = out["amount"] * out["quantity"]
    total = out["revenue"].sum()
    out["revenue_share"] = out["revenue"] / total if total else 0.0
    return out
```

```python
class TestAddRevenueFeatures:
    def test_computes_revenue_as_amount_times_quantity(self):
        df = pd.DataFrame({"amount": [10.0, 20.0], "quantity": [2, 3]})
        assert add_revenue_features(df)["revenue"].tolist() == [20.0, 60.0]

    def test_revenue_share_sums_to_one(self):
        df = pd.DataFrame({"amount": [10.0, 20.0, 5.0], "quantity": [1, 1, 1]})
        assert np.isclose(add_revenue_features(df)["revenue_share"].sum(), 1.0)

    def test_does_not_mutate_the_input(self):       # the one people forget
        df = pd.DataFrame({"amount": [10.0], "quantity": [2]})
        add_revenue_features(df)
        assert list(df.columns) == ["amount", "quantity"]

    def test_raises_on_missing_columns(self):
        with pytest.raises(MissingColumnsError, match="quantity"):
            add_revenue_features(pd.DataFrame({"amount": [10.0]}))

    def test_handles_all_zero_revenue(self):        # no division by zero
        df = pd.DataFrame({"amount": [0.0, 0.0], "quantity": [1, 2]})
        assert add_revenue_features(df)["revenue_share"].tolist() == [0.0, 0.0]

    @pytest.mark.parametrize(("amount", "quantity", "expected"), [
        (10.0, 1, 10.0), (0.0, 5, 0.0), (2.5, 4, 10.0),
        (-10.0, 2, -20.0),                            # refunds are negative
    ])
    def test_revenue_for_various_inputs(self, amount, quantity, expected):
        df = pd.DataFrame({"amount": [amount], "quantity": [quantity]})
        assert add_revenue_features(df)["revenue"].iloc[0] == expected
```

`@pytest.mark.parametrize` turns one test definition into four independent test cases. In a real project the tests live in `tests/`, and you run the whole suite by typing `pytest`. The repository's own suite runs with `.venv/bin/pytest`.

### What to test in data science code

| Test this | Because |
| :--- | :--- |
| **Shape**: row and column counts after a join | Silent row explosion from a many-to-many join is extremely common |
| **No mutation** of inputs | Chapter 2's aliasing bug, caught automatically |
| **Edge cases**: empty, one row, all-null, all-zero | Division by zero and empty aggregations |
| **Boundaries**: negative, zero, very large | Refunds, free items, outliers |
| **Invariants**: shares sum to 1, no nulls in a key | Encodes the assumption your maths relies on |
| **Known answers**: a tiny hand-computed example | Pins the definition down |

> [!TIP]
> Don't chase 100% coverage. Test the transformations whose failure would be
> **silent**: those are the ones that produce a plausible-looking wrong number.

## Try it yourself

Exercises 7.1 to 7.5 in notebook 07; solutions in notebook 09.

1. **Narrow the catch.** A `load_and_clean(path)` wraps reading, date parsing and float conversion in one bare `except:` that returns an empty DataFrame. List three distinct failures it hides, and rewrite it so each is handled or propagated properly.
2. **Custom exception hierarchy.** Design exceptions for an ETL job: source unreachable, schema mismatch, validation failure, destination write failure, all under one base class.
3. **Instrument a function.** Add logging to `add_revenue_features`: `INFO` on entry with the row count, `WARNING` if more than 5% of amounts are null, `DEBUG` with the resulting shape.
4. **Tests first.** Before implementing `pct_change(series)`, write tests for a normal series, a zero denominator, a single element, an empty series and nulls. Then make them pass.
5. **Catch a real bug.** Write a test that fails when `customers.merge(orders, on="customer_id", how="left")` silently multiplies rows because `orders` contains duplicate IDs. (Then look up pandas' `validate=` argument.)

> [!KEY]
> - **Catch narrowly.** `except Exception:` hides bugs; a bare `except:` swallows Ctrl-C too.
> - **Never return a plausible default on failure.** That's how wrong numbers reach dashboards.
> - Use `raise ... from exc` to keep the causal chain.
> - **Custom exceptions** let callers tell "bad data" from "broken code".
> - Use **`logging`**, not `print`: lazy `%s` arguments, one logger per module, `logger.exception` inside `except`.
> - **`assert` is for bugs; `raise` is for data.** Assertions vanish under `-O`.
> - Test **shape, mutation, edge cases and invariants**: the failures that are silent are the ones worth testing.
