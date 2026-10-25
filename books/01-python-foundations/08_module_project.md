# Project: A Pipeline You Can Trust
@subtitle Every idea in the module, in one tested ETL job
@time About 120 minutes
@notebook 08_module_project.ipynb
@learn
- How to turn two badly behaved exports into one clean, typed Parquet table.
- Separating a thin **impure shell** (I/O) from a **pure core** (transformations).
- Counting, logging and reporting every row the pipeline drops, with the reason.
- A **quality gate** that refuses to publish output it can't vouch for.
- Testing the pieces, and proving the safety net actually fires.
@end

@lead Every chapter so far has handed you a tool. This one hands you a job. You've just joined the data team at a fictional retailer, Nordwind, and on your first morning two nightly exports land in your lap. Neither is well behaved.

## The brief

- **`orders.csv`** comes from the order-management system. It's Latin-1 encoded, uses European decimal commas, has inconsistent casing, writes `"N/A"` for nulls, and contains duplicate rows whenever a retry fires.
- **`customers.json`** comes from the CRM, as a nested API dump.

**Your job:** turn these into one clean, typed, analysis-ready Parquet table, and produce a short revenue readout from it. The pipeline must be **defensive** (bad data fails loudly), **observable** (logged) and **tested**.

| Chapter | What the project uses from it |
| :--- | :--- |
| 2 | Immutability, float comparison, string cleaning, `nan` semantics |
| 3 | `set` lookups, `defaultdict`, comprehensions, sorting |
| 4 | Pure functions, keyword-only arguments, closures, decorators |
| 5 | A frozen `@dataclass` for configuration, a mutable one for the run report |
| 6 | `pathlib`, encodings, CSV → Parquet |
| 7 | Custom exceptions, logging, validation, `pytest` |

![The shape of the pipeline: a pure core of transformations wrapped in a thin shell that touches files, with a quality gate before anything is published.](assets/images/book/m01_pipeline.png){width=100}

## Step 0: generate the mess

In real life the data arrives from upstream. The notebook synthesises it, deliberately messy and reproducibly so, with a fixed seed. Seven defects are planted, each one met in production by the course's author: channels like `" PAID"` and `"organic "`, missing amounts written four different ways (`"N/A"`, `""`, `"-"`, `"null"`), decimal commas, a dozen negative quantities, duplicated rows from retries, orders pointing at a customer who doesn't exist, and a semicolon-separated, Latin-1 encoded file.

> [!TIP]
> The notebook's first move is to print the **raw bytes** of the file before
> assuming anything about it. That ten-second habit catches encoding and
> separator problems that otherwise cost half an hour.

## Step 1: write the contract down first

Before any transformation, state what "clean" means. Encoding the rules as a **frozen dataclass** makes them explicit, testable, and changeable in exactly one place:

```python
@dataclass(frozen=True)
class PipelineConfig:
    orders_path: Path
    customers_path: Path
    output_path: Path
    min_amount: float = 0.01                    # business rules with names,
    max_amount: float = 100_000.0               # not magic numbers scattered
    valid_channels: frozenset[str] = frozenset(  # through the code
        {"organic", "paid", "referral", "email"})
    max_reject_rate: float = 0.10               # fail the run above this

    def __post_init__(self) -> None:
        if not 0 <= self.max_reject_rate <= 1:
            raise ValueError("max_reject_rate must be between 0 and 1")
```

Alongside it sits a *mutable* dataclass, `PipelineReport`, that records what happened during a run: rows read, duplicates removed, unparseable amounts, orphan references, rows written. Its `reject_rate` is a `@property`, so it can never disagree with the counts it's computed from.

## Step 2: domain exceptions

```python
class PipelineError(Exception): ...            # catch-all for callers
class SourceReadError(PipelineError): ...      # a file is missing or unreadable
class SchemaError(PipelineError): ...          # expected columns are absent
class DataQualityError(PipelineError): ...     # too much input was unusable
```

## Step 3: extract (the impure shell)

The extract functions are the only ones that touch the filesystem. They read with the quirks declared **explicitly** (`encoding="latin-1"`, `sep=";"`, everything as `str`) and validate the column set immediately, raising `SchemaError` with a helpful message if it's wrong. Reading everything as text is deliberate: nothing in that file can be trusted to parse itself, so we parse it ourselves, on purpose.

## Step 4: transform (the pure core)

Every function from here on is **pure**: DataFrame in, new DataFrame out, no mutation of its input, no I/O. That's what makes each one testable on its own.

### Parsing European amounts

```python
def parse_amount(raw: str | float | None) -> float | None:
    """'1.234,56' → 1234.56; any missing-value token → None."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip()
    if text.casefold() in {"", "n/a", "na", "-", "null", "none", "nan"}:
        return None                    # O(1) set membership, Chapter 3's habit
    try:
        return float(text.replace(".", "").replace(",", "."))
    except ValueError:
        return None                    # surface it as missing, never as 0
```

Returning `None`, not `0`, lets the caller **count** the failures instead of hiding them.

### The other steps

- **`deduplicate`** drops exact duplicate rows and records how many.
- **`normalise_columns`** coerces IDs with `pd.to_numeric(..., errors="coerce")`, parses dates, trims and lower-cases text with `.str.casefold()`, and maps `parse_amount` over the amounts.
- **`apply_business_rules`** removes rows that break the contract, **counting each reason separately**. It uses a small closure with `nonlocal` (Chapter 4) to drop and tally in one move:

```python
def drop(mask: pd.Series, reason: str) -> None:
    nonlocal out
    if n := int(mask.sum()):
        rejections[reason] = n
        out = out.loc[~mask]

drop(out["amount"].isna(), "amount missing or unparseable")
drop(out["quantity"].isna() | (out["quantity"] <= 0), "quantity not positive")
drop(~out["channel"].isin(config.valid_channels), "unknown channel")
```

- **`join_customers`** attaches customer attributes with a **left** join, so orders whose customer is unknown stay visible and get counted rather than silently vanishing as they would in an inner join. `validate="many_to_one"` makes pandas raise if the customer table has duplicate IDs, and a row-count check guards against the classic silent explosion:

```python
merged = orders.merge(customers, left_on="customer_id", right_on="id",
                      how="left", validate="many_to_one")
if len(merged) != len(orders):
    raise DataQualityError(f"join changed the row count: "
                           f"{len(orders)} → {len(merged)}")
```

- **`add_features`** derives `revenue`, `order_month` and `is_weekend`, and converts repeated strings to `category` to save memory.

## Step 5: orchestrate, behind a quality gate

`run_pipeline` composes the steps, logs each one, and, crucially, **refuses to publish** if too much of the input was rejected:

```python
if report.reject_rate > config.max_reject_rate:
    raise DataQualityError(
        f"reject rate {report.reject_rate:.1%} exceeds the "
        f"{config.max_reject_rate:.0%} threshold; refusing to publish")
```

A pipeline that publishes a half-empty table without complaint is worse than one that fails: the failure gets fixed, but the half-empty table gets *used*.

> [!STORY] Sixteen thousand cases that never arrived
> In early October 2020, Public Health England discovered that 15,841
> positive COVID-19 test results from the previous week had never been
> passed on for contact tracing. Lab results arrived as CSV files and were
> loaded into a spreadsheet template saved in the old `.xls` format, which
> holds at most 65,536 rows per sheet. When a file was bigger than that, the
> extra rows were simply cut off: no error, no warning, just a smaller table.
> Nothing in the process compared "rows read" with "rows written". A single
> count check, the kind of quality gate this chapter builds, would have
> raised the alarm on day one.

## Step 6: the readout

A pipeline nobody reads from isn't finished. The notebook builds monthly revenue per acquisition channel and a per-channel table of orders, revenue and average order value, then charts both. Each channel keeps the same colour in every panel, so the colour follows the entity, not its rank. A short summary block reports total revenue, average and median order value, the largest channel's share, and the reject rate.

> [!NOTE]
> The median order value sits well below the mean again: Chapter 1's
> right-skew, back for an encore. Reporting only the average order value
> would overstate what a typical order looks like.

## Step 7: test it

The pipeline is only trustworthy if its parts are. The tests cover the categories from Chapter 7: known answers for `parse_amount` (`"1.234,56"`, `"89,50"`, every null token), whitespace and casing for `clean_text_column`, and **invariants** such as "revenue equals amount × quantity" and "an empty run has a reject rate of zero, not a `ZeroDivisionError`".

## Step 8: read back what you wrote

The last check is to load the Parquet file exactly as a downstream consumer would and run a checklist against it: row count matches, dtypes preserved, no null revenue, every amount and quantity positive, every channel valid, no duplicate rows, and total revenue reconciling with amount × quantity. Building it, the course found a subtle trap here: text cleaned with pandas' nullable `string` dtype turned into categories that Parquet normalised back to plain `str` on read. The round trip was *almost* lossless, which is the worst kind: it passes a glance and fails an equality check. Normalising with `.astype(str).astype("category")` before writing fixes it.

## Step 9: prove the safety net fires

A safety net you've never seen catch anything isn't a safety net. The notebook deliberately runs the pipeline with `max_reject_rate=0.001`, far stricter than the real reject rate, and checks that `DataQualityError` is raised and that no output file was written.

## Your turn: extensions

Each of these is a real requirement you'd receive on the job.

| # | Extension | Difficulty |
| :--- | :--- | :--- |
| 1 | **Promote** `parse_amount` and `clean_text_column` into `src/dstoolkit/cleaning.py`, with tests, and import them back. | ★ |
| 2 | **Stream** the CSV in chunks so the pipeline handles files bigger than RAM; check totals are unchanged. | ★★ |
| 3 | **Rejected-rows report**: write discarded rows to `rejected_rows.parquet` with a `rejection_reason` column. | ★★ |
| 4 | **Idempotency**: two runs give byte-identical output; use `atomic_write` so a crash can't corrupt the table. | ★★ |
| 5 | A **`@monitored` decorator** that logs each step's name, duration and row counts in and out. | ★★★ |
| 6 | **Declarative schema validation**: one generic validator driven by a dict of column rules. Compare with `pandera`. | ★★★ |

## What you built

A pipeline that reads badly formatted sources defensively; separates a thin impure shell from a pure core; carries its configuration in a frozen dataclass so nothing drifts mid-run; counts and logs every row it drops, with the reason; refuses to publish output it can't vouch for; is tested where the real bugs live; and writes typed, compressed Parquet that reloads exactly as written.

That's a genuinely professional shape for a data pipeline, and every later module plugs into it. Module 03 replaces the transformations with richer pandas, Module 10 adds feature engineering, Module 21 adds experiment tracking, and Module 22 puts a model behind an API.

> [!KEY]
> - Write the **contract** first, as a frozen dataclass: named rules, one place to change them.
> - Keep I/O in a **thin shell**; make every transformation a **pure, testable function**.
> - **Count every dropped row**, with its reason; never let rows vanish silently.
> - Use **left joins + `validate=` + a row-count check** to make join bugs loud.
> - A **quality gate** that refuses to publish is a feature, not a failure.
> - **Read back what you wrote**, and **prove** your safety nets fire.
