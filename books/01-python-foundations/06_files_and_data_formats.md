# Files Without Fear
@subtitle Paths, context managers, and choosing a data format on evidence
@time About 75 minutes
@notebook 06_files_and_data_formats.ipynb
@learn
- `pathlib`, the modern way to handle paths, and why string concatenation breaks.
- Context managers (`with`), and how to write your own.
- CSV: the format everyone uses and nobody reads the spec for.
- JSON: nested data, and the types it silently can't represent.
- **Parquet**: columnar storage, and why it's smaller and faster than CSV.
- A measured comparison, so you can choose a format on evidence.
@end

@lead Data arrives as files, and files are where a surprising number of bugs are born: a path that only works on one laptop, a write that dies halfway, a date that quietly became a string. None of this is glamorous. All of it decides whether your numbers are right.

## `pathlib`: stop building paths with strings

Gluing paths together with `+` is fragile. It breaks across operating systems, doubled slashes creep in, and nothing checks your work. `pathlib.Path` treats a path as a structured object, and the `/` operator joins pieces with the right separator for whatever OS you're on:

```python
from pathlib import Path

REPO_ROOT = find_repo_root()          # see below
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)   # safe to run again and again
```

`parents=True` creates missing intermediate folders; `exist_ok=True` makes the call idempotent, so re-running the notebook is harmless.

### Find the project root, don't hard-code it

Counting `.parent` calls breaks the moment a notebook moves one level. Searching upwards for a file that only exists at the root doesn't:

```python
def find_repo_root(marker: str = "pyproject.toml") -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"no {marker} found above {Path.cwd()}")
```

### Paths come apart into named pieces

```python
sample = RAW_DIR / "customers_2024_q1.csv"
sample.name                      # 'customers_2024_q1.csv'
sample.stem                      # 'customers_2024_q1'
sample.suffix                    # '.csv'
sample.parent                    # .../data/raw
sample.with_suffix(".parquet")   # same place, new extension
sorted(RAW_DIR.glob("*.csv"))    # one level; .rglob() recurses
```

> [!TIP]
> Define your paths **once**, near the top of a notebook, relative to a
> project root you compute. An absolute path like
> `/Users/ana/Desktop/project/data.csv` is the single most common reason a
> colleague can't run your notebook.

## Context managers

A file must be closed. Leaving files open leaks operating-system handles and, worse for data work, buffered writes may never reach the disk. The `with` statement guarantees cleanup **even if an exception is raised** inside the block:

```python
with open(path, "a", encoding="utf-8") as f:
    f.write("second line\n")
# f is closed here, error or no error

path.write_text("overwritten\n", encoding="utf-8")   # pathlib shortcut
path.read_text(encoding="utf-8")
```

### Writing your own: an atomic write

Any setup-and-teardown pair is a candidate for a context manager. `@contextmanager` turns a generator into one: everything before `yield` is setup; everything after is guaranteed teardown.

Here's a genuinely useful one. Scheduled data jobs crash, and a half-written output file that replaces a good one silently poisons everything downstream. The fix: write to a temporary file, and swap it into place **only on success**:

```python
from contextlib import contextmanager

@contextmanager
def atomic_write(path: Path, encoding: str = "utf-8"):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    handle = temp_path.open("w", encoding=encoding)
    try:
        yield handle                       # the body of the `with` runs here
    except Exception:
        handle.close()
        temp_path.unlink(missing_ok=True)  # throw the partial file away
        raise                              # and let the caller see the error
    else:
        handle.close()
        temp_path.replace(path)            # atomic on POSIX: old or new, never half
```

Crash inside the `with` and the original file is untouched, with no `.tmp` left behind.

## A dataset worth measuring

To compare formats meaningfully, the notebook builds 200,000 synthetic orders with a realistic mix of columns: integer IDs, dates, low-cardinality strings (`channel`, `country`), money amounts, quantities and a boolean.

> [!TIP]
> Always check `df.dtypes` right after building or loading a DataFrame. A
> numeric column that arrived as `object` is the most common cause of a slow,
> wrong analysis, and it's invisible in `.head()`.

## CSV: universal, and full of traps

CSV is a *convention*, not a strict standard. Everybody supports it, which is its only real advantage.

> [!STORY] A format nobody wrote down for thirty years
> Comma-separated values were in use on mainframes by the early 1970s. Yet
> the first serious attempt to document the format, RFC 4180, only appeared
> in **2005**, and even that is an "informational" memo describing common
> practice, not a binding standard. That's why every tool has its own
> opinions about quoting, line endings, separators and decimal marks, and why
> "it's just a CSV" is famous last words.

Write with `index=False`. Otherwise pandas writes the row index as an unnamed first column, which comes back as `Unnamed: 0` the next time anyone reads the file.

### What CSV loses

A CSV file is *just text*. Every type annotation is thrown away on write, and pandas has to guess them all back on read. Round-trip the orders table and `order_date` comes back as a plain string, so date arithmetic fails or, worse, text sorting puts `2024-10-01` before `2024-2-01`.

> [!STORY] When spreadsheets renamed human genes
> Type-guessing isn't just a pandas problem. Excel, helpfully, converts text
> that *looks* like a date into a date. Gene symbols such as SEPT2 (Septin 2)
> and MARCH1 became "2-Sep" and "1-Mar". In 2016 a study by Ziemann, Eren and
> El-Osta found the problem in roughly one in five genomics papers that
> shipped gene lists as Excel supplements. The fix eventually came from the
> other side: in 2020 the HUGO Gene Nomenclature Committee **renamed** the
> affected genes (SEPT1 became SEPTIN1, MARCH1 became MARCHF1) so that
> spreadsheets would leave them alone. When software guesses your types,
> sooner or later it guesses wrong.

The fix is to **tell** pandas what you expect instead of letting it guess:

```python
reloaded = pd.read_csv(
    csv_path,
    parse_dates=["order_date"],
    dtype={
        "order_id": "int32",
        "customer_id": "int32",
        "channel": "category",     # repeated strings → big memory savings
        "country": "category",
        "quantity": "int8",
    },
)
```

An explicit dtype map is both faster and safer than inference, and `category` cuts memory dramatically for columns with few distinct values.

### The other CSV traps

| Trap | Symptom | Fix |
| :--- | :--- | :--- |
| Wrong encoding | `UnicodeDecodeError` | `encoding="latin-1"` or `"cp1252"` |
| European decimals | `1.234,56` read as text | `decimal=","`, `sep=";"` |
| Leading zeros | `00123` becomes `123` | `dtype={"zip": str}` |
| Huge numeric IDs | precision lost above 2⁵³ | `dtype={"id": str}` |
| `"NA"`, `"N/A"`, `"-"` | not recognised as missing | `na_values=["NA", "-"]` |
| File bigger than RAM | `MemoryError` | `chunksize=100_000` |

That last row is Chapter 4's generator idea in CSV form: read a chunk, process it, let it go.

```python
total = 0.0
for chunk in pd.read_csv(csv_path, chunksize=50_000, usecols=["amount"]):
    total += chunk["amount"].sum()
```

## JSON: nested data

JSON handles hierarchy, which CSV can't, and it's the format of almost every web API you'll pull data from. Write it with `indent=2` (readable, diffable) and `ensure_ascii=False` (accented characters stay as themselves).

### Flattening for analysis

`pd.json_normalize` turns nested structures into a flat table: the bridge between an API response and a DataFrame.

```python
flat = pd.json_normalize(
    api_response,
    record_path="results",                    # the list that becomes rows
    meta=["request_id", ["meta", "page"]],    # fields copied onto every row
)
```

> [!WARNING]
> **JSON has no date type and no NaN.** Dates become strings, and `NaN` and
> `Infinity` aren't valid JSON at all. Python's `json` module writes them
> anyway, producing files that stricter parsers in other languages reject.
> Pass `allow_nan=False` to catch it, and write `null` deliberately instead.

> [!NOTE]
> JSON was popularised in the early 2000s by Douglas Crockford, who liked to
> say he *discovered* it rather than invented it: it's simply the syntax
> JavaScript already used for literal objects. It became formal standards
> later (ECMA-404 in 2013).

## Parquet: the format for data science

Parquet stores data **column by column** instead of row by row, with per-column compression and the schema embedded in the file.

![Row storage keeps each record together. Column storage keeps each field together, all of one type.](assets/images/book/m01_row_vs_column.png){width=100}

Three consequences follow directly from that layout:

1. **Compression is far better.** A column holds one type, often with few distinct values.
2. **Reading a subset of columns is nearly free.** The other columns are never touched. This is called *column pruning*.
3. **The schema travels with the data.** Dtypes survive the round trip exactly: `orders.equals(pd.read_parquet(path))` is `True`.

```python
orders.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
pd.read_parquet(path, columns=["channel", "amount"])    # reads only two columns
```

> [!STORY] From Google's Dremel to your laptop
> In 2010 Google published a paper on **Dremel**, the system it used to run
> interactive queries over trillions of rows. One of its key ideas was storing
> even nested records in a columnar layout. Engineers at Twitter and Cloudera
> took that idea and in 2013 released **Apache Parquet** as an open format.
> The trick that lets a data warehouse scan only the columns a query needs
> is the same one that makes your `read_parquet(columns=[...])` fast.

## A measured comparison

Enough assertion. Here's the same 200,000-row table in three formats, measured on the machine that built this book:

![Size on disk and time to read back the same table. Your exact numbers will differ; the ranking won't.](assets/images/book/m01_format_benchmark.png){width=100}

## Choosing a format

| Format | Use it when | Avoid it when |
| :--- | :--- | :--- |
| **CSV** | A human must open it; exchanging with a non-technical team | Data is large, typed or nested |
| **JSON** | Nested data; API payloads; configuration | Tabular data at any real scale |
| **Parquet** | **Any analytical dataset you control**: the default | Someone needs to read it in a text editor |
| **Pickle** | Short-lived, Python-only caching | *Anything shared or long-lived* |
| **Feather** | Very fast temporary hand-off between pandas/R sessions | Long-term archives |

> [!WARNING]
> **Never load a pickle from an untrusted source.** Unpickling can execute
> arbitrary code by design, so a malicious `.pkl` file is simply a program
> you agreed to run. Pickles are also fragile across library versions.

### The convention this course follows

```text
data/raw/         ← exactly as received, never modified (CSV, JSON, whatever)
data/processed/   ← cleaned, typed, analysis-ready → always Parquet
```

A small method chain shows the convention in action: read raw, fix types, derive a column, filter, and write processed Parquet.

```python
clean = (
    pd.read_csv(csv_path, parse_dates=["order_date"])
    .assign(
        channel=lambda df: df["channel"].astype("category"),
        revenue=lambda df: (df["amount"] * df["quantity"]).round(2),
    )
    .loc[lambda df: df["amount"] > 0]
)
clean.to_parquet(PROCESSED_DIR / "orders_clean.parquet", index=False)
```

## Try it yourself

Exercises 6.1 to 6.5 in notebook 06; solutions in notebook 09.

1. **Path manipulation.** From `Path("data/raw/sales_2024_q1.csv")`, using only `pathlib`, produce `data/processed/sales_2024_q1.parquet`, the year and quarter as strings, and a sibling `sales_2024_q1_backup.csv`.
2. **Diagnose a broken CSV.** Read a semicolon-separated file with `001;Café;1.234,56;15/03/2024` rows so that `id` keeps its leading zeros, `price` is a float and `date` is a datetime.
3. **Nested JSON.** Flatten an order with a nested `customer` and a list of `items` into one row per item, carrying the order fields onto each row.
4. **Prove column pruning.** Build a 50-column, 500,000-row DataFrame. Time reading **one** column from Parquet and from CSV, and explain the ratio using the storage layout.
5. **Safe writing.** Extend `atomic_write` with `backup: bool`, keeping the previous version as `<name>.bak`.

> [!KEY]
> - Use **`pathlib`**, never string concatenation. Compute a project root; never hard-code absolute paths.
> - **Always use `with`** for files: it closes them even when an exception fires.
> - **CSV loses every dtype.** Pass `parse_dates` and an explicit `dtype` map; write with `index=False`.
> - **JSON has no date and no NaN.** Convert deliberately before writing.
> - **Parquet is the default for analytical data**: smaller, faster, self-describing, and it supports column pruning.
> - **Never unpickle untrusted data.**
> - Convention: `data/raw/` is immutable input; `data/processed/` is Parquet output.
