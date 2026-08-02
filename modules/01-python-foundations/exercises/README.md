# Module 01 — Exercises

Every notebook in this module ends with an exercise section. This page collects
them in one place so you can work through them without scrolling, and points at
the worked solutions.

> [!WARNING]
> **Attempt each exercise before opening the solution.** Reading a worked answer
> produces a convincing feeling of understanding that vanishes the moment you
> face a blank cell. Ten minutes of being stuck teaches more than an hour of
> reading.

---

## Where the solutions are

All solutions live in a single executed notebook:

📓 **[`../notebooks/09_exercise_solutions.ipynb`](../notebooks/09_exercise_solutions.ipynb)**

Each solution includes the reasoning, and — where more than one answer is
defensible — the alternatives with their trade-offs.

---

## The exercise list

### Notebook 01 — Course Introduction & Setup

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 1.1 | Environment audit | Interpreter and package introspection |
| 1.2 | Reproduce the analysis with a different seed | Sampling variability |
| 1.3 | Break the notebook deliberately | Hidden state, out-of-order execution |
| 1.4 | Lifecycle mapping | Framing a real data product |

### Notebook 02 — Python Fundamentals

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 2.1 | Predict before you run | `+` vs `+=`, tuple immutability |
| 2.2 | A float comparison bug | IEEE 754, tolerance-based comparison |
| 2.3 | Safe enrichment | Non-mutating functions |
| 2.4 | Text cleaning | String normalisation idioms |
| 2.5 | Missing-value audit | `nan != nan` |

### Notebook 03 — Data Structures

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 3.1 | Pick the structure | Complexity-driven choice |
| 3.2 | Fix the slow function | O(n×m) → O(n+m) with a set |
| 3.3 | Comprehension practice | list / dict / set comprehensions |
| 3.4 | Group without pandas | `defaultdict` |
| 3.5 | Multi-key sorting | Key tuples, mixed sort directions |

### Notebook 04 — Functions & Functional Programming

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 4.1 | Fix the signature | Keyword-only args, type hints, docstrings |
| 4.2 | Spot the trap | Mutable default arguments |
| 4.3 | Write a decorator | `functools.wraps`, argument validation |
| 4.4 | Generator pipeline | Lazy evaluation, constant memory |
| 4.5 | Closure counter | `nonlocal` |

### Notebook 05 — Object-Oriented Programming

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 5.1 | Class or function? | Design judgement |
| 5.2 | Dunder practice | `__sub__`, `__mul__`, `__format__` |
| 5.3 | Dataclass conversion | `@dataclass`, `__post_init__` |
| 5.4 | Build an estimator | The scikit-learn `fit`/`transform` contract |
| 5.5 | Find the bug | Shared mutable class attributes |

### Notebook 06 — Files & Data Formats

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 6.1 | Path manipulation | `pathlib` |
| 6.2 | Diagnose a broken CSV | Separators, encodings, decimals, dates |
| 6.3 | Nested JSON | `pd.json_normalize` |
| 6.4 | Prove the column-pruning claim | Columnar vs row storage |
| 6.5 | Safe writing | Context managers, atomic writes |

### Notebook 07 — Errors, Logging & Testing

| # | Exercise | Skill |
| :-- | :--- | :--- |
| 7.1 | Narrow the catch | Specific exception handling |
| 7.2 | Custom exception hierarchy | Domain errors, retryability |
| 7.3 | Instrument a function | `logging` levels |
| 7.4 | Write the tests first | Test-driven design, edge cases |
| 7.5 | Catch a real bug | Merge row explosion, `validate=` |

### Notebook 08 — Module Project

The project ends with six graded extensions (⭐ to ⭐⭐⭐). They are open-ended by
design and have no single right answer — they are the closest thing in this
module to real work.

| # | Extension | Difficulty |
| :-- | :--- | :-- |
| 1 | Promote code into `dstoolkit` | ⭐ |
| 2 | Stream the input in chunks | ⭐⭐ |
| 3 | A rejected-rows report | ⭐⭐ |
| 4 | Idempotency + atomic writes | ⭐⭐ |
| 5 | A `@monitored` decorator | ⭐⭐⭐ |
| 6 | Declarative schema validation | ⭐⭐⭐ |

---

## How to check your own work

Beyond comparing with the solutions, two habits generalise:

1. **Write a test.** If you cannot state what the correct output is, you do not
   yet understand the problem.
2. **Restart & Run All.** If your solution only works when cells run in the order
   you happened to click them, it does not work.
