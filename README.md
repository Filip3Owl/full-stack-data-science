# Full-Stack Data Science with Python

A complete, hands-on course that takes you from the Python language itself all the
way to **machine learning models running in production**. Every concept is taught
with real, executable code in Jupyter notebooks: no pseudo-code, no toy snippets
that break when you run them.

> **Who this is for:** anyone aiming for a Data Scientist / ML Engineer role at a
> mid-to-large company. The curriculum mirrors the actual stack those teams use
> day to day.

---

## Table of Contents

- [Philosophy](#philosophy)
- [The Data Science Stack](#the-data-science-stack)
- [Curriculum](#curriculum)
- [Web Version](#web-version)
- [Getting Started](#getting-started)
- [Repository Layout](#repository-layout)
- [How to Study](#how-to-study)
- [Conventions](#conventions)

---

## Philosophy

Three rules shape every notebook in this repository:

| Rule | What it means in practice |
| :--- | :--- |
| **Run it, don't read it** | Every notebook is committed *with its outputs already executed*, so you can read it like a book, but you learn by re-running and breaking it. |
| **Explain the "why"** | Code comments explain intent, not syntax. Markdown cells explain the statistical or engineering reasoning behind the choice. |
| **Production-shaped** | We favour the patterns real teams use (pipelines, type hints, tests, reproducible seeds) over shortcuts that only work in a demo. |

---

## The Data Science Stack

A data scientist at a large company is expected to be fluent across five layers.
This course covers all of them:

```
┌───────────────────────────────────────────────────────────────────┐
│  5. COMMUNICATION      storytelling · dashboards · stakeholder     │
│                        readouts · experiment write-ups             │
├───────────────────────────────────────────────────────────────────┤
│  4. PRODUCTION         MLflow · FastAPI · Docker · CI/CD ·         │
│     (MLOps)            monitoring · drift · model registry         │
├───────────────────────────────────────────────────────────────────┤
│  3. MODELLING          scikit-learn · XGBoost · PyTorch ·          │
│                        statsmodels · causal inference              │
├───────────────────────────────────────────────────────────────────┤
│  2. ANALYSIS           NumPy · pandas · matplotlib · seaborn ·     │
│                        probability · hypothesis testing            │
├───────────────────────────────────────────────────────────────────┤
│  1. FOUNDATIONS        Python · data structures · OOP · SQL ·      │
│                        Git · testing · virtual environments        │
└───────────────────────────────────────────────────────────────────┘
```

---

## Curriculum

### Part I: Foundations

| # | Module | Key topics | Status |
| :-: | :--- | :--- | :-: |
| **01** | [**Python Foundations for Data Science**](modules/01-python-foundations/) | Environment & tooling, types, control flow, data structures, comprehensions, functions, generators, decorators, OOP, file I/O, errors, logging, testing, type hints | ✅ **Ready** |
| **02** | [**NumPy & Numerical Computing**](modules/02-numpy-numerical-computing/) | ndarrays, views vs copies, dtypes & precision, broadcasting, aggregations & masking, linear algebra, random generators, strides & memory | ✅ **Ready** |
| 03 | pandas & Data Wrangling | Series/DataFrame, indexing, joins, group-by, reshaping, time indexes, method chaining | 🔜 |
| 04 | Data Visualization | matplotlib grammar, seaborn statistical plots, plotly interactivity, chart selection, accessibility | 🔜 |
| 05 | SQL & Databases | SELECT → window functions, CTEs, query planning, SQLAlchemy, DuckDB, warehouse patterns | 🔜 |

### Part II: Statistics & Inference

| # | Module | Key topics | Status |
| :-: | :--- | :--- | :-: |
| 06 | Descriptive Statistics & EDA | Central tendency, dispersion, skew/kurtosis, correlation, outliers, EDA workflow | 🔜 |
| 07 | Probability & Distributions | Random variables, discrete/continuous distributions, CLT, law of large numbers, simulation | 🔜 |
| 08 | Inferential Statistics | Estimation, confidence intervals, hypothesis tests, power, multiple comparisons, bootstrap |🔜 |
| 09 | Experimentation & A/B Testing | Experiment design, sample sizing, sequential testing, CUPED, causal inference basics | 🔜 |

### Part III: Machine Learning

| # | Module | Key topics | Status |
| :-: | :--- | :--- | :-: |
| 10 | Data Cleaning & Feature Engineering | Missing data, encoding, scaling, leakage, pipelines, ColumnTransformer | 🔜 |
| 11 | Supervised Learning: Regression | Linear/regularised models, assumptions, residual analysis, tree regressors | 🔜 |
| 12 | Supervised Learning: Classification | Logistic regression, SVM, kNN, naive Bayes, class imbalance, thresholds | 🔜 |
| 13 | Model Evaluation & Tuning | Cross-validation, metrics, ROC/PR, calibration, grid/random/Bayesian search | 🔜 |
| 14 | Ensembles & Gradient Boosting | Bagging, random forests, XGBoost, LightGBM, CatBoost, stacking | 🔜 |
| 15 | Unsupervised Learning | k-means, hierarchical, DBSCAN, PCA, t-SNE/UMAP, anomaly detection | 🔜 |
| 16 | Time Series & Forecasting | Decomposition, stationarity, ARIMA, SARIMAX, Prophet, backtesting | 🔜 |

### Part IV: Deep Learning & Unstructured Data

| # | Module | Key topics | Status |
| :-: | :--- | :--- | :-: |
| 17 | Deep Learning Fundamentals | Tensors, autograd, MLPs, optimisers, regularisation, training loops in PyTorch | 🔜 |
| 18 | Natural Language Processing | Text preprocessing, TF-IDF, embeddings, transformers, fine-tuning, LLM usage patterns | 🔜 |
| 19 | Computer Vision | CNNs, augmentation, transfer learning, segmentation basics | 🔜 |

### Part V: Scale & Production

| # | Module | Key topics | Status |
| :-: | :--- | :--- | :-: |
| 20 | Big Data Processing | PySpark DataFrames, lazy evaluation, partitioning, Dask, DuckDB at scale | 🔜 |
| 21 | MLOps I: Reproducibility | Project structure, config, experiment tracking with MLflow, model registry, DVC concepts | 🔜 |
| 22 | MLOps II: Deployment | FastAPI serving, Docker, batch vs. online inference, CI/CD, cloud deployment | 🔜 |
| 23 | Monitoring & Explainability | Data/concept drift, performance monitoring, SHAP, fairness, model governance | 🔜 |
| 24 | Capstone & Communication | End-to-end project, storytelling, executive summaries, portfolio & interview prep | 🔜 |

---

## Web Version

The whole course can be read in a browser: lessons with their executed outputs,
search, dark mode, and progress saved in your browser. The site is generated
from the notebooks and READMEs, so there is nothing extra to keep in sync.

```bash
uv pip install --python .venv/bin/python -e ".[web]"
.venv/bin/python scripts/build_site.py --serve     # http://127.0.0.1:8000
```

Code cells are read-only on the site. To run them, use JupyterLab as described below.

---

## Getting Started

### Prerequisites

- **Python 3.12** (the course pins this version for reproducibility)
- **git**
- ~5 GB of free disk space if you install the deep-learning extras

### 1. Clone and enter the project

```bash
git clone <your-fork-url> full-stack-data-science
cd full-stack-data-science
```

### 2. Create the virtual environment

The project ships with a `.venv` created by [uv](https://docs.astral.sh/uv/), a
fast Python package manager. Pick whichever tool you have:

<details>
<summary><b>Option A: uv (recommended, fastest)</b></summary>

```bash
# Install uv once (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install the core stack
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```
</details>

<details>
<summary><b>Option B: standard library venv</b></summary>

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```
</details>

### 3. Register the Jupyter kernel

This makes the environment selectable from inside JupyterLab:

```bash
.venv/bin/python -m ipykernel install --user \
    --name dsc-py312 \
    --display-name "Python 3.12 (Data Science Course)"
```

### 4. Launch JupyterLab

```bash
.venv/bin/jupyter lab
```

Then open `modules/01-python-foundations/notebooks/` and start with notebook `01`.

> **Select the right kernel.** In the top-right of each notebook, make sure the
> kernel reads *Python 3.12 (Data Science Course)*. If it says anything else,
> click it and switch; otherwise imports will fail.

### 5. (Optional) Install extras for later modules

```bash
uv pip install --python .venv/bin/python -e ".[advanced-ml]"     # Modules 14-16
uv pip install --python .venv/bin/python -e ".[deep-learning]"   # Modules 17-19
uv pip install --python .venv/bin/python -e ".[big-data]"        # Module 20
uv pip install --python .venv/bin/python -e ".[mlops]"           # Modules 05, 21-23
```

### Verify your setup

```bash
.venv/bin/python scripts/check_environment.py
```

You should see a green check for every core package.

---

## Repository Layout

```
full-stack-data-science/
├── README.md                   ← you are here
├── pyproject.toml              ← dependencies & tool configuration
├── requirements-lock.txt       ← exact pinned versions (reproducibility)
├── .gitignore
│
├── assets/
│   └── images/                 ← generated diagrams used across notebooks
│
├── data/
│   ├── raw/                    ← immutable inputs (never edited by hand)
│   ├── processed/              ← cleaned outputs produced by notebooks
│   └── external/               ← third-party downloads
│
├── modules/
│   ├── 01-python-foundations/
│   │   ├── README.md           ← module syllabus, objectives, checklist
│   │   ├── notebooks/          ← the lessons (.ipynb, executed)
│   │   ├── scripts/            ← plain-text .py mirrors of each notebook
│   │   └── exercises/          ← practice problems + solutions
│   └── 02-numpy-numerical-computing/
│       └── …                   ← same layout, one directory per module
│
├── web/                        ← templates and styles of the web version
├── src/dstoolkit/              ← reusable helper package importable everywhere
├── scripts/                    ← project-level utilities
└── tests/                      ← pytest suite for src/
```

**Why `src/dstoolkit/`?** Real projects do not keep logic locked inside
notebooks. As the course progresses, code that proves useful gets promoted from
a notebook into this package, gets a test, and is then imported back. You will
practise that promotion loop from Module 01 onward.

---

## How to Study

Each module follows the same rhythm:

1. **Read the module `README.md`**: learning objectives and time estimate.
2. **Work through the notebooks in order**: they build on each other.
3. **Run every cell yourself.** Then change a value and predict the output
   *before* re-running it. This is where the learning actually happens.
4. **Do the exercises** in `exercises/` before opening the solutions.
5. **Complete the module project**: an integrated problem using everything covered.

**Suggested pace:** one module per week at ~6-8 hours/week. The full course is
roughly a 6-month part-time commitment.

---

## Conventions

These hold across every notebook so you always know what you are looking at.

### Callout blocks

> [!NOTE]
> Extra context worth knowing, but not required to move on.

> [!TIP]
> A shortcut, idiom, or habit used by professionals.

> [!WARNING]
> A common mistake that produces silently wrong results.

### Code style

- **PEP 8** enforced by `ruff`, formatted with `black` (88-char lines).
- **Type hints** on every function that lives in `src/`.
- **Docstrings** in NumPy style, the standard across the scientific Python ecosystem.
- **Seeded randomness**: every stochastic cell sets a seed so your output matches ours.

### Running quality checks

```bash
.venv/bin/ruff check .        # lint
.venv/bin/black --check .     # formatting
.venv/bin/pytest              # tests for src/
```

---

## License

MIT: use it, fork it, teach with it.
