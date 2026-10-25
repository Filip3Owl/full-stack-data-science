# Welcome to the Job
@subtitle What data scientists really do, and the workshop you'll need
@time About 60 minutes
@notebook 01_course_introduction_and_setup.ipynb
@learn
- What a data scientist actually spends the week doing, and which parts this course covers.
- The data science lifecycle, and why it is a loop rather than a pipeline.
- How to build a **reproducible** Python environment, and why that is a professional duty, not a nicety.
- How to work in Jupyter without falling into its two classic traps.
- A first end-to-end taste: data in, insight out, honest conclusion.
@end

@lead Every field has a story it likes to tell about itself. Data science's favourite story is about clever models. The reality is messier, more interesting and, frankly, more useful to know on day one: most of the job is making sure the numbers deserve to be believed.

This first chapter is the map. We'll look at what the work really looks like, set up a workshop that won't betray you, and then run a miniature analysis from start to finish so you can feel the shape of the whole thing before we zoom into the parts.

## What does a data scientist actually do?

Job titles vary wildly, but in a mid-sized or large company the week almost always breaks down into four recurring activities.

| Activity | What it looks like | Share of time |
| :--- | :--- | :--- |
| **Framing** | Turning "can we reduce churn?" into a measurable question with a defined success metric | ~15% |
| **Data work** | Finding, joining, cleaning and validating data; discovering a column changed meaning in March | ~50% |
| **Modelling & analysis** | Statistics, experiments, machine learning: the part that gets taught | ~20% |
| **Delivery** | Shipping the model, writing the readout, convincing a stakeholder, monitoring what you shipped | ~15% |

![Where the week goes. Half of it is data work, and that is not a sign of a badly run team. It *is* the job.](assets/images/book/m01_time_split.png){width=100}

That 50% is the headline. A course that only teaches modelling prepares you for a fifth of your working life. This one tries to cover all four columns.

> [!STORY] The day a job title was invented
> Around 2008, two people building data teams at fast-growing web companies,
> DJ Patil at LinkedIn and Jeff Hammerbacher at Facebook, needed a name for
> the people they were hiring: part statistician, part engineer, part detective.
> "Data scientist" stuck. By 2012 the *Harvard Business Review* was calling it
> "the sexiest job of the 21st century" (Davenport & Patil). The skills were
> old; the bundle was new, and the bundle is what this course is about.

### Three roles, blurry borders

You'll hear three titles, and their responsibilities overlap more every year:

- **Data Analyst** describes what *happened*: SQL, dashboards, descriptive statistics.
- **Data Scientist** explains *why* it happened and predicts what *will* happen: statistics, experiments, machine learning.
- **ML Engineer** makes the prediction *run reliably* for millions of users: pipelines, serving, monitoring.

The market increasingly expects a data scientist to be credible in all three columns. That is exactly what "full stack" means in this course's title.

## The lifecycle is a loop

Nearly every project travels through the same eight stages. The single most important thing to internalise is that **the arrows go backwards too**.

![The data science lifecycle. Findings at any stage can send you back to an earlier one.](assets/images/ds_lifecycle.png){width=72}

During modelling you'll discover that a feature leaks information from the future, and you'll go back to feature engineering. You'll present results and learn the business question was subtly different, and you'll go back to the very first box. That is not a sign you planned badly. It is the process working.

| Stage | The question you're asking | Where the course covers it |
| :--- | :--- | :--- |
| 1. Business understanding | "What decision will this change?" | Modules 09, 24 |
| 2. Data collection | "Where does this live, and can I trust it?" | Modules 05, 20 |
| 3. Cleaning & preparation | "What do the missing values *mean*?" | Modules 03, 10 |
| 4. Exploratory analysis | "What does the data actually look like?" | Modules 04, 06 |
| 5. Feature engineering | "What representation makes the signal learnable?" | Module 10 |
| 6. Modelling | "Which model, with which hyperparameters?" | Modules 11–19 |
| 7. Evaluation | "Is it better than what we do today? By how much? How sure am I?" | Modules 08, 13 |
| 8. Deployment & monitoring | "Is it still working three months later?" | Modules 21–23 |

> [!STORY] A map, a pump handle, and the whole loop in 1854
> In the summer of 1854 cholera tore through Soho in London, killing hundreds
> of people in a few streets within days. The physician John Snow did not
> believe the popular theory that the disease spread through bad air. He
> **framed** a sharper question (is it the water?), **collected** data by
> walking door to door recording deaths, **explored** it by plotting them on a
> street map, and saw them cluster around one public pump on Broad Street.
> He **delivered** his case to the local council, which removed the pump's
> handle on 8 September.
>
> Then came the part people forget. The outbreak was already waning when the
> handle came off, so the intervention alone could not prove his point. Snow
> went *back around the loop*, comparing death rates between households
> supplied by two different water companies, and built the stronger argument
> that eventually changed public health. Question, data, exploration,
> evaluation, back again: the lifecycle is older than computers.

## Reproducibility starts with the environment

### One project, one environment

Imagine two projects on your laptop. Project A was built against an old version of pandas; project B needs a newer one where several behaviours changed. With a single system-wide Python you can only keep one of them happy.

A **virtual environment** is the fix: an isolated folder with its own Python interpreter and its own installed packages. One per project. They are cheap to create, disposable, and they eliminate a whole category of "works on my machine" bugs.

```text
full-stack-data-science/
└── .venv/                     ← this project's private environment
    ├── bin/python             ← its own interpreter
    └── lib/python3.12/site-packages/
                               ← its own packages, invisible to others
```

> [!WARNING]
> Never `pip install` into your operating system's Python. On macOS and Linux
> other system tools depend on it, and you can break them.

### Three levels of "I wrote down my dependencies"

Professional teams distinguish three levels, and you should too:

![The three levels of reproducibility. Humans edit level 2; machines generate level 3; production deploys from level 3.](assets/images/book/m01_reproducibility.png){width=100}

- **Loose** (`pandas`) means "some version of pandas". It reproduces almost nothing.
- **Constrained** (`pandas>=2.2,<3`) reproduces the *API*. This is what `pyproject.toml` declares.
- **Pinned** (`pandas==3.0.5`) reproduces the *exact bits*. This is what `requirements-lock.txt` captures.

### Check what your notebook is really running

The ground truth is not what your terminal says. It is what the *kernel* says:

```python
import sys
from pathlib import Path

print(sys.executable)                 # the interpreter running THIS notebook
in_venv = sys.prefix != sys.base_prefix
print(f"In a venv? {in_venv}")
```

When `sys.prefix` and `sys.base_prefix` differ, you are inside a virtual environment. If the path points somewhere like `/usr/bin/python3`, your notebook is attached to the wrong kernel. Fix it from the kernel picker before going any further, or nothing later in the course will import.

### The course's own helper package

The repository ships a small package, `dstoolkit`, in `src/`. It is installed in **editable mode** (`pip install -e .`), so the import picks up your edits immediately, with no reinstall and no fragile `sys.path` hacks. One call to `setup_plot_style()` then gives every chart in every notebook the same fonts, colours and spacing.

```python
from dstoolkit import setup_plot_style
setup_plot_style()
```

> [!STORY] A spreadsheet that shaped economic policy
> In 2010 two Harvard economists, Carmen Reinhart and Kenneth Rogoff,
> published a paper suggesting that economic growth falls sharply once public
> debt passes 90% of GDP. It was quoted widely in debates about austerity.
> In 2013 Thomas Herndon, a graduate student at the University of
> Massachusetts Amherst, tried to reproduce the result for a class
> assignment and couldn't. When the authors shared their spreadsheet, he
> found that a formula's cell range stopped short and quietly left five
> countries out of one average, alongside other debatable choices.
>
> Correcting it weakened the headline finding considerably. The lesson for
> us isn't "spreadsheets are evil". It is that **analysis nobody else can
> re-run is analysis nobody can check**, including you, six months from now.

## Jupyter without the foot-guns

A notebook is **not** a script. It is a long-running Python process (the *kernel*) plus a document of cells that you can send to it **in any order**. That freedom is what makes notebooks wonderful for exploration, and it is also the source of both classic notebook bugs.

### Trap 1: hidden state

The little counter `[n]` beside each cell tells you the order cells *actually ran*, which may have nothing to do with their order on the page.

![Three cells, run out of order. The printed row count describes a DataFrame that no longer exists.](assets/images/book/m01_hidden_state.png){width=100}

**The fix** is a habit worth building today: before you trust a result or share a notebook, run **Kernel → Restart Kernel and Run All Cells**. If it fails, your notebook was lying to you.

### Trap 2: notebooks are not a substitute for functions

A 200-line cell can't be tested, reused or reviewed. The professional pattern is a **promotion loop**:

![The promotion loop: working code graduates from a notebook cell into a tested package, and comes back as an import.](assets/images/book/m01_promotion_loop.png){width=100}

You'll practise this loop for real in this module's project.

### The magic commands worth knowing

Jupyter adds "magics": commands prefixed with `%` (one line) or `%%` (the whole cell).

| Magic | What it does |
| :--- | :--- |
| `%timeit` | Runs a statement many times and reports a robust average. Use it to choose between two implementations. |
| `%%time` | Times the whole cell exactly once. For slow, one-off work. Must be the first line of the cell. |
| `%who` / `%whos` | Lists the variables in the kernel, a quick way to spot hidden state. |
| `%debug` | Drops you into a post-mortem debugger after an exception. |
| `%load_ext autoreload` + `%autoreload 2` | Re-imports your `src/` modules automatically after you edit them. |
| `!command` | Runs a shell command, e.g. `!ls data/raw`. |

> [!TIP]
> Put `%load_ext autoreload` and `%autoreload 2` at the top of any notebook
> where you're actively editing code in `src/`. Without them you have to
> restart the kernel to see your changes.

## A first taste of the whole loop

Enough setup. Let's run the lifecycle in miniature. The scenario comes up in essentially every consumer company: **do customers acquired through paid ads spend less than customers who found us organically?**

We simulate the data so the analysis is self-contained and, thanks to a fixed seed, you'll see exactly the numbers the notebook shows.

```python
rng = np.random.default_rng(seed=42)
N_CUSTOMERS = 2_000

channel = rng.choice(["organic", "paid"], size=N_CUSTOMERS, p=[0.55, 0.45])
spend = np.where(
    channel == "organic",
    rng.lognormal(mean=4.30, sigma=0.55, size=N_CUSTOMERS),
    rng.lognormal(mean=4.05, sigma=0.60, size=N_CUSTOMERS),
)
customers = pd.DataFrame({"customer_id": np.arange(1, N_CUSTOMERS + 1),
                          "channel": channel, "spend_usd": spend.round(2)})
```

Why a *lognormal* distribution? Because real customer spend is right-skewed: most people spend a little, and a few spend a lot. Module 07 explains where that shape comes from.

### Look before you leap

The first question about any dataset is always the same: *what's in it, and is any of it obviously broken?* Count rows, count missing values, then summarise by group:

```python
summary = customers.groupby("channel")["spend_usd"].agg(
    n="count", mean="mean", median="median", std="std",
)
```

> [!WARNING]
> In both groups the **mean sits well above the median**. That is the
> fingerprint of a right-skewed distribution, and it means the average alone
> paints a misleading picture of a "typical" customer. Reporting only the
> mean here is one of the most common analytical mistakes in industry.

A pair of charts (overlapping histograms for the shape, box plots for a like-for-like comparison) shows paid customers sitting a little lower. The notebook draws both.

### Is the difference real?

A professional's next question is immediate: *could this gap be noise?* That question, the difference between an observation and a **defensible claim**, is the subject of Modules 07 and 08. Here is a preview using Welch's t-test, which compares two means without assuming the groups share a variance:

```python
from scipy import stats

t_stat, p_value = stats.ttest_ind(organic, paid, equal_var=False)
```

> [!STORY] The brewer who signed himself "Student"
> The t-test was born in a brewery. William Sealy Gosset worked for Guinness
> in Dublin, where he needed to draw conclusions from small samples of barley
> and hops. Guinness didn't want competitors to know it employed
> statisticians, so when Gosset published his method in 1908 he signed it
> "Student". That is why, over a century later, you'll still read about
> "Student's t-distribution".

> [!NOTE]
> A small p-value says the gap is unlikely *if* both channels truly spent the
> same. It does **not** say paid ads *cause* lower spend: the channels differ
> in many ways at once. Separating correlation from causation is Module 09.

That's the whole loop in one screen: a question, some data, a look at its shape, a quantified comparison, and an honest statement of what it does and does not prove.

## How the repository is organised

```text
full-stack-data-science/
├── pyproject.toml              ← dependency declarations (level 2)
├── requirements-lock.txt       ← exact pins (level 3)
├── assets/images/              ← generated diagrams
├── data/
│   ├── raw/                    ← immutable inputs; NEVER edit by hand
│   └── processed/              ← outputs of your notebooks
├── modules/01-python-foundations/
│   ├── notebooks/              ← the lessons
│   ├── scripts/                ← plain-text .py mirrors
│   └── exercises/
├── src/dstoolkit/              ← shared, tested, importable code
└── tests/                      ← pytest suite for src/
```

### The one rule about raw data

**Raw data is immutable.** You never edit it, never overwrite it, never "just fix that one bad row" in Excel. Every transformation is code that reads from `raw/` and writes to `processed/`. If the cleaning lives in code, anyone can re-run it and get your numbers. If it lives in an undocumented manual edit, nobody can, including you.

## The stories in this book

Each chapter includes a short episode from the history of science and industry, chosen because it makes one idea impossible to forget. Here they are on one page; sources are listed at the back of the book.

![The true stories told in this book, in chronological order.](assets/images/book/m01_timeline.png){width=100}

## Try it yourself

These match the exercises at the end of notebook 01; worked solutions are in notebook 09.

1. **Environment audit.** Print the Python version as a `(major, minor, micro)` tuple, whether you're in a virtual environment, and the versions of `numpy`, `pandas` and `matplotlib`. Every package has a `__version__` attribute.
2. **A different seed.** Re-run the customer analysis with `seed=7`. Do the group means change? Does the *conclusion*? What does that tell you about trusting a single sample?
3. **Break the notebook on purpose.** Run all cells, then re-run only the cell that creates `customers` with `N_CUSTOMERS = 50`, and look at `summary`. Explain what happened in one sentence, and name the menu command that prevents it.
4. **Lifecycle mapping.** Pick a data product you use (a recommendation feed, a fraud alert, a delivery ETA) and write one sentence per lifecycle stage about what its team probably had to do.

> [!KEY]
> - Data science is about **half data work**. The modelling is the small, visible tip.
> - The lifecycle is a **loop**: going backwards is the process working.
> - **One virtual environment per project.** Declare loose constraints for humans; generate exact pins for machines.
> - A notebook is a long-running process. **Restart & Run All** before you trust or share a result.
> - Promote working code out of notebooks into a tested package.
> - **Raw data is immutable.** Every transformation is code.
