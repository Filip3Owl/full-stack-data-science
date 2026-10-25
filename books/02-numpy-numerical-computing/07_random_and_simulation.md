# Dice You Can Replay
@subtitle Reproducible randomness, Monte Carlo, and the bootstrap
@time About 75 minutes
@notebook 07_random_and_simulation.ipynb
@learn
- The modern `Generator` API, and why the legacy `np.random.seed` is a liability.
- Reproducibility that survives parallelism, via `SeedSequence` and `spawn`.
- The distributions you'll actually draw from, and when each one applies.
- Sampling: with and without replacement, weighted, permutations.
- **Monte Carlo**: answering questions that have no formula.
- **The bootstrap**: confidence intervals without distributional assumptions.
@end

@lead Computers are deterministic machines, so the "random" numbers they produce are really an extremely long, extremely well-shuffled sequence that starts from a seed. That's not a weakness; it's a superpower. The same seed replays the same dice, which means a simulation can be checked, debugged and reproduced by anyone. This chapter is the bridge into the statistics half of the course: everything in Modules 07 to 09 either uses these tools or is checked against them.

## The `Generator` API

NumPy has two random systems. Use the newer one.

```python
np.random.seed(42); np.random.rand(3)       # ✘ legacy: one hidden global state
rng = np.random.default_rng(seed=42)        # ✔ modern: an object you own
rng.random(3)
```

### Why the global state is a problem

`np.random.seed` sets state that *every* library in the process shares. If scikit-learn, a plotting call or a colleague's helper draws a number between your seed and your draw, your "reproducible" result silently changes:

```python
np.random.seed(42); first = np.random.rand(3)
np.random.seed(42); third_party_function(); second = np.random.rand(3)
np.array_equal(first, second)     # False: reproducibility broken
```

A `Generator` keeps its state privately, inside the object, so nothing else can disturb it.

![Left: every caller draws from one shared global state. Right: a parent SeedSequence spawns independent, reproducible streams, one per worker.](assets/images/book/m02_seed_streams.png){width=100}

> [!TIP]
> **Create one `Generator` at the top of your script or notebook and pass it
> down.** Functions that need randomness should take `rng` as a parameter
> rather than reaching for a global, the same "no hidden state" principle as
> Module 01's pure functions:
>
> ```python
> def simulate_returns(rng: np.random.Generator, n_days: int = 252):
>     return rng.normal(loc=0.0005, scale=0.01, size=n_days)
> ```

### Independent streams for parallel work

Giving each worker process `seed + i` is a genuine mistake: nearby seeds can produce correlated streams. `SeedSequence` spawning gives provably independent children, and re-spawning from the same parent reproduces them exactly:

```python
children = np.random.SeedSequence(42).spawn(4)
workers = [np.random.default_rng(child) for child in children]
```

> [!STORY] A best-seller made of random digits
> Before computers could generate random numbers cheaply, scientists bought
> them by the book. In 1955 the RAND Corporation published *A Million Random
> Digits with 100,000 Normal Deviates*, produced by an electronic "roulette
> wheel" and then carefully tested and corrected for bias. It was used for
> decades to design experiments and run simulations by hand. Its pages are
> about as unpredictable as reading gets, and they inspired a long-running
> internet joke of tongue-in-cheek reader reviews. Today one line, `rng =
> np.random.default_rng(42)`, gives you more digits than RAND could print,
> and the same ones every time.

## The distributions you'll use

Each distribution models a different kind of process. Choosing one is a **modelling decision**, not a technical one.

![Six distributions you'll draw from constantly, each with the kind of process it describes.](assets/images/book/m02_distributions.png){width=100}

| Distribution | Models | You'll meet it in |
| :--- | :--- | :--- |
| **Uniform** | any value in a range, equally likely | random initialisation, shuffling |
| **Normal** | sums of many small effects | measurement error, residuals |
| **Lognormal** | quantities that are *products* of effects | income, spend, session length |
| **Exponential** | waiting time between events | churn, time to failure |
| **Binomial** | successes in *n* independent trials | conversions out of visitors |
| **Poisson** | count of events in a fixed window | arrivals per hour, defects per batch |

> [!NOTE]
> Customer spend is **lognormal**, not normal, which is why the mean sat well
> above the median in Module 01's very first analysis. Recognising the
> generating process tells you which summary statistic is honest.

## Sampling

```python
population = np.array(["A", "B", "C", "D", "E"])
rng.choice(population, 8)                             # with replacement
rng.choice(population, 4, replace=False)              # without
rng.choice(population, 8, p=[0.6, 0.2, 0.1, 0.05, 0.05])   # weighted

rng.permutation(arr)    # returns a shuffled COPY
rng.shuffle(arr)        # shuffles IN PLACE and returns None
```

A train/test split is just a permutation cut in two, which is what scikit-learn does for you:

```python
indices = rng.permutation(n_samples)
split = int(0.8 * n_samples)
train_idx, test_idx = indices[:split], indices[split:]
```

## Monte Carlo

When a quantity is hard to derive, **simulate it**: draw many random outcomes and average. The estimate converges at a rate of `1/√n`, which is slow, but it works for problems where no formula exists at all.

The classic warm-up estimates π. Scatter points uniformly over a unit square; the fraction landing inside the quarter circle approaches π/4:

```python
def estimate_pi(rng: np.random.Generator, n_points: int) -> float:
    points = rng.random((n_points, 2))
    inside = (points**2).sum(axis=1) <= 1.0
    return 4.0 * inside.mean()
```

![Left: what the simulation does. Right: the running estimate wobbling towards π, inside a band of ±2 standard errors that narrows like 1/√n.](assets/images/book/m02_monte_carlo.png){width=100}

That `1/√n` is the catch: **a hundred times more points buys only ten times more accuracy.**

> [!STORY] A card game, a hospital bed, and the hydrogen bomb
> In 1946 the mathematician Stanislaw Ulam was recovering from an illness and
> passing the time with solitaire. He wondered what the chance was of a game
> coming out, tried to calculate it, gave up, and realised he could simply
> *play many games and count*. He took the idea to John von Neumann, and the
> two applied it to neutron diffusion problems at Los Alamos that had no
> tractable formula, running the calculations on ENIAC, one of the first
> electronic computers. Their colleague Nicholas Metropolis suggested the
> code name **Monte Carlo**, after the casino where Ulam's uncle liked to
> gamble. Your `estimate_pi` is a direct descendant.

> [!NOTE]
> Estimating π by chance is far older still. In 1777 the French naturalist
> Georges-Louis Leclerc, Comte de Buffon, analysed the probability that a
> needle dropped on a floor of parallel boards crosses a line, a probability
> that involves π. "Buffon's needle" is often called the first Monte Carlo
> experiment.

### Checking simulation against a formula

*"A checkout page converts at 12%. If 40 people visit, what's the probability that at least 8 convert?"* This one does have a closed form, which makes it a good test: half a million simulated visits agree with the exact binomial answer to about three decimal places, just as `1/√n` predicts.

```python
conversions = rng.binomial(40, 0.12, 500_000)
simulated = (conversions >= 8).mean()
exact = 1 - stats.binom.cdf(7, 40, 0.12)
```

## The bootstrap

The bootstrap estimates the uncertainty of *any* statistic by **resampling the data you already have, with replacement**. It needs **no assumption** about the underlying distribution, which is why it works for medians, ratios and correlations where classical formulas are awkward or don't exist.

![One observed sample, resampled with replacement: some points appear twice, some vanish. Compute the statistic on each resample, thousands of times, and read off the percentiles.](assets/images/book/m02_bootstrap.png){width=100}

The notebook's `bootstrap_ci` draws all the resamples in one vectorised call instead of a loop:

```python
def bootstrap_ci(data, statistic=np.mean, *, n_resamples=10_000,
                 confidence=0.95, rng=None):
    rng = rng or np.random.default_rng()
    n = len(data)
    resample_idx = rng.integers(0, n, size=(n_resamples, n))   # (B, n) at once
    estimates = statistic(data[resample_idx], axis=1)
    alpha = 1 - confidence
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(data)), float(low), float(high)
```

On 300 right-skewed spend values, it gives a 95% interval for the **mean** and, just as easily, for the **median**, which has no simple closed-form standard error. The bootstrap doesn't care. The notebook confirms the interval matches SciPy's own `scipy.stats.bootstrap` to within 1%.

> [!NOTE]
> Plot the bootstrap distribution of the *mean* and it looks close to normal,
> even though the data itself is strongly skewed. That's the **Central Limit
> Theorem** appearing on its own; Module 07 develops it properly.

> [!STORY] Pulling yourself up by your bootstraps
> In 1979 the Stanford statistician Bradley Efron published "Bootstrap
> methods: another look at the jackknife". The name comes from the old
> expression about pulling yourself up by your own bootstraps, an impossible
> feat, because the method seems to get something from nothing: uncertainty
> estimates from a single sample, without new data and without a formula.
> What made it practical was cheap computing. A calculation that would have
> taken a 1950s statistician weeks became a few seconds. Efron received the
> International Prize in Statistics in 2018, largely for this idea.

## Try it yourself

Exercises 7.1 to 7.5 in notebook 07; solutions in notebook 10.

1. **Break and fix reproducibility.** Show that an unrelated function called between `np.random.seed` and a draw changes the result; rewrite with an injected `Generator` and show it can't happen.
2. **Choose the distribution.** Name and justify one for: support tickets per hour; time between two tickets; whether each of 500 emails is opened; yearly revenue per customer; the error of a well-calibrated sensor.
3. **The birthday problem.** Simulate the chance that at least two people in a group of `k` share a birthday, for `k` from 2 to 60. Find the smallest `k` above 50% and compare with the exact formula.
4. **Bootstrap a difference.** For two groups of spend, bootstrap a 95% interval for the **difference in medians**. Does it contain zero, and what does that let you claim, and not claim?
5. **Sample size by simulation.** A page converts at 10%. Find the sample size per arm needed to detect a lift to 11% with 80% power at α = 0.05, by simulating 2,000 A/B tests per candidate size. That's the core of Module 09.

> [!KEY]
> - Use **`np.random.default_rng(seed)`**. The legacy global state makes reproducibility fragile.
> - **Inject the generator** into functions rather than reaching for a global.
> - For parallel work use **`SeedSequence.spawn`**, never `seed + i`.
> - Match the **distribution to the process**: counts → Poisson, waiting times → exponential, multiplicative quantities → lognormal.
> - **Monte Carlo** answers questions with no formula; its error shrinks like **1/√n**.
> - **The bootstrap** gives an interval for any statistic, with no distributional assumption: resample with replacement and take percentiles.
> - Vectorise the resampling: `rng.integers(0, n, size=(B, n))`.
