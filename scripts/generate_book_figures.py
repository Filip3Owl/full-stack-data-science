"""Generate the illustrations used by the companion books in ``books/``.

The notebooks' own diagrams live in ``scripts/generate_figures.py``; the books
reuse several of those and add the figures below. Same rules apply: diagrams are
code, the palette is the validated course palette, and every chart carries
direct labels so that no identity rests on colour alone.

    .venv/bin/python scripts/generate_book_figures.py            # everything
    .venv/bin/python scripts/generate_book_figures.py --only m02  # one module

Figures are drawn at their **printed size**: the book's text column is 6.3 in
wide, so a full-width figure is ``figsize=(6.3, …)`` and its 8.5 pt labels print
at 8.5 pt — the same size as the captions beneath them. Every figure is written
to ``assets/images/book/`` as a 220 dpi PNG.

A few charts are *measurements* (timings, file sizes). They are re-measured on
every run, so the exact numbers vary by machine while the shapes do not.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from dstoolkit.plotting import COLORS, setup_plot_style

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "assets" / "images" / "book"
FONTS_DIR = REPO_ROOT / "assets" / "fonts"

DPI = 220
FULL = 6.3          # the book's text width, in inches

PRIMARY, ACCENT, TEAL = COLORS["primary"], COLORS["accent"], COLORS["teal"]
PURPLE, DANGER, MUTED, TEXT = COLORS["purple"], COLORS["danger"], COLORS["muted"], COLORS["text"]
SOFT_TEXT = "#5B6068"
GRID = "#E8E8E8"

# Light tints of each hue, for fills behind dark outlines and labels.
TINT = {
    PRIMARY: "#E3ECF7",
    ACCENT: "#F8E9DD",
    TEAL: "#DDF1EE",
    PURPLE: "#ECE3F3",
    DANGER: "#F6E1DF",
    MUTED: "#EFEFEF",
}

MONO = "JetBrains Mono NL"


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def setup_book_style() -> None:
    """Course style, re-typeset in the book's faces at print sizes."""
    setup_plot_style(dpi=DPI)
    for path in FONTS_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update({
        # DejaVu after Inter: matplotlib falls back per glyph, so symbols such
        # as ✔ and ᵀ still render.
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.8,
        "axes.titlepad": 8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "savefig.dpi": DPI,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"  ✔ {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def canvas(width: float, height: float, xlim=None, ylim=None):
    """A blank, unit-scaled drawing surface ``width × height`` inches."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(*(xlim or (0, width)))
    ax.set_ylim(*(ylim or (0, height)))
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def box(ax, x, y, w, h, label="", *, color=PRIMARY, fill=None, text_color=None,
        size=8.5, weight="normal", family=None, radius=0.06, lw=1.2, z=3,
        ha="center", va="center"):
    """Rounded rectangle with optional centred label."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill if fill is not None else TINT.get(color, "white"),
        edgecolor=color, linewidth=lw, zorder=z,
    ))
    if label:
        tx = x + w / 2 if ha == "center" else x + 0.08
        ax.text(tx, y + h / 2, label, ha=ha, va=va, fontsize=size,
                fontweight=weight, color=text_color or TEXT, zorder=z + 1,
                family=family or mpl.rcParams["font.family"], linespacing=1.35)


def arrow(ax, start, end, *, color=MUTED, lw=1.2, style="-|>", rad=0.0, z=2, ls="-",
          scale=10):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw, mutation_scale=scale, zorder=z, linestyle=ls,
        shrinkA=0, shrinkB=0,
    ))


def label(ax, x, y, text, *, size=8.5, color=TEXT, weight="normal", ha="center",
          va="center", family=None, style="normal", z=5, **kw):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            family=family or mpl.rcParams["font.family"], fontstyle=style, zorder=z,
            linespacing=1.35, **kw)


def cell_grid(ax, x0, y0, values, *, size=0.34, color=PRIMARY, highlight=None,
              hl_color=ACCENT, fmt="{}", text_size=7.6, fill="white"):
    """Draw a 2-D array as a grid of cells; ``highlight`` is a boolean mask."""
    values = np.atleast_2d(values)
    rows, cols = values.shape
    for i in range(rows):
        for j in range(cols):
            hot = highlight is not None and bool(highlight[i, j])
            ax.add_patch(Rectangle(
                (x0 + j * size, y0 - (i + 1) * size), size, size,
                facecolor=TINT[hl_color] if hot else fill,
                edgecolor=hl_color if hot else color, linewidth=1.1 if hot else 0.7,
                zorder=3 if hot else 2,
            ))
            ax.text(x0 + (j + 0.5) * size, y0 - (i + 0.5) * size, fmt.format(values[i, j]),
                    ha="center", va="center", fontsize=text_size, family=MONO,
                    color=TEXT, zorder=4)


def clean_axes(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7)
    ax.grid(False, axis="x" if grid_axis == "y" else "y")
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C8CBD0")
    ax.tick_params(colors=SOFT_TEXT, length=3)


def best_of(fn, *args, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - start)
    return best


def timeline(name: str, events: list[tuple[int, str, str]]) -> None:
    """A two-column vertical timeline: year, a dot on a spine, one-line label."""
    half = (len(events) + 1) // 2
    columns = [events[:half], events[half:]]
    row_h = 0.3
    fig, ax = canvas(FULL, row_h * half + 0.15)
    for c, column in enumerate(columns):
        x = 0.05 + c * 3.2
        top = row_h * half
        ax.plot([x + 0.55, x + 0.55], [top - 0.05, top - row_h * len(column) + 0.12],
                color="#D0D3D8", lw=1.2, zorder=1)
        for k, (year, text, color) in enumerate(column):
            y = top - (k + 0.5) * row_h
            label(ax, x + 0.42, y, str(year), size=8, weight="bold", color=color,
                  ha="right")
            ax.scatter([x + 0.55], [y], s=26, color=color, zorder=3,
                       edgecolors="white", linewidths=0.8)
            label(ax, x + 0.7, y, text.replace("\n", " "), size=7.6, color=TEXT,
                  ha="left")
    save(fig, name)


# ===========================================================================
# MODULE 01 — Python Foundations
# ===========================================================================
def m01_cover() -> None:
    """Abstract cover: labels (names) tied to objects, the idea the module turns on."""
    rng = np.random.default_rng(8)
    fig, ax = canvas(6.3, 3.3)
    palette = [PRIMARY, ACCENT, TEAL, PURPLE, DANGER]

    centres = []
    while len(centres) < 15:
        c = rng.uniform([0.5, 0.45], [5.8, 2.85])
        if all(np.hypot(*(c - d)) > 0.78 for d in centres):
            centres.append(c)

    for k, (cx, cy) in enumerate(centres):
        color = palette[k % 5]
        r = rng.uniform(0.2, 0.36)
        ax.add_patch(Circle((cx, cy), r, facecolor=TINT[color], edgecolor=color,
                            linewidth=1.3, zorder=3))
        for _ in range(rng.integers(1, 3)):
            angle = rng.uniform(0, 2 * np.pi)
            dist = r + rng.uniform(0.32, 0.55)
            tx, ty = cx + dist * np.cos(angle), cy + dist * np.sin(angle)
            ax.plot([tx, cx + r * np.cos(angle)], [ty, cy + r * np.sin(angle)],
                    color=color, linewidth=0.8, alpha=0.8, zorder=2)
            box(ax, tx - 0.17, ty - 0.07, 0.34, 0.14, color=color, fill="white",
                radius=0.03, lw=0.9)
    save(fig, "m01_cover.png")


def m01_timeline() -> None:
    timeline("m01_timeline.png", [
        (1854, "Snow maps\ncholera", PRIMARY),
        (1908, "'Student'\nt-test", TEAL),
        (1947, "the first\n'bug'", DANGER),
        (1953, "Luhn invents\nhashing", ACCENT),
        (1964, "word counts settle\nthe Federalist", PURPLE),
        (1967, "Simula\nclasses", PRIMARY),
        (1982, "Vancouver index\nrounding", DANGER),
        (1991, "Python\n0.9.0", TEAL),
        (1999, "Mars Climate\nOrbiter lost", DANGER),
        (2004, "MapReduce", ACCENT),
        (2010, "scikit-learn\nreleased", PURPLE),
        (2013, "Parquet;\nReinhart–Rogoff", PRIMARY),
        (2020, "PHE loses\n16k cases", DANGER),
    ])


def m01_time_split() -> None:
    """Waffle chart of where a data scientist's time goes."""
    parts = [("Framing", 15, PRIMARY), ("Data work", 50, ACCENT),
             ("Modelling &\nanalysis", 20, TEAL), ("Delivery", 15, PURPLE)]
    fig, ax = canvas(FULL, 1.95)
    size, gap = 0.255, 0.05
    step = size + gap
    cells = [c for _, n, c in parts for _ in range(n)]
    for k, color in enumerate(cells):
        col, row = divmod(k, 5)
        ax.add_patch(FancyBboxPatch(
            (0.05 + col * step, 1.55 - row * step * 0.62), size, size * 0.55,
            boxstyle="round,pad=0,rounding_size=0.025", facecolor=color,
            edgecolor="none"))
    done = 0
    for name, n, color in parts:
        x0 = 0.05 + done / 5 * step
        x1 = 0.05 + (done + n) / 5 * step - gap
        ax.plot([x0, x1], [0.42, 0.42], color=color, lw=2.2, solid_capstyle="butt")
        label(ax, (x0 + x1) / 2, 0.17, f"{name}  {n}%", size=7.6, color=TEXT)
        done += n
    save(fig, "m01_time_split.png")


def m01_reproducibility() -> None:
    fig, ax = canvas(6.3, 2.2)
    steps = [
        ("1  Loose", "pandas", "“some version of pandas”", MUTED),
        ("2  Constrained", "pandas>=2.2,<3", "reproduces the API\n→ pyproject.toml", PRIMARY),
        ("3  Pinned", "pandas==3.0.5", "reproduces the exact bits\n→ requirements-lock.txt", TEAL),
    ]
    for k, (title, spec, meaning, color) in enumerate(steps):
        x, y = 0.05 + k * 2.1, 0.15 + k * 0.42
        box(ax, x, y, 1.95, 0.9, color=color, fill=TINT[color], radius=0.07)
        label(ax, x + 0.12, y + 0.72, title, size=8.4, weight="bold", ha="left",
              color=color if color != MUTED else SOFT_TEXT)
        label(ax, x + 0.12, y + 0.47, spec, size=7.8, family=MONO, ha="left")
        label(ax, x + 0.12, y + 0.2, meaning, size=6.9, color=SOFT_TEXT, ha="left")
    arrow(ax, (0.3, 1.45), (4.0, 2.05), color=MUTED, lw=1, rad=-0.1)
    label(ax, 1.2, 1.98, "more reproducible", size=7.2, color=SOFT_TEXT, style="italic")
    save(fig, "m01_reproducibility.png")


def m01_hidden_state() -> None:
    fig, ax = canvas(6.3, 2.35)
    cells = [
        ("[1]", "df = load_data()", PRIMARY),
        ("[3]", "df = df[df['revenue'] > 0]", DANGER),
        ("[2]", "print(len(df))", ACCENT),
    ]
    label(ax, 0.05, 2.22, "Order on the page", size=8, weight="bold", ha="left")
    for k, (counter, code, color) in enumerate(cells):
        y = 1.6 - k * 0.62
        label(ax, 0.32, y + 0.21, counter, size=8, family=MONO, color=color, weight="bold")
        box(ax, 0.6, y, 2.75, 0.42, code, color="#C8CBD0", fill="#F5F6F8", size=7.6,
            family=MONO, ha="left", radius=0.04, lw=0.8)

    label(ax, 3.9, 2.22, "Order the kernel actually ran them", size=8, weight="bold",
          ha="left")
    order = [(PRIMARY, "1  load"), (ACCENT, "2  print — reports N rows"),
             (DANGER, "3  filter — df changes after the print")]
    for k, (color, text) in enumerate(order):
        y = 1.68 - k * 0.5
        ax.scatter([4.0], [y + 0.1], s=55, color=color, zorder=3)
        label(ax, 4.2, y + 0.1, text, size=7.6, ha="left")
        if k < 2:
            arrow(ax, (4.0, y - 0.02), (4.0, y - 0.3), color=MUTED, lw=1)
    label(ax, 3.9, 0.12, "Run it again and the filter applies twice.\n"
          "Fix: Kernel → Restart & Run All.", size=7.3, color=DANGER, ha="left")
    save(fig, "m01_hidden_state.png")


def m01_promotion_loop() -> None:
    fig, ax = canvas(6.3, 1.5)
    steps = ["explore\nin a cell", "wrap it in\na function", "move it to\nsrc/dstoolkit",
             "write a\ntest", "import it\nback"]
    colors_ = [MUTED, PRIMARY, PRIMARY, TEAL, TEAL]
    for k, (text, color) in enumerate(zip(steps, colors_, strict=True)):
        x = 0.05 + k * 1.27
        box(ax, x, 0.45, 1.0, 0.62, text, color=color, size=7.6,
            text_color=TEXT if color != MUTED else SOFT_TEXT)
        if k < 4:
            arrow(ax, (x + 1.03, 0.76), (x + 1.24, 0.76), color=MUTED)
    arrow(ax, (5.65, 0.42), (0.55, 0.42), color=TEAL, rad=-0.07, lw=1, ls="--")
    label(ax, 3.1, 0.0, "reuse it in the next notebook", size=7.2, color=TEAL,
          style="italic")
    save(fig, "m01_promotion_loop.png")


def m01_float_gap() -> None:
    """0.1 + 0.2 lands one representable step above the double nearest 0.3."""
    fig, ax = plt.subplots(figsize=(FULL, 1.9))
    base = 0.3
    ulp = np.spacing(base)
    ks = np.arange(-3, 4)
    xs = base + ks * ulp
    ax.set_xlim(xs[0] - ulp * 0.6, xs[-1] + ulp * 0.6)
    ax.set_ylim(-1.1, 1.75)
    ax.axis("off")
    ax.plot([xs[0] - ulp * 0.5, xs[-1] + ulp * 0.5], [0, 0], color="#C8CBD0", lw=1.4)
    for x in xs:
        ax.plot([x, x], [-0.12, 0.12], color="#9AA0A8", lw=1)

    exact = 0.3
    nearest = float(np.float64(0.3))
    computed = 0.1 + 0.2
    ax.annotate("", xy=(nearest, 0.14), xytext=(nearest, 0.75),
                arrowprops={"arrowstyle": "-|>", "color": PRIMARY, "lw": 1.3})
    ax.text(nearest - ulp * 0.08, 0.82, "the double we get\nwhen we type 0.3", ha="right",
            va="bottom", fontsize=7.4, color=PRIMARY)
    ax.annotate("", xy=(computed, 0.14), xytext=(computed, 0.75),
                arrowprops={"arrowstyle": "-|>", "color": DANGER, "lw": 1.3})
    ax.text(computed + ulp * 0.08, 0.82, "0.1 + 0.2\n= 0.30000000000000004", ha="left",
            va="bottom", fontsize=7.4, color=DANGER)
    ax.scatter([exact], [0], marker="|", s=200, color=TEXT, zorder=5, linewidths=1.6)
    ax.text(exact, -0.32, "true 0.3\n(not representable)", ha="center", va="top",
            fontsize=7.2, color=SOFT_TEXT)
    ax.annotate("", xy=(xs[-2], -0.55), xytext=(xs[-1], -0.55),
                arrowprops={"arrowstyle": "<->", "color": MUTED, "lw": 0.9})
    ax.text((xs[-2] + xs[-1]) / 2, -0.68, f"one step ≈ {ulp:.1e}", ha="center",
            va="top", fontsize=7, color=SOFT_TEXT)
    ax.text(xs[0] - ulp * 0.55, 1.72, "Zoomed in on the number line around 0.3: "
            "only the tick marks exist as float64 values", fontsize=7.6,
            color=SOFT_TEXT, ha="left", va="top")
    save(fig, "m01_float_gap.png")


def m01_shallow_deep() -> None:
    fig, ax = canvas(6.3, 2.5)

    def lst(x, y, title, color):
        label(ax, x + 0.45, y + 0.62, title, size=7.8, weight="bold", color=color)
        box(ax, x, y, 0.9, 0.4, "list", color=color, size=7.4, family=MONO)

    def obj(x, y, color, text="{'price': 10}"):
        box(ax, x, y, 1.15, 0.38, text, color=color, size=6.8, family=MONO)

    label(ax, 1.3, 2.38, "Shallow copy — copy.copy(x)", size=8.4, weight="bold")
    lst(0.1, 1.45, "original", PRIMARY)
    lst(1.65, 1.45, "shallow", ACCENT)
    obj(0.75, 0.35, PRIMARY)
    arrow(ax, (0.55, 1.43), (1.15, 0.78), color=PRIMARY)
    arrow(ax, (2.1, 1.43), (1.5, 0.78), color=ACCENT)
    label(ax, 1.33, 0.1, "one shared inner dict — edits leak", size=7, color=DANGER)

    ax.plot([3.2, 3.2], [0.1, 2.4], color=GRID, lw=1)

    label(ax, 4.85, 2.38, "Deep copy — copy.deepcopy(x)", size=8.4, weight="bold")
    lst(3.6, 1.45, "original", PRIMARY)
    lst(5.2, 1.45, "deep", TEAL)
    obj(3.48, 0.35, PRIMARY)
    obj(5.08, 0.35, TEAL)
    arrow(ax, (4.05, 1.43), (4.05, 0.78), color=PRIMARY)
    arrow(ax, (5.65, 1.43), (5.65, 0.78), color=TEAL)
    label(ax, 4.85, 0.1, "independent all the way down", size=7, color=TEAL)
    save(fig, "m01_shallow_deep.png")


def m01_hash_table() -> None:
    fig, ax = canvas(6.3, 2.55)
    names = ["ana", "bruno", "carla", "diego", "elena", "felipe"]

    label(ax, 1.4, 2.42, "list: check one by one — O(n)", size=8.3, weight="bold",
          color=DANGER)
    for k, name in enumerate(names):
        hot = name == "elena"
        box(ax, 0.05 + k * 0.5, 1.7, 0.47, 0.36, name, color=DANGER if hot else "#C8CBD0",
            fill=TINT[DANGER] if hot else "white", size=6.3, family=MONO, radius=0.03)
    for k in range(4):
        arrow(ax, (0.28 + k * 0.5, 1.62), (0.28 + (k + 1) * 0.5, 1.62), color=DANGER,
              lw=0.8, rad=0.5, scale=7)
    label(ax, 1.4, 1.28, "'elena' in names → 5 comparisons", size=7.2, color=SOFT_TEXT)

    ax.plot([3.15, 3.15], [0.1, 2.45], color=GRID, lw=1)

    label(ax, 4.75, 2.42, "set: hash, then jump — O(1)", size=8.3, weight="bold",
          color=TEAL)
    box(ax, 3.35, 1.62, 0.75, 0.38, "'elena'", color=TEAL, size=7, family=MONO)
    arrow(ax, (4.12, 1.81), (4.5, 1.81), color=TEAL)
    box(ax, 4.52, 1.62, 0.6, 0.38, "hash()", color=TEAL, fill="white", size=7,
        family=MONO)
    arrow(ax, (5.14, 1.75), (5.43, 1.38), color=TEAL)
    for k in range(7):
        y = 2.2 - k * 0.3
        filled = k in (0, 2, 3, 5)
        hot = k == 3
        ax.add_patch(Rectangle((5.45, y - 0.27), 0.75, 0.27,
                               facecolor=TINT[TEAL] if hot else "white",
                               edgecolor=TEAL if hot else "#C8CBD0", lw=0.9))
        label(ax, 6.29, y - 0.135, str(k), size=6.3, color=SOFT_TEXT, family=MONO)
        if filled:
            label(ax, 5.825, y - 0.135, ["ana", "", "carla", "elena", "", "bruno"][k],
                  size=6.3, family=MONO)
    label(ax, 4.3, 0.75, "bucket 3 → found,\nwhatever the size", size=7.2,
          color=SOFT_TEXT)
    save(fig, "m01_hash_table.png")


def m01_set_vs_list() -> None:
    """Measured: membership cost for a list versus a set, as size grows."""
    rng = np.random.default_rng(42)
    sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    list_t, set_t = [], []
    for size in sizes:
        universe = list(range(size))
        lookups = rng.integers(size, size * 2, size=300).tolist()
        as_set = set(universe)
        list_t.append(best_of(lambda c, q: [v in c for v in q], universe, lookups,
                              repeats=1))
        set_t.append(best_of(lambda c, q: [v in c for v in q], as_set, lookups))

    fig, ax = plt.subplots(figsize=(FULL * 0.62, 2.6))
    ax.plot(sizes, np.array(list_t) * 1e3, marker="o", color=DANGER)
    ax.plot(sizes, np.array(set_t) * 1e3, marker="s", color=TEAL)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("container size (elements)")
    ax.set_ylabel("time for 300 lookups (ms)")
    clean_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    ax.text(sizes[-1] * 0.8, list_t[-1] * 1e3, "list  O(n)", color=DANGER,
            fontweight="bold", ha="right", va="bottom", fontsize=8)
    ax.text(sizes[-1] * 0.8, set_t[-1] * 1e3 * 1.6, "set  O(1)", color=TEAL,
            fontweight="bold", ha="right", va="bottom", fontsize=8)
    ratio = list_t[-1] / set_t[-1]
    magnitude = 10 ** max(int(np.log10(ratio)) - 1, 0)
    ax.set_title(f"At 500k items the set is ≈{round(ratio / magnitude) * magnitude:,.0f}× "
                 "faster", loc="left")
    save(fig, "m01_set_vs_list.png")


def m01_comprehension() -> None:
    fig, ax = canvas(6.3, 1.45)
    parts = [
        ("[", None, 0.0),
        ("t * 9 / 5 + 32", PRIMARY, 0.0),
        ("for t in temps", ACCENT, 0.0),
        ("if t > 20", TEAL, 0.0),
        ("]", None, 0.0),
    ]
    notes = {PRIMARY: "what to produce", ACCENT: "where items come from",
             TEAL: "optional filter"}
    x = 0.25
    for text, color, _ in parts:
        width = 0.105 * len(text) + 0.12
        if color:
            box(ax, x, 0.72, width, 0.42, text, color=color, size=9, family=MONO,
                radius=0.05)
            ax.plot([x + width / 2, x + width / 2], [0.7, 0.42], color=color, lw=1)
            label(ax, x + width / 2, 0.28, notes[color], size=7.4, color=color,
                  weight="bold")
        else:
            label(ax, x + width / 2, 0.93, text, size=14, family=MONO, color=SOFT_TEXT)
        x += width + 0.12
    save(fig, "m01_comprehension.png")


def m01_generator_memory() -> None:
    sizes = np.array([10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000])
    list_mib = np.array([sys.getsizeof([i * 1.5 for i in range(n)]) for n in sizes]) / 1024**2
    gen_bytes = sys.getsizeof(i * 1.5 for i in range(10))

    fig, ax = plt.subplots(figsize=(FULL * 0.62, 2.5))
    ax.plot(sizes, list_mib, marker="o", color=DANGER)
    ax.plot(sizes, np.full(len(sizes), gen_bytes / 1024**2), marker="s", color=TEAL)
    clean_axes(ax)
    ax.set_xlabel("number of elements")
    ax.set_ylabel("memory held (MiB)")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.text(sizes[-1] * 0.94, list_mib[-1], f"list  {list_mib[-1]:.1f} MiB",
            color=DANGER, ha="right", va="center", fontweight="bold", fontsize=8)
    ax.text(sizes[-1], 0.55, f"generator  {gen_bytes} bytes, flat",
            color=TEAL, ha="right", va="bottom", fontweight="bold", fontsize=8)
    ax.set_title("A generator's footprint does not grow", loc="left")
    save(fig, "m01_generator_memory.png")


def m01_lazy_pipeline() -> None:
    fig, ax = canvas(6.3, 1.5)
    stages = [("file", MUTED), ("parse_lines()", PRIMARY), ("filter_large()", ACCENT),
              ("for record in …", TEAL)]
    for k, (text, color) in enumerate(stages):
        x = 0.05 + k * 1.6
        box(ax, x, 0.62, 1.3, 0.5, text, color=color, size=7.6, family=MONO,
            text_color=TEXT)
        if k < 3:
            arrow(ax, (x + 1.33, 0.87), (x + 1.57, 0.87), color=MUTED)
    for k in range(3):
        x = 1.45 + k * 1.6
        ax.add_patch(Circle((x, 0.87), 0.06, color=ACCENT, zorder=6))
    label(ax, 3.2, 0.25, "one record at a time is pulled through the whole chain —\n"
          "nothing runs until the final loop asks for the next item", size=7.3,
          color=SOFT_TEXT)
    save(fig, "m01_lazy_pipeline.png")


def m01_decorator() -> None:
    fig, ax = canvas(6.3, 2.05)
    box(ax, 1.3, 0.12, 3.7, 1.72, color=PURPLE, fill=TINT[PURPLE], radius=0.1)
    label(ax, 1.45, 1.66, "wrapper  — returned by @timed", size=7.8, weight="bold",
          color=PURPLE, ha="left")
    box(ax, 2.2, 0.62, 1.9, 0.62, "slow_sum(n)\nthe original", color=PRIMARY,
        fill="white", size=7.6)
    label(ax, 1.75, 0.93, "start\nclock", size=7, color=SOFT_TEXT)
    label(ax, 4.55, 0.93, "stop clock,\nprint time", size=7, color=SOFT_TEXT)
    arrow(ax, (0.2, 0.93), (1.42, 0.93), color=MUTED)
    label(ax, 0.55, 1.12, "call", size=7, color=SOFT_TEXT)
    arrow(ax, (2.0, 0.93), (2.18, 0.93), color=MUTED)
    arrow(ax, (4.12, 0.93), (4.3, 0.93), color=MUTED)
    arrow(ax, (4.95, 0.93), (6.1, 0.93), color=MUTED)
    label(ax, 5.65, 1.12, "result", size=7, color=SOFT_TEXT)
    label(ax, 3.15, 0.32, "@timed  is just  slow_sum = timed(slow_sum)", size=7.4,
          family=MONO, color=TEXT)
    save(fig, "m01_decorator.png")


def m01_mutable_default() -> None:
    fig, ax = canvas(6.3, 1.65)
    box(ax, 0.05, 0.55, 1.55, 0.62, "def f(v, acc=[])\nruns ONCE", color=DANGER,
        size=7.4, family=MONO)
    box(ax, 2.35, 0.5, 1.0, 0.72, "the one\nlist object", color=DANGER, fill="white",
        size=7.6)
    arrow(ax, (1.62, 0.86), (2.33, 0.86), color=DANGER)
    for k, text in enumerate(["f(1) → [1]", "f(2) → [1, 2]", "f(3) → [1, 2, 3]"]):
        y = 1.3 - k * 0.48
        label(ax, 5.1, y, text, size=7.6, family=MONO, color=TEXT)
        arrow(ax, (4.2, y), (3.4, 0.86), color=MUTED, lw=0.9, rad=0.0)
    label(ax, 3.1, 0.12, "every call that omits acc shares — and grows — the same list",
          size=7.2, color=SOFT_TEXT)
    save(fig, "m01_mutable_default.png")


def m01_estimator() -> None:
    fig, ax = canvas(6.3, 2.2)
    box(ax, 0.05, 1.3, 1.25, 0.55, "training data\nX_train", color=MUTED, size=7.4,
        text_color=TEXT)
    box(ax, 1.8, 1.3, 1.15, 0.55, "fit(X)", color=PRIMARY, size=8.4, family=MONO,
        weight="bold")
    box(ax, 3.45, 1.22, 1.45, 0.7, "learned state\nlower_bounds_\nupper_bounds_",
        color=ACCENT, size=7, family=MONO)
    arrow(ax, (1.32, 1.57), (1.78, 1.57))
    arrow(ax, (2.97, 1.57), (3.43, 1.57))
    label(ax, 2.38, 2.05, "LEARN", size=7.6, weight="bold", color=PRIMARY)

    box(ax, 0.05, 0.2, 1.25, 0.55, "any data\nX_new", color=MUTED, size=7.4,
        text_color=TEXT)
    box(ax, 1.8, 0.2, 1.15, 0.55, "transform(X)", color=TEAL, size=8.0, family=MONO,
        weight="bold")
    box(ax, 5.25, 0.2, 1.0, 0.55, "clipped X", color=TEAL, fill="white", size=7.4)
    arrow(ax, (1.32, 0.47), (1.78, 0.47))
    arrow(ax, (2.97, 0.47), (5.23, 0.47))
    arrow(ax, (4.17, 1.2), (3.6, 0.52), color=ACCENT, ls="--")
    label(ax, 2.38, 0.95, "APPLY — never learns", size=7.6, weight="bold", color=TEAL)
    label(ax, 5.75, 1.57, "trailing _ =\nlearned from data", size=7, color=ACCENT)
    save(fig, "m01_estimator.png")


def m01_composition() -> None:
    fig, ax = canvas(6.3, 2.2)
    label(ax, 1.45, 2.08, "Inheritance — “is a”", size=8.4, weight="bold")
    box(ax, 0.65, 1.35, 1.6, 0.45, "Transformer (ABC)", color=PURPLE, size=7.4)
    for k, name in enumerate(["LogTransformer", "SqrtTransformer"]):
        x = 0.05 + k * 1.5
        box(ax, x, 0.45, 1.35, 0.42, name, color=PRIMARY, size=7)
        arrow(ax, (x + 0.67, 0.89), (1.45, 1.33), color=MUTED)
    label(ax, 1.45, 0.12, "subclasses share one interface", size=7, color=SOFT_TEXT)

    ax.plot([3.15, 3.15], [0.1, 2.1], color=GRID, lw=1)

    label(ax, 4.75, 2.08, "Composition — “has a”", size=8.4, weight="bold")
    box(ax, 3.4, 0.4, 2.75, 1.4, color=TEAL, fill=TINT[TEAL], radius=0.08)
    label(ax, 3.52, 1.62, "Pipeline", size=7.8, weight="bold", color=TEAL, ha="left")
    for k, name in enumerate(["Sqrt", "Log", "…"]):
        box(ax, 3.55 + k * 0.88, 0.75, 0.72, 0.5, name, color=PRIMARY, fill="white",
            size=7.2)
        if k < 2:
            arrow(ax, (4.28 + k * 0.88, 1.0), (4.42 + k * 0.88, 1.0), color=MUTED)
    label(ax, 4.75, 0.12, "behaviour comes from combining parts", size=7,
          color=SOFT_TEXT)
    save(fig, "m01_composition.png")


def m01_row_vs_column() -> None:
    fig, ax = canvas(6.3, 2.35)
    rows = [["1", "organic", "BR", "54.20"], ["2", "paid", "US", "120.00"],
            ["3", "paid", "BR", "17.50"]]
    col_colors = [PRIMARY, ACCENT, TEAL, PURPLE]
    names = ["id", "channel", "country", "amount"]

    label(ax, 1.45, 2.22, "CSV — stored row by row", size=8.4, weight="bold")
    for i, row in enumerate(rows):
        y = 1.6 - i * 0.5
        x = 0.05
        for j, value in enumerate(row):
            w = 0.18 + 0.075 * len(value)
            box(ax, x, y, w, 0.36, value, color=col_colors[j], fill=TINT[col_colors[j]],
                size=6.8, family=MONO, radius=0.03, lw=0.8)
            x += w + 0.05
    label(ax, 1.45, 0.25, "reading one column still parses every byte\nand every type is re-guessed",
          size=7, color=SOFT_TEXT)

    ax.plot([3.15, 3.15], [0.1, 2.3], color=GRID, lw=1)

    label(ax, 4.75, 2.22, "Parquet — stored column by column", size=8.4, weight="bold")
    for j, (name, color) in enumerate(zip(names, col_colors, strict=True)):
        y = 1.75 - j * 0.4
        label(ax, 3.62, y + 0.15, name, size=7, color=color, weight="bold", ha="right")
        values = [r[j] for r in rows]
        box(ax, 3.7, y, 2.45, 0.3, "  ".join(values) + "  …", color=color,
            fill=TINT[color], size=6.8, family=MONO, radius=0.03, lw=0.8)
    label(ax, 4.75, 0.25, "one type per column: compresses well,\nskip the columns you do not need",
          size=7, color=SOFT_TEXT)
    save(fig, "m01_row_vs_column.png")


def m01_format_benchmark() -> None:
    """Measured: size and read time of the same table in CSV, JSON and Parquet."""
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 200_000
    orders = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "customer_id": rng.integers(1, 20_000, n),
        "order_date": pd.to_datetime("2024-01-01")
        + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "channel": rng.choice(["organic", "paid", "referral", "email"], n),
        "country": rng.choice(["BR", "US", "PT", "ES", "MX"], n),
        "amount": np.round(rng.lognormal(4.0, 0.8, n), 2),
        "quantity": rng.integers(1, 10, n),
        "is_gift": rng.choice([True, False], n, p=[0.15, 0.85]),
    })

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        writers = {
            "CSV": (lambda p: orders.to_csv(p, index=False),
                    lambda p: pd.read_csv(p, parse_dates=["order_date"])),
            "JSON": (lambda p: orders.to_json(p, orient="records", date_format="iso"),
                     lambda p: pd.read_json(p, orient="records")),
            "Parquet": (lambda p: orders.to_parquet(p, index=False),
                        lambda p: pd.read_parquet(p)),
        }
        for name, (write, read) in writers.items():
            path = tmp / f"bench.{name.lower()}"
            write(path)
            results[name] = (path.stat().st_size / 1024**2, best_of(read, path))

    names = list(results)
    format_colors = [ACCENT, DANGER, TEAL]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    for ax, idx, title, unit in [(axes[0], 0, "File size", "MB"),
                                 (axes[1], 1, "Time to read it back", "s")]:
        values = [results[n][idx] for n in names]
        ax.barh(names[::-1], values[::-1], color=format_colors[::-1], height=0.58)
        ax.set_title(title, loc="left")
        clean_axes(ax, grid_axis="x")
        ax.set_xlim(0, max(values) * 1.32)
        for k, value in enumerate(values[::-1]):
            ax.text(value, k, f"  {value:.2f} {unit}", va="center", fontsize=7.6,
                    color=TEXT)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=2)
    save(fig, "m01_format_benchmark.png")


def m01_try_flow() -> None:
    fig, ax = canvas(6.3, 2.0)
    box(ax, 0.05, 0.75, 1.0, 0.5, "try:\nrisky work", color=PRIMARY, size=7.4)
    box(ax, 1.75, 1.3, 1.25, 0.5, "except E:\nhandle E", color=DANGER, size=7.4)
    box(ax, 1.75, 0.2, 1.25, 0.5, "else:\nfollow-on work", color=TEAL, size=7.4)
    box(ax, 3.75, 0.75, 1.15, 0.5, "finally:\ncleanup", color=PURPLE, size=7.4)
    arrow(ax, (1.07, 1.05), (1.73, 1.52), color=DANGER)
    label(ax, 1.25, 1.42, "raised", size=6.8, color=DANGER)
    arrow(ax, (1.07, 0.95), (1.73, 0.45), color=TEAL)
    label(ax, 1.22, 0.5, "no error", size=6.8, color=TEAL)
    arrow(ax, (3.02, 1.55), (3.73, 1.08), color=MUTED)
    arrow(ax, (3.02, 0.45), (3.73, 0.92), color=MUTED)
    arrow(ax, (4.92, 1.0), (5.4, 1.0), color=MUTED)
    label(ax, 5.85, 1.0, "always\nruns", size=7.4, color=PURPLE, weight="bold")
    save(fig, "m01_try_flow.png")


def m01_log_levels() -> None:
    levels = [("DEBUG", "shapes, intermediate counts", MUTED),
              ("INFO", "“loaded 200,000 rows”", PRIMARY),
              ("WARNING", "“12 nulls imputed”", ACCENT),
              ("ERROR", "“one partition failed”", DANGER),
              ("CRITICAL", "“database unreachable”", PURPLE)]
    fig, ax = canvas(6.3, 2.2)
    for k, (name, example, color) in enumerate(levels):
        y = 0.12 + k * 0.4
        w = 1.2 + k * 0.32
        box(ax, 0.05, y, w, 0.32, name, color=color, size=7.6, weight="bold",
            family=MONO, radius=0.04,
            text_color=SOFT_TEXT if color == MUTED else TEXT)
        label(ax, w + 0.2, y + 0.16, example, size=7.4, color=SOFT_TEXT, ha="left")
    ax.plot([0.0, 6.2], [0.49, 0.49], color=DANGER, lw=1, ls="--")
    label(ax, 6.2, 0.36, "level=INFO hides everything below this line",
          size=7, color=DANGER, ha="right")
    save(fig, "m01_log_levels.png")


def m01_pipeline() -> None:
    fig, ax = canvas(6.3, 2.75)
    box(ax, 0.05, 0.2, 6.2, 2.35, color=MUTED, fill="#FAFAFA", radius=0.1, lw=0.9)
    label(ax, 0.2, 2.38, "impure shell — touches files", size=7.6, color=SOFT_TEXT,
          ha="left", weight="bold")
    box(ax, 0.15, 0.92, 0.92, 0.9, "extract\norders.csv\ncustomers\n.json", color=PRIMARY,
        size=6.2)
    box(ax, 1.17, 0.45, 3.7, 1.75, color=TEAL, fill=TINT[TEAL], radius=0.08)
    label(ax, 1.29, 2.03, "pure core — DataFrame in, new DataFrame out", size=7.4,
          color=TEAL, weight="bold", ha="left")
    steps = ["dedupe", "normalise", "business\nrules", "join\ncustomers", "add\nfeatures"]
    for k, text in enumerate(steps):
        x = 1.27 + k * 0.71
        box(ax, x, 1.05, 0.64, 0.6, text, color=TEAL, fill="white", size=6.2)
        if k < 4:
            arrow(ax, (x + 0.645, 1.35), (x + 0.705, 1.35), color=MUTED, scale=6)
    label(ax, 3.12, 0.7, "every dropped row is counted, with its reason", size=6.8,
          color=SOFT_TEXT)
    box(ax, 5.0, 1.05, 0.55, 0.6, "quality\ngate", color=DANGER, size=6.6)
    box(ax, 5.68, 1.05, 0.5, 0.6, "load\nparquet", color=PRIMARY, size=6.2)
    arrow(ax, (1.08, 1.37), (1.26, 1.37))
    arrow(ax, (4.87, 1.37), (4.98, 1.37))
    arrow(ax, (5.57, 1.37), (5.66, 1.37))
    label(ax, 5.27, 0.82, "reject rate\n> 10% → stop", size=6.3, color=DANGER, va="top")
    save(fig, "m01_pipeline.png")


# ===========================================================================
# MODULE 02 — NumPy & Numerical Computing
# ===========================================================================
def m02_cover() -> None:
    """Abstract cover: an isometric 3-D array with one slice lit up."""
    fig, ax = canvas(6.3, 3.3)
    rng = np.random.default_rng(3)
    shape = (4, 6, 8)        # depth, rows, cols
    dx, dy = np.array([0.36, 0.0]), np.array([0.0, 0.36])
    dz = np.array([0.2, 0.13])
    origin = np.array([0.9, 0.25])
    ramp = mpl.colors.LinearSegmentedColormap.from_list(
        "ramp", ["#FFFFFF", TINT[PRIMARY], "#9DB8DB", PRIMARY])

    for d in range(shape[0] - 1, -1, -1):
        for r in range(shape[1]):
            for c in range(shape[2]):
                base = origin + c * dx + (shape[1] - 1 - r) * dy + d * dz
                lit = c == 5
                face = TINT[ACCENT] if lit and d == 0 else ramp(rng.uniform(0.05, 0.6))
                edge = ACCENT if lit else "#B9C7DA"
                if d == 0:
                    ax.add_patch(Rectangle(base, 0.36, 0.36, facecolor=face,
                                           edgecolor=edge, linewidth=0.9 if lit else 0.6,
                                           zorder=10 - d))
                if r == 0:
                    top = np.array([base + dy, base + dy + dx, base + dy + dx + dz,
                                    base + dy + dz])
                    ax.add_patch(plt.Polygon(top, facecolor=TINT[ACCENT] if lit else "#EEF3FA",
                                             edgecolor=edge, linewidth=0.6, zorder=9 - d))
                if c == shape[2] - 1:
                    side = np.array([base + dx, base + dx + dz, base + dx + dz + dy,
                                     base + dx + dy])
                    ax.add_patch(plt.Polygon(side, facecolor="#DCE6F2",
                                             edgecolor="#B9C7DA", linewidth=0.6,
                                             zorder=9 - d))
    save(fig, "m02_cover.png")


def m02_timeline() -> None:
    timeline("m02_timeline.png", [
        (1946, "Monte Carlo\n(Ulam, ENIAC)", PURPLE),
        (1957, "Fortran stores\ncolumn-major", PRIMARY),
        (1962, "Iverson's APL:\nwhole-array maths", ACCENT),
        (1979, "BLAS", TEAL),
        (1992, "LAPACK", TEAL),
        (1995, "Numeric", PRIMARY),
        (2001, "Numarray", PRIMARY),
        (2006, "NumPy 1.0", ACCENT),
        (2016, "LIGO hears\ngravitational waves", PURPLE),
        (2019, "first image of\na black hole", PURPLE),
        (2024, "NumPy 2.0", ACCENT),
    ])


def m02_speed_bars() -> None:
    """Measured: four ways to sum a million integers."""
    n = 1_000_000
    py_list = list(range(n))
    arr = np.arange(n, dtype=np.int64)

    def loop(values):
        total = 0
        for v in values:
            total += v
        return total

    timings = [
        ("for loop over a list", best_of(loop, py_list), DANGER),
        ("sum() on a list", best_of(sum, py_list), DANGER),
        ("sum() on an ndarray", best_of(sum, arr), DANGER),
        ("arr.sum()", best_of(np.sum, arr), TEAL),
    ]
    fig, ax = plt.subplots(figsize=(FULL, 2.1))
    names = [t[0] for t in timings][::-1]
    values = [t[1] * 1e3 for t in timings][::-1]
    ax.barh(names, values, color=[t[2] for t in timings][::-1], height=0.58)
    ax.set_xscale("log")
    ax.set_xlim(min(values) * 0.5, max(values) * 6)
    clean_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("time to sum 1,000,000 integers (ms, log scale)")
    for k, v in enumerate(values):
        ax.text(v * 1.12, k, f"{v:.1f} ms", va="center", fontsize=7.6)
    save(fig, "m02_speed_bars.png")


def m02_shapes() -> None:
    fig, ax = canvas(6.3, 1.85)
    s = 0.36
    cell_grid(ax, 0.3, 1.5, np.array([[1, 2, 3]]), size=s)
    label(ax, 0.84, 1.68, "shape (3,)", size=8, weight="bold", family=MONO)
    label(ax, 0.84, 0.88, "1-D: no rows,\nno columns", size=7.2, color=SOFT_TEXT)

    cell_grid(ax, 2.75, 1.6, np.array([[1], [2], [3]]), size=s, color=TEAL)
    label(ax, 2.93, 1.75, "shape (3, 1)", size=8, weight="bold", family=MONO)
    label(ax, 3.55, 0.95, "a column", size=7.2, color=SOFT_TEXT, ha="left")

    cell_grid(ax, 4.75, 1.5, np.array([[1, 2, 3]]), size=s, color=ACCENT)
    label(ax, 5.29, 1.68, "shape (1, 3)", size=8, weight="bold", family=MONO)
    label(ax, 5.29, 0.88, "a row", size=7.2, color=SOFT_TEXT)
    label(ax, 3.15, 0.12, "same three numbers, three different behaviours under "
          "broadcasting and @", size=7.2, color=SOFT_TEXT)
    save(fig, "m02_shapes.png")


def m02_indexing() -> None:
    arr = np.arange(1, 13).reshape(3, 4)
    cases = [
        ("arr[0]", np.s_[0, :]), ("arr[:, 0]", np.s_[:, 0]),
        ("arr[1:, 2:]", np.s_[1:, 2:]), ("arr[::2]", np.s_[::2, :]),
    ]
    fig, ax = canvas(6.3, 1.7)
    s = 0.3
    for k, (title, sl) in enumerate(cases):
        mask = np.zeros_like(arr, dtype=bool)
        mask[sl] = True
        x0 = 0.12 + k * 1.58
        cell_grid(ax, x0, 1.3, arr, size=s, highlight=mask, color="#B9C2CE")
        label(ax, x0 + 0.6, 1.5, title, size=8, family=MONO, weight="bold")
    save(fig, "m02_indexing.png")


def m02_int_wrap() -> None:
    """int8 as an odometer: one step past 127 lands on -128."""
    fig, ax = canvas(FULL, 1.35)
    values = [123, 124, 125, 126, 127, -128, -127, -126, -125, -124]
    w = 0.5
    x0 = 0.6
    for k, v in enumerate(values):
        color = PRIMARY if v >= 0 else DANGER
        box(ax, x0 + k * (w + 0.04), 0.55, w, 0.38, str(v), color=color, size=7.6,
            family=MONO, radius=0.04)
    arrow(ax, (x0 + 4 * (w + 0.04) + w / 2, 0.96), (x0 + 5 * (w + 0.04) + w / 2, 0.96),
          color=ACCENT, rad=-0.6, lw=1.4)
    label(ax, x0 + 4.5 * (w + 0.04) + w / 2, 1.27, "+1", size=8.4, weight="bold",
          color=ACCENT)
    label(ax, 0.3, 0.74, "…", size=10, color=SOFT_TEXT)
    label(ax, x0 + 10 * (w + 0.04) + 0.1, 0.74, "…", size=10, color=SOFT_TEXT)
    label(ax, FULL / 2, 0.2, "int8 holds 256 values; past the largest it wraps to the "
          "smallest — silently", size=7.4, color=SOFT_TEXT)
    save(fig, "m02_int_wrap.png")


def m02_float_digits() -> None:
    fig, ax = plt.subplots(figsize=(FULL * 0.5, 2.6))
    names = ["float16", "float32", "float64"]
    digits = [np.finfo(t).precision for t in (np.float16, np.float32, np.float64)]
    bars = ax.bar(names, digits, color=[MUTED, ACCENT, PRIMARY], width=0.56)
    bars[0].set_color("#B0B4BA")
    clean_axes(ax)
    ax.set_ylabel("decimal digits you can trust")
    ax.set_ylim(0, 18)
    for k, (d, t) in enumerate(zip(digits, (np.float16, np.float32, np.float64),
                                   strict=True)):
        ax.text(k, d + 0.4, f"{d} digits\n{np.dtype(t).itemsize} bytes", ha="center",
                va="bottom", fontsize=7.4)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Precision is a budget", loc="left")
    save(fig, "m02_float_digits.png")


def m02_accumulation() -> None:
    """Measured: drift of a naive float32 running total of 0.1 versus np.sum."""
    checkpoints = np.unique(np.logspace(2, 6, 30).astype(int))
    naive_err = []
    total = np.float32(0.0)
    step = np.float32(0.1)
    count = 0
    for target in checkpoints:
        while count < target:
            total = np.float32(total + step)
            count += 1
        naive_err.append(abs(float(total) - target * 0.1) / (target * 0.1))
    pairwise_err = [abs(float(np.full(n, 0.1, dtype=np.float32).sum()) - n * 0.1) / (n * 0.1)
                    for n in checkpoints]
    f64_err = [abs(np.full(n, 0.1).sum() - n * 0.1) / (n * 0.1) + 1e-17 for n in checkpoints]

    fig, ax = plt.subplots(figsize=(FULL, 2.5))
    ax.plot(checkpoints, np.array(naive_err) * 100, color=DANGER)
    ax.plot(checkpoints, np.array(pairwise_err) * 100 + 1e-12, color=ACCENT)
    ax.plot(checkpoints, np.array(f64_err) * 100, color=TEAL)
    ax.set_xscale("log")
    ax.set_yscale("log")
    clean_axes(ax)
    ax.set_xlabel("number of 0.1s added")
    ax.set_ylabel("relative error (%)")
    ax.text(checkpoints[-1] * 0.9, naive_err[-1] * 100, "hand-written float32 loop  ",
            color=DANGER, ha="right", va="bottom", fontweight="bold", fontsize=7.8)
    ax.text(checkpoints[-1] * 0.9, pairwise_err[-1] * 100 * 0.05,
            "np.sum on float32 (pairwise)  ", color=ACCENT, ha="right", va="top",
            fontweight="bold", fontsize=7.8)
    ax.text(checkpoints[-1] * 0.9, f64_err[-1] * 100 * 1.6, "float64  ", color=TEAL,
            ha="right", va="bottom", fontweight="bold", fontsize=7.8)
    ax.set_title(f"After a million additions the naive float32 total is "
                 f"{naive_err[-1]:.1%} off", loc="left")
    save(fig, "m02_accumulation.png")


def m02_broadcast_grid() -> None:
    fig, ax = canvas(6.3, 1.95)
    s = 0.3
    col = np.arange(1, 5).reshape(4, 1)
    row = np.arange(1, 6).reshape(1, 5)
    cell_grid(ax, 0.2, 1.62, col, size=s, color=PRIMARY)
    # the stretched copies, ghosted
    for j in range(1, 5):
        for i in range(4):
            ax.add_patch(Rectangle((0.2 + j * s, 1.62 - (i + 1) * s), s, s,
                                   facecolor="white", edgecolor=PRIMARY, lw=0.5,
                                   linestyle=(0, (2, 2)), alpha=0.6))
            ax.text(0.2 + (j + 0.5) * s, 1.62 - (i + 0.5) * s, str(i + 1),
                    ha="center", va="center", fontsize=7, family=MONO, color="#A9B4C4")
    label(ax, 0.95, 1.8, "column (4, 1)", size=7.8, family=MONO, weight="bold",
          color=PRIMARY)
    label(ax, 2.0, 1.0, "×", size=13, color=SOFT_TEXT)
    cell_grid(ax, 2.35, 1.62, row, size=s, color=ACCENT)
    for i in range(1, 4):
        for j in range(5):
            ax.add_patch(Rectangle((2.35 + j * s, 1.62 - (i + 1) * s), s, s,
                                   facecolor="white", edgecolor=ACCENT, lw=0.5,
                                   linestyle=(0, (2, 2)), alpha=0.6))
            ax.text(2.35 + (j + 0.5) * s, 1.62 - (i + 0.5) * s, str(j + 1),
                    ha="center", va="center", fontsize=7, family=MONO, color="#D4AE8E")
    label(ax, 3.1, 1.8, "row (1, 5)", size=7.8, family=MONO, weight="bold", color=ACCENT)
    label(ax, 4.15, 1.0, "=", size=13, color=SOFT_TEXT)
    cell_grid(ax, 4.45, 1.62, col * row, size=s, color=TEAL)
    label(ax, 5.2, 1.8, "result (4, 5)", size=7.8, family=MONO, weight="bold", color=TEAL)
    label(ax, 3.15, 0.12, "dashed cells are never stored — NumPy stretches size-1 "
          "dimensions virtually", size=7.2, color=SOFT_TEXT)
    save(fig, "m02_broadcast_grid.png")


def m02_silent_bug() -> None:
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    actual = np.array([1.1, 2.1, 2.9, 4.2])
    errors = pred - actual[:, None]
    fig, ax = canvas(6.3, 2.1)
    s = 0.36
    mask = np.eye(4, dtype=bool)
    cell_grid(ax, 3.4, 1.85, errors.round(1), size=s, highlight=mask, hl_color=TEAL,
              color=DANGER, fmt="{:+.1f}", text_size=6.8)
    label(ax, 4.12, 1.98, "errors: shape (4, 4)", size=7.8, family=MONO, weight="bold",
          color=DANGER)
    label(ax, 0.1, 1.6, "predictions   shape (4,)", size=7.8, family=MONO, ha="left")
    label(ax, 0.1, 1.3, "actuals       shape (4, 1)", size=7.8, family=MONO, ha="left",
          color=DANGER)
    label(ax, 0.1, 0.95, "predictions - actuals", size=7.8, family=MONO, ha="left")
    arrow(ax, (2.4, 0.95), (3.3, 1.15), color=MUTED)
    label(ax, 0.1, 0.45, "Only the 4 teal cells are real errors.\nThe other 12 are "
          "predictions compared with\nthe wrong rows — and the MSE averages all 16.",
          size=7.2, color=SOFT_TEXT, ha="left")
    save(fig, "m02_silent_bug.png")


def m02_memory_trap() -> None:
    n = np.logspace(2, 5.2, 50)
    result_bytes = n**2 * 8
    fig, ax = plt.subplots(figsize=(FULL * 0.62, 2.5))
    ax.plot(n, n * 8, color=PRIMARY)
    ax.plot(n, result_bytes, color=DANGER)
    ax.set_xscale("log")
    ax.set_yscale("log")
    clean_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    for b, txt in [(1024**2, "1 MiB"), (1024**3, "1 GiB"), (64 * 1024**3, "64 GiB of RAM")]:
        ax.axhline(b, color=MUTED, lw=0.7, ls=":")
        ax.text(n[0] * 1.1, b * 1.15, txt, fontsize=6.8, color=SOFT_TEXT)
    ax.set_xlabel("length of each input vector")
    ax.set_ylabel("bytes")
    ax.text(n[-1], result_bytes[-1] * 0.002, "a[:, None] - b[None, :]",
            color=DANGER, ha="right", va="top", fontweight="bold", fontsize=7.6,
            family=MONO)
    ax.text(n[-1] * 0.8, n[-1] * 8 * 3, "each input  ", color=PRIMARY, ha="right",
            fontweight="bold", fontsize=7.6)
    ax.set_title("Inputs grow like n; the result grows like n²", loc="left")
    save(fig, "m02_memory_trap.png")


def m02_mask() -> None:
    values = np.array([12, 45, 7, 88, 23, 61, 4, 39])
    mask = values > 30
    fig, ax = canvas(6.3, 1.75)
    s = 0.42
    x0 = 1.7
    label(ax, 1.55, 1.42, "values", size=7.8, family=MONO, ha="right")
    cell_grid(ax, x0, 1.63, values[None, :], size=s, highlight=mask[None, :],
              color="#B9C2CE")
    label(ax, 1.55, 0.92, "values > 30", size=7.8, family=MONO, ha="right")
    for j, m in enumerate(mask):
        ax.add_patch(Rectangle((x0 + j * s, 0.7), s, s * 0.95,
                               facecolor=TINT[TEAL] if m else "white",
                               edgecolor=TEAL if m else "#C8CBD0", lw=0.8))
        label(ax, x0 + (j + 0.5) * s, 0.9, "T" if m else "F", size=7.4, family=MONO,
              color=TEAL if m else SOFT_TEXT)
    label(ax, 3.4, 0.35, f"values[values > 30]  →  {values[mask].tolist()}     "
          f"mask.sum() → {mask.sum()}     mask.mean() → {mask.mean():.1%}",
          size=7.4, family=MONO)
    save(fig, "m02_mask.png")


def m02_tukey() -> None:
    rng = np.random.default_rng(42)
    data = rng.lognormal(3, 0.6, 2000)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    fig, (ax_h, ax_b) = plt.subplots(2, 1, figsize=(FULL, 2.7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax_h.hist(data, bins=60, color=PRIMARY, edgecolor="white", linewidth=0.3)
    for f in (hi,):
        ax_h.axvline(f, color=DANGER, ls="--", lw=1.1)
        ax_b.axvline(f, color=DANGER, ls="--", lw=1.1)
    ax_h.text(hi, ax_h.get_ylim()[1] * 0.92, "  Q3 + 1.5 × IQR\n  anything right of "
              f"here is flagged ({(data > hi).mean():.1%})", color=DANGER, fontsize=7.4,
              va="top")
    clean_axes(ax_h)
    ax_h.set_ylabel("count")
    ax_b.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=True,
                 boxprops={"facecolor": TINT[PRIMARY], "edgecolor": PRIMARY},
                 medianprops={"color": TEXT}, whiskerprops={"color": PRIMARY},
                 capprops={"color": PRIMARY},
                 flierprops={"marker": "o", "markersize": 2.5,
                             "markerfacecolor": DANGER, "markeredgecolor": "none"})
    ax_b.set_yticks([])
    ax_b.spines["left"].set_visible(False)
    ax_b.grid(False)
    ax_b.set_xlabel("value (right-skewed data)")
    ax_h.set_title("Tukey's fences: the box is Q1–Q3; whiskers stop at 1.5 × IQR",
                   loc="left")
    save(fig, "m02_tukey.png")


def m02_cosine() -> None:
    fig, ax = plt.subplots(figsize=(FULL * 0.5, 2.7))
    vecs = [((1, 1), "reference", PRIMARY), ((3.2, 3.2), "same direction  cos = 1",
            TEAL), ((-1, 1), "orthogonal  cos = 0", ACCENT),
            ((-1, -1), "opposite  cos = −1", DANGER)]
    for (x, y), _, color in vecs[1:] + vecs[:1]:
        ax.annotate("", xy=(x, y), xytext=(0, 0), zorder=4 if color == PRIMARY else 3,
                    arrowprops={"arrowstyle": "-|>", "color": color,
                                "lw": 2.6 if color == PRIMARY else 1.6})
    ax.text(3.3, 3.0, "5× longer,\ncos = 1", color=TEAL, fontsize=7.4, ha="left", va="top")
    ax.text(-1.05, 1.1, "cos = 0", color=ACCENT, fontsize=7.4, ha="right")
    ax.text(-1.05, -1.1, "cos = −1", color=DANGER, fontsize=7.4, ha="right", va="top")
    ax.text(1.15, 0.7, "reference", color=PRIMARY, fontsize=7.4, ha="left",
            fontweight="bold")
    ax.set_xlim(-2.4, 4.6)
    ax.set_ylim(-1.8, 4.0)
    ax.set_aspect("equal")
    ax.axhline(0, color="#D0D3D8", lw=0.8)
    ax.axvline(0, color="#D0D3D8", lw=0.8)
    ax.axis("off")
    ax.set_title("Only the angle counts", loc="left")
    save(fig, "m02_cosine.png")


def m02_least_squares() -> None:
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(0, 10, 18))
    y = 1.2 + 0.8 * x + rng.normal(0, 1.1, x.size)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    fig, ax = plt.subplots(figsize=(FULL * 0.5, 2.7))
    for xi, yi, fi in zip(x, y, fit, strict=True):
        ax.plot([xi, xi], [yi, fi], color=DANGER, lw=0.9, alpha=0.8)
    ax.scatter(x, y, s=18, color=PRIMARY, zorder=3, edgecolors="white", linewidths=0.6)
    xs = np.linspace(0, 10, 2)
    ax.plot(xs, beta[0] + beta[1] * xs, color=TEXT, lw=1.6)
    clean_axes(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.text(0.2, 9.4, f"ŷ = {beta[0]:.2f} + {beta[1]:.2f}x", fontsize=7.8, color=TEXT)
    ax.text(9.9, -0.6, "red segments = residuals;\nleast squares minimises\n"
            "the sum of their squares", fontsize=7, color=DANGER, ha="right",
            va="bottom")
    ax.set_title("The line that best fits", loc="left")
    save(fig, "m02_least_squares.png")


def m02_condition() -> None:
    n = 12
    hilbert = 1.0 / (np.arange(1, n + 1)[:, None] + np.arange(n))
    mats = {"identity": np.eye(4), "well conditioned": np.array([[4.0, 1.0], [1.0, 3.0]]),
            "nearly singular": np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]]),
            "Hilbert 12×12": hilbert}
    lost = [np.log10(np.linalg.cond(m)) for m in mats.values()]
    fig, ax = plt.subplots(figsize=(FULL, 1.9))
    ax.barh(list(mats)[::-1], lost[::-1], color=[DANGER, DANGER, PRIMARY, PRIMARY],
            height=0.55)
    ax.axvline(15.95, color=TEXT, lw=1, ls="--")
    ax.text(15.8, 3.35, "float64 holds ≈16 digits", ha="right", fontsize=7.2, color=TEXT)
    for k, v in enumerate(lost[::-1]):
        ax.text(v + 0.2, k, f"{v:.1f}", va="center", fontsize=7.6)
    clean_axes(ax, grid_axis="x")
    ax.set_xlim(0, 18.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("significant digits lost  ≈ log10(cond(A))")
    save(fig, "m02_condition.png")


def m02_pca() -> None:
    rng = np.random.default_rng(42)
    pts = rng.multivariate_normal([0, 0], [[3.0, 2.4], [2.4, 2.5]], size=500)
    c = pts - pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(c.T))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.7))
    a.scatter(c[:, 0], c[:, 1], s=5, alpha=0.35, color=PRIMARY, edgecolors="none")
    for i, color in enumerate([DANGER, ACCENT]):
        d = vecs[:, i] * np.sqrt(vals[i]) * 2.4
        a.annotate("", xy=d, xytext=(0, 0),
                   arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2})
        a.text(d[0] * 1.12, d[1] * 1.12, f"PC{i + 1}  {vals[i] / vals.sum():.0%}",
               color=color, fontsize=7.6, fontweight="bold",
               ha="left" if d[0] > 0 else "right", va="center",
               bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                     "edgecolor": "none", "alpha": 0.85})
    for axis in (a,):
        axis.set_xlim(-7, 7)
        axis.set_ylim(-5.5, 5.5)
    a.set_aspect("equal")
    clean_axes(a)
    a.set_title("Eigenvectors of the covariance", loc="left")
    p = c @ vecs
    b.scatter(p[:, 0], p[:, 1], s=5, alpha=0.35, color=TEAL, edgecolors="none")
    b.set_xlim(-7, 7)
    b.set_ylim(-5.5, 5.5)
    b.set_aspect("equal")
    clean_axes(b)
    b.set_xlabel("PC1")
    b.set_ylabel("PC2")
    b.set_title("Same data in PC coordinates", loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "m02_pca.png")


def m02_svd() -> None:
    rng = np.random.default_rng(42)
    M = rng.random((60, 3)) @ rng.random((3, 40)) + rng.normal(0, 0.01, (60, 40))
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    ks = np.arange(1, 11)
    errs = [np.linalg.norm(M - U[:, :k] @ np.diag(s[:k]) @ Vt[:k]) / np.linalg.norm(M)
            for k in ks]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.35))
    a.bar(ks, s[:10], color=[PRIMARY] * 3 + ["#B0B4BA"] * 7, width=0.62)
    a.set_yscale("log")
    clean_axes(a)
    a.set_xticks(ks)
    a.set_xlabel("component")
    a.set_title("Singular values drop after 3", loc="left")
    a.text(3.6, s[2], "← true rank", fontsize=7.4, color=PRIMARY, va="center")
    b.plot(ks, np.array(errs) * 100, marker="o", color=TEAL)
    clean_axes(b)
    b.set_xticks(ks)
    b.set_xlabel("rank k kept")
    b.set_ylabel("relative error (%)")
    b.set_title("Rank 3 already reconstructs it", loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "m02_svd.png")


def m02_distributions() -> None:
    rng = np.random.default_rng(42)
    n = 10_000
    dists = [("uniform", rng.random(n), "any value equally likely"),
             ("normal", rng.normal(0, 1, n), "sum of many small effects"),
             ("lognormal", rng.lognormal(0, 0.5, n), "product of effects: spend"),
             ("exponential", rng.exponential(1, n), "waiting time"),
             ("binomial (20, 0.3)", rng.binomial(20, 0.3, n), "successes in n trials"),
             ("poisson (3)", rng.poisson(3, n), "events per window")]
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.6))
    for ax, (name, sample, note) in zip(axes.ravel(), dists, strict=True):
        discrete = name.startswith(("binomial", "poisson"))
        bins = np.arange(sample.min(), sample.max() + 2) - 0.5 if discrete else 40
        ax.hist(sample, bins=bins, color=PRIMARY, edgecolor="white", linewidth=0.3)
        ax.text(0, 1.2, name, transform=ax.transAxes, fontsize=8.2, fontweight="bold",
                va="bottom")
        ax.text(0, 1.04, note, transform=ax.transAxes, fontsize=6.9, color=SOFT_TEXT,
                va="bottom")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.tick_params(labelsize=6.8)
    fig.tight_layout(h_pad=2.2)
    save(fig, "m02_distributions.png")


def m02_monte_carlo() -> None:
    rng = np.random.default_rng(1)
    pts = rng.random((2500, 2))
    inside = (pts**2).sum(axis=1) <= 1
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.8))
    a.scatter(*pts[inside].T, s=2.5, color=TEAL, alpha=0.6, edgecolors="none")
    a.scatter(*pts[~inside].T, s=2.5, color=ACCENT, alpha=0.6, edgecolors="none")
    t = np.linspace(0, np.pi / 2, 100)
    a.plot(np.cos(t), np.sin(t), color=TEXT, lw=1.2)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.grid(False)
    a.set_title(f"π ≈ 4 × {inside.mean():.3f} = {4 * inside.mean():.3f}", loc="left")
    halo = {"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none"}
    a.text(0.3, 0.35, "inside", color=TEAL, fontweight="bold", fontsize=7.8, bbox=halo)
    a.text(0.8, 0.9, "outside", color=ACCENT, fontweight="bold", fontsize=7.8, bbox=halo)

    rng2 = np.random.default_rng(2)
    nmax = 200_000
    d = rng2.random((nmax, 2))
    run = 4 * np.cumsum((d**2).sum(axis=1) <= 1) / np.arange(1, nmax + 1)
    at = np.unique(np.logspace(1, np.log10(nmax), 300).astype(int))
    env = 4 * np.sqrt(np.pi / 4 * (1 - np.pi / 4)) / np.sqrt(at)
    b.fill_between(at, np.pi - 2 * env, np.pi + 2 * env, color=MUTED, alpha=0.18,
                   linewidth=0)
    b.plot(at, run[at - 1], color=PRIMARY, lw=1.1)
    b.axhline(np.pi, color=DANGER, ls="--", lw=1)
    b.set_xscale("log")
    b.set_ylim(2.85, 3.45)
    clean_axes(b)
    b.set_xlabel("number of points")
    b.text(12, 2.88, "grey band: ±2 standard errors, shrinking like 1/√n",
           fontsize=6.8, color=SOFT_TEXT)
    b.text(at[-1], np.pi + 0.01, "true π", color=DANGER, ha="right", va="bottom",
           fontsize=7.4)
    b.set_title("Convergence: slow, but predictable", loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "m02_monte_carlo.png")


def m02_bootstrap() -> None:
    fig, ax = canvas(6.3, 2.3)
    palette = [PRIMARY, ACCENT, TEAL, PURPLE, DANGER, "#B0B4BA"]
    sample = list(range(6))
    rng = np.random.default_rng(5)

    def row(x, y, idx, title):
        label(ax, x - 0.12, y + 0.13, title, size=7.4, ha="right", color=SOFT_TEXT)
        for k, i in enumerate(idx):
            ax.add_patch(Circle((x + k * 0.32 + 0.13, y + 0.13), 0.12,
                                facecolor=palette[i], edgecolor="white", lw=0.8))

    row(1.55, 1.9, sample, "observed sample")
    for r in range(3):
        idx = rng.integers(0, 6, 6)
        row(1.55, 1.25 - r * 0.45, idx, f"resample {r + 1}")
        label(ax, 3.65, 1.38 - r * 0.45, f"→ statistic #{r + 1}", size=7.2, ha="left",
              color=TEXT)
    label(ax, 1.55, 0.02, "drawn WITH replacement: some points repeat, some vanish",
          size=7, color=SOFT_TEXT, ha="left")
    label(ax, 5.0, 1.95, "repeat 10,000×,\nthen take the 2.5th and\n97.5th percentiles",
          size=7.2, color=PURPLE, weight="bold", ha="left", va="top")
    save(fig, "m02_bootstrap.png")


def m02_seed_streams() -> None:
    fig, ax = canvas(6.3, 2.15)
    label(ax, 1.45, 2.03, "Legacy: one hidden global state", size=8.2, weight="bold",
          color=DANGER)
    box(ax, 0.75, 1.15, 1.4, 0.48, "np.random\nglobal state", color=DANGER, size=7.2)
    for k, who in enumerate(["your code", "a library", "a colleague"]):
        x = 0.05 + k * 1.0
        box(ax, x, 0.2, 0.85, 0.36, who, color="#C8CBD0", fill="white", size=6.8)
        arrow(ax, (x + 0.42, 0.58), (1.45, 1.13), color=DANGER, lw=0.9)
    ax.plot([3.15, 3.15], [0.1, 2.1], color=GRID, lw=1)
    label(ax, 4.75, 2.03, "Modern: SeedSequence(42).spawn(4)", size=8.2, weight="bold",
          color=TEAL)
    box(ax, 3.95, 1.2, 1.6, 0.42, "SeedSequence(42)", color=TEAL, size=6.8, family=MONO)
    for k in range(4):
        x = 3.35 + k * 0.72
        box(ax, x, 0.25, 0.62, 0.4, f"rng {k}", color=PRIMARY, size=6.8, family=MONO)
        arrow(ax, (4.75, 1.18), (x + 0.31, 0.67), color=TEAL, lw=0.9)
    label(ax, 4.75, 0.06, "independent, reproducible streams", size=7, color=SOFT_TEXT)
    save(fig, "m02_seed_streams.png")


def m02_strides() -> None:
    fig, ax = canvas(6.3, 2.2)
    arr = np.arange(12).reshape(3, 4)
    cell_grid(ax, 0.1, 1.95, arr, size=0.36, color=PRIMARY)
    label(ax, 0.82, 0.62, "what you see:\nshape (3, 4)", size=7.4, color=SOFT_TEXT)
    s = 0.36
    x0, y0 = 2.2, 1.35
    for k in range(12):
        row = k // 4
        ax.add_patch(Rectangle((x0 + k * s, y0), s, s,
                               facecolor=TINT[[PRIMARY, ACCENT, TEAL][row]],
                               edgecolor="#9AA7B8", lw=0.7))
        label(ax, x0 + (k + 0.5) * s, y0 + s / 2, str(k), size=7.2, family=MONO)
    label(ax, x0, y0 + 0.62, "what is stored: one flat buffer of int64 (8 bytes each)",
          size=7.4, color=SOFT_TEXT, ha="left")
    arrow(ax, (x0 + 0.5 * s, y0 - 0.05), (x0 + 1.5 * s, y0 - 0.05), color=ACCENT,
          rad=0.6, lw=1.1)
    label(ax, x0 + s, y0 - 0.35, "next column:\n+8 bytes", size=6.9, color=ACCENT)
    arrow(ax, (x0 + 0.5 * s, y0 - 0.55), (x0 + 4.5 * s, y0 - 0.55), color=TEAL,
          rad=0.35, lw=1.1)
    label(ax, x0 + 2.5 * s, y0 - 1.0, "next row: +32 bytes", size=6.9, color=TEAL)
    label(ax, x0 + 9 * s, 0.4, "strides = (32, 8)\narr.T → strides (8, 32)\n"
          "same buffer, no copy", size=7.4, family=MONO, color=TEXT)
    save(fig, "m02_strides.png")


def m02_c_vs_f() -> None:
    fig, ax = canvas(6.3, 1.9)
    s = 0.34
    arr = np.arange(1, 13).reshape(3, 4)
    for k, (title, order, color) in enumerate([("C order (row-major) — NumPy default",
                                                "C", PRIMARY),
                                               ("Fortran order (column-major)", "F",
                                                ACCENT)]):
        x0 = 0.15 + k * 3.2
        cell_grid(ax, x0, 1.6, arr, size=s, color="#B9C2CE")
        label(ax, x0, 1.78, title, size=7.6, weight="bold", color=color, ha="left")
        flat = arr.ravel(order=order)
        pos = {v: np.argwhere(arr == v)[0] for v in flat}
        pts = [(x0 + (pos[v][1] + 0.5) * s, 1.6 - (pos[v][0] + 0.5) * s) for v in flat]
        xs, ys = zip(*pts, strict=True)
        ax.plot(xs, ys, color=color, lw=1.0, alpha=0.45, zorder=3)
        ax.scatter(xs[0], ys[0], color=color, s=60, zorder=3, alpha=0.35)
        label(ax, x0 + 1.55, 0.95, "memory order:\n" + " ".join(map(str, flat[:7])) + " …",
              size=6.9, family=MONO, color=SOFT_TEXT, ha="left")
    label(ax, 3.15, 0.1, "reading along the stored order uses every byte of each "
          "64-byte cache line", size=7.2, color=SOFT_TEXT)
    save(fig, "m02_c_vs_f.png")


def m02_temporaries() -> None:
    fig, ax = canvas(6.3, 1.85)
    label(ax, 0.05, 1.72, "y = (x * 2 + 1) ** 2", size=8, family=MONO, ha="left",
          weight="bold")
    for k, (text, color) in enumerate([("x", MUTED), ("x * 2", DANGER),
                                       ("… + 1", DANGER), ("… ** 2", PRIMARY)]):
        box(ax, 0.05 + k * 0.78, 0.95, 0.68, 0.48, text, color=color, size=7,
            family=MONO, text_color=TEXT)
    label(ax, 1.6, 0.68, "two throw-away arrays, each as big as x", size=7,
          color=DANGER)
    ax.plot([3.25, 3.25], [0.3, 1.8], color=GRID, lw=1)
    label(ax, 3.45, 1.72, "np.multiply(x, 2, out=buf) …", size=8, family=MONO,
          ha="left", weight="bold")
    box(ax, 3.45, 0.95, 0.68, 0.48, "x", color=MUTED, size=7, family=MONO,
        text_color=TEXT)
    box(ax, 4.3, 0.95, 0.9, 0.48, "buf", color=TEAL, size=7, family=MONO)
    label(ax, 4.75, 0.68, "one buffer, reused for every step", size=7, color=TEAL)
    label(ax, 3.15, 0.15, "peak memory: 4 arrays  vs  2 arrays", size=7.4,
          color=SOFT_TEXT)
    save(fig, "m02_temporaries.png")


def m02_escalation() -> None:
    steps = [("Python loop over elements", "vectorise it", PRIMARY),
             ("genuinely sequential loop", "a SciPy routine (lfilter, cumsum…)", PRIMARY),
             ("no library exists", "Numba @njit", TEAL),
             ("peak memory is the limit", "in-place ops, chunks, np.memmap", ACCENT),
             ("mostly zeros", "scipy.sparse", ACCENT),
             ("bigger than one machine", "Dask / Spark (Module 20)", PURPLE)]
    fig, ax = canvas(6.3, 2.6)
    for k, (symptom, fix, color) in enumerate(steps):
        y = 2.2 - k * 0.4
        x = 0.05 + k * 0.15
        box(ax, x, y, 2.3, 0.32, symptom, color="#C8CBD0", fill="white", size=7,
            ha="left")
        arrow(ax, (x + 2.35, y + 0.16), (x + 2.6, y + 0.16), color=MUTED, scale=8)
        box(ax, x + 2.65, y, 2.45, 0.32, fix, color=color, size=7, ha="left")
    save(fig, "m02_escalation.png")


def m02_kmeans() -> None:
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=42)
    def run(seed):
        c = X[np.random.default_rng(seed).choice(len(X), 4, replace=False)]
        for _ in range(20):
            lab = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            c = np.array([X[lab == k].mean(0) if (lab == k).any() else c[k]
                          for k in range(4)])
        return ((X - c[lab]) ** 2).sum()

    # A start that is visibly poor but still reaches the global optimum, so the
    # three panels tell the story of the algorithm rather than of bad luck.
    inertias = {seed: run(seed) for seed in range(40)}
    best = min(inertias.values())
    seed = next(s for s, v in inertias.items() if v < best * 1.001 and s > 0)
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), 4, replace=False)]
    palette = [PRIMARY, ACCENT, TEAL, PURPLE]
    snapshots = []
    for it in range(8):
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        labels_ = d.argmin(1)
        if it in (0, 1, 7):
            snapshots.append((it, centroids.copy(), labels_.copy()))
        centroids = np.array([X[labels_ == k].mean(0) for k in range(4)])
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2))
    titles = ["start: random centroids", "after one update", "converged"]
    for ax, (_, c, lab), title in zip(axes, snapshots, titles, strict=True):
        for k in range(4):
            ax.scatter(*X[lab == k].T, s=3, color=palette[k], alpha=0.55, edgecolors="none")
        ax.scatter(*c.T, s=70, marker="X", color=TEXT, edgecolors="white", linewidths=1)
        ax.set_title(title, loc="left", fontsize=8.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout(w_pad=0.8)
    save(fig, "m02_kmeans.png")


FIGURES: dict[str, Callable[[], None]] = {
    f.__name__: f for f in [
        m01_cover, m01_timeline, m01_time_split, m01_reproducibility, m01_hidden_state,
        m01_promotion_loop, m01_float_gap, m01_shallow_deep, m01_hash_table,
        m01_set_vs_list, m01_comprehension, m01_generator_memory, m01_lazy_pipeline,
        m01_decorator, m01_mutable_default, m01_estimator, m01_composition,
        m01_row_vs_column, m01_format_benchmark, m01_try_flow, m01_log_levels,
        m01_pipeline,
        m02_cover, m02_timeline, m02_speed_bars, m02_shapes, m02_indexing,
        m02_int_wrap, m02_float_digits, m02_accumulation, m02_broadcast_grid,
        m02_silent_bug, m02_memory_trap, m02_mask, m02_tukey, m02_cosine,
        m02_least_squares, m02_condition, m02_pca, m02_svd, m02_distributions,
        m02_monte_carlo, m02_bootstrap, m02_seed_streams, m02_strides, m02_c_vs_f,
        m02_temporaries, m02_escalation, m02_kmeans,
    ]
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="name prefix filter, e.g. m01 or m02_pca")
    args = parser.parse_args()

    setup_book_style()
    selected = {k: f for k, f in FIGURES.items() if not args.only or k.startswith(args.only)}
    print(f"Generating {len(selected)} book figure(s) into "
          f"{OUTPUT_DIR.relative_to(REPO_ROOT)}/\n")
    for make in selected.values():
        make()
    print(f"\nDone — {len(selected)} figure(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
