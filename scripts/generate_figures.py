"""Generate the conceptual diagrams used by the Module 01 notebooks.

Diagrams are *code*, not binary assets dropped into the repo: regenerate them
any time the palette or wording changes.

    .venv/bin/python scripts/generate_figures.py

Every figure is written to ``assets/images/`` as PNG at 200 dpi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from dstoolkit.plotting import COLORS, setup_plot_style

# Resolve paths relative to the repository root so the script works from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "assets" / "images"

DPI = 200


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    facecolor: str = "#FFFFFF",
    edgecolor: str = COLORS["primary"],
    textcolor: str | None = None,
    fontsize: float = 10,
    fontweight: str = "normal",
    linewidth: float = 1.6,
    zorder: int = 3,
) -> tuple[float, float]:
    """Draw a rounded rectangle with centred text. Returns its centre point."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)

    cx, cy = x + width / 2, y + height / 2
    ax.text(
        cx,
        cy,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor or COLORS["text"],
        zorder=zorder + 1,
        linespacing=1.5,
    )
    return cx, cy


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
    style: str = "-|>",
    connection: str = "arc3,rad=0",
    linewidth: float = 1.6,
    linestyle: str = "-",
    zorder: int = 2,
) -> None:
    """Draw an arrow between two points."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            connectionstyle=connection,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            mutation_scale=16,
            zorder=zorder,
        )
    )


def arc_arrow(
    ax: plt.Axes,
    radius: float,
    theta_start_deg: float,
    theta_end_deg: float,
    *,
    color: str = "#B8B8B8",
    linewidth: float = 1.8,
) -> None:
    """Draw an arrow that follows a circular arc of the given radius.

    Built from an explicit polyline rather than a bezier ``connectionstyle`` so
    the curve genuinely tracks the circle and the head always points along the
    direction of travel.
    """
    thetas = np.linspace(np.deg2rad(theta_start_deg), np.deg2rad(theta_end_deg), 40)
    xs, ys = radius * np.cos(thetas), radius * np.sin(thetas)

    # Shaft: everything but the final segment, which the arrowhead replaces.
    ax.plot(xs[:-1], ys[:-1], color=color, linewidth=linewidth, zorder=2,
            solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(
            (xs[-2], ys[-2]),
            (xs[-1], ys[-1]),
            arrowstyle="-|>",
            color=color,
            linewidth=linewidth,
            mutation_scale=18,
            zorder=2,
        )
    )


def elbow_connector(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    bus_y: float,
    *,
    color: str = "#C8C8C8",
    linewidth: float = 1.5,
) -> None:
    """Draw an orthogonal parent→child connector via a horizontal bus line.

    This is the standard tree/org-chart connector: down from the parent, across
    at ``bus_y``, then down into the child.
    """
    x0, y0 = start
    x1, y1 = end
    ax.plot([x0, x0], [y0, bus_y], color=color, linewidth=linewidth, zorder=1)
    ax.plot([x0, x1], [bus_y, bus_y], color=color, linewidth=linewidth, zorder=1)
    ax.add_patch(
        FancyArrowPatch(
            (x1, bus_y),
            (x1, y1),
            arrowstyle="-|>",
            color=color,
            linewidth=linewidth,
            mutation_scale=14,
            zorder=1,
        )
    )


def blank_axes(figsize: tuple[float, float], xlim=(0, 10), ylim=(0, 10)):
    """Create a figure with a clean, unit-scaled drawing surface."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure to the assets directory and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔ {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 1 — The data science lifecycle
# ---------------------------------------------------------------------------
def fig_ds_lifecycle() -> None:
    """The loop a data science project actually runs in."""
    fig, ax = blank_axes((9, 9), xlim=(-5.9, 5.9), ylim=(-5.9, 5.9))

    stages = [
        ("1. Business\nUnderstanding", COLORS["primary"]),
        ("2. Data\nCollection", COLORS["primary"]),
        ("3. Data Cleaning\n& Preparation", COLORS["accent"]),
        ("4. Exploratory\nData Analysis", COLORS["accent"]),
        ("5. Feature\nEngineering", COLORS["accent"]),
        ("6. Modelling", COLORS["teal"]),
        ("7. Evaluation", COLORS["teal"]),
        ("8. Deployment\n& Monitoring", COLORS["purple"]),
    ]

    n = len(stages)
    radius = 4.35
    # Start at the top (90°) and lay stages out clockwise, the direction people
    # instinctively read a cycle.
    angles = np.linspace(90, 90 - 360, n, endpoint=False)

    for (label, color), angle_deg in zip(stages, angles, strict=True):
        rad = np.deg2rad(angle_deg)
        cx, cy = radius * np.cos(rad), radius * np.sin(rad)

        w, h = 2.05, 1.05
        rounded_box(
            ax,
            (cx - w / 2, cy - h / 2),
            w,
            h,
            label,
            edgecolor=color,
            facecolor="white",
            fontsize=8.5,
            fontweight="bold",
            textcolor=color,
        )

    # Arcs connecting consecutive stages, drawn on the same circle the boxes sit
    # on. `pad` is the angular gap left either side so the arc clears each box.
    # 15° clears the widest box (the ones at top/bottom subtend ~13.3°).
    pad = 15.0
    step = 360.0 / n
    for angle_deg in angles:
        arc_arrow(ax, radius, angle_deg - pad, angle_deg - step + pad)

    # Centre annotation: the point students most often miss.
    ax.text(
        0,
        0.55,
        "It is a LOOP,\nnot a pipeline",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=COLORS["text"],
        linespacing=1.4,
    )
    ax.text(
        0,
        -0.75,
        "Findings at any stage send\nyou back to an earlier one.\n"
        "Expect 60–80% of your time\nin stages 2–5.",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["muted"],
        linespacing=1.6,
    )

    ax.set_title(
        "The Data Science Lifecycle",
        fontsize=15,
        fontweight="bold",
        loc="center",
        pad=18,
    )
    save(fig, "ds_lifecycle.png")


# ---------------------------------------------------------------------------
# Figure 2 — The full-stack course map
# ---------------------------------------------------------------------------
def fig_stack_map() -> None:
    """The five layers of the data science stack, mapped to course modules."""
    fig, ax = blank_axes((11, 7.2), xlim=(0, 11), ylim=(0, 7.2))

    layers = [
        (
            "COMMUNICATION",
            "Storytelling · dashboards · stakeholder readouts · experiment write-ups",
            "Module 24",
            COLORS["purple"],
        ),
        (
            "PRODUCTION  (MLOps)",
            "MLflow · FastAPI · Docker · CI/CD · drift monitoring · model registry",
            "Modules 21–23",
            COLORS["danger"],
        ),
        (
            "MODELLING",
            "scikit-learn · XGBoost · PyTorch · statsmodels · causal inference",
            "Modules 10–19",
            COLORS["teal"],
        ),
        (
            "ANALYSIS",
            "NumPy · pandas · matplotlib · seaborn · probability · inference",
            "Modules 02–09",
            COLORS["accent"],
        ),
        (
            "FOUNDATIONS",
            "Python · data structures · OOP · SQL · Git · testing · environments",
            "Module 01",
            COLORS["primary"],
        ),
    ]

    box_h = 1.12
    gap = 0.22
    y = 0.45

    for title, subtitle, modules, color in reversed(layers):
        # Main layer band.
        rounded_box(
            ax,
            (0.35, y),
            8.05,
            box_h,
            "",
            facecolor=color + "14",  # 8% alpha via hex suffix
            edgecolor=color,
            linewidth=2.0,
        )
        ax.text(
            0.72, y + box_h * 0.66, title,
            fontsize=11, fontweight="bold", color=color, va="center",
        )
        ax.text(
            0.72, y + box_h * 0.28, subtitle,
            fontsize=8.5, color=COLORS["text"], va="center",
        )

        # Module badge on the right.
        rounded_box(
            ax, (8.62, y + 0.24), 2.0, box_h - 0.48, modules,
            facecolor=color, edgecolor=color, textcolor="white",
            fontsize=9, fontweight="bold",
        )
        y += box_h + gap

    # Upward arrow showing progression through the stack.
    arrow(
        ax, (0.16, 0.5), (0.16, y - gap - 0.05),
        color=COLORS["muted"], linewidth=2.0,
    )
    ax.text(
        -0.02, (y - gap) / 2, "increasing scope & seniority",
        rotation=90, ha="center", va="center",
        fontsize=8.5, color=COLORS["muted"], style="italic",
    )

    ax.set_title(
        "The Full-Stack Data Science Skill Map",
        fontsize=15, fontweight="bold", loc="center", pad=16,
    )
    save(fig, "course_stack_map.png")


# ---------------------------------------------------------------------------
# Figure 3 — Names, objects and references
# ---------------------------------------------------------------------------
def fig_names_and_objects() -> None:
    """Python variables are labels bound to objects, not boxes holding values."""
    fig, ax = blank_axes((10, 5.2), xlim=(0, 10), ylim=(0, 5.2))

    # --- The wrong mental model (left) ---------------------------------
    ax.text(
        2.3, 4.75, "✘  The C / Java mental model",
        ha="center", fontsize=11, fontweight="bold", color=COLORS["danger"],
    )
    ax.text(
        2.3, 4.32, "a variable is a box that holds a value",
        ha="center", fontsize=8.5, color=COLORS["muted"], style="italic",
    )

    rounded_box(
        ax, (0.7, 2.9), 1.4, 0.9, "a\n42",
        edgecolor=COLORS["danger"], fontsize=10, fontweight="bold",
    )
    rounded_box(
        ax, (2.5, 2.9), 1.4, 0.9, "b\n42",
        edgecolor=COLORS["danger"], fontsize=10, fontweight="bold",
    )
    ax.text(
        2.3, 2.35, "two boxes, two independent copies",
        ha="center", fontsize=8.5, color=COLORS["danger"],
    )

    # --- The correct model (right) -------------------------------------
    ax.text(
        7.2, 4.75, "✔  How Python actually works",
        ha="center", fontsize=11, fontweight="bold", color=COLORS["teal"],
    )
    ax.text(
        7.2, 4.32, "a name is a label bound to an object",
        ha="center", fontsize=8.5, color=COLORS["muted"], style="italic",
    )

    # Names as small tags.
    rounded_box(
        ax, (5.15, 3.45), 1.0, 0.55, "a",
        edgecolor=COLORS["primary"], fontsize=10, fontweight="bold",
    )
    rounded_box(
        ax, (5.15, 2.45), 1.0, 0.55, "b",
        edgecolor=COLORS["primary"], fontsize=10, fontweight="bold",
    )

    # One shared object.
    obj = Circle(
        (8.35, 3.28), 0.72,
        facecolor=COLORS["teal"] + "1A", edgecolor=COLORS["teal"],
        linewidth=2.0, zorder=3,
    )
    ax.add_patch(obj)
    ax.text(
        8.35, 3.40, "42", ha="center", va="center",
        fontsize=15, fontweight="bold", color=COLORS["teal"], zorder=4,
    )
    ax.text(
        8.35, 3.03, "int object", ha="center", va="center",
        fontsize=7.5, color=COLORS["muted"], zorder=4,
    )

    arrow(ax, (6.25, 3.72), (7.66, 3.45), color=COLORS["primary"])
    arrow(ax, (6.25, 2.72), (7.66, 3.10), color=COLORS["primary"])

    ax.text(
        7.2, 1.95, "one object, two names pointing at it",
        ha="center", fontsize=8.5, color=COLORS["teal"],
    )
    ax.text(
        7.2, 1.52, "a is b  →  True", ha="center",
        fontsize=9.5, fontweight="bold", color=COLORS["text"],
        family="monospace",
    )

    # Divider between the two panels.
    ax.plot([4.75, 4.75], [1.2, 4.95], color="#DDDDDD", linewidth=1.2, zorder=1)

    ax.text(
        5.0, 0.62,
        "Assignment never copies. It binds a name to an object — "
        "which is why the next figure matters.",
        ha="center", fontsize=9, color=COLORS["text"],
    )

    ax.set_title(
        "Variables in Python: Names, Not Boxes",
        fontsize=14, fontweight="bold", loc="center", pad=14,
    )
    save(fig, "names_and_objects.png")


# ---------------------------------------------------------------------------
# Figure 4 — Mutable vs immutable
# ---------------------------------------------------------------------------
def fig_mutable_vs_immutable() -> None:
    """Why `b = a` behaves differently for an int and for a list."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # ================= LEFT: immutable =================
    ax = axes[0]
    ax.set_title(
        "IMMUTABLE  (int, str, tuple, frozenset)",
        fontsize=11, fontweight="bold", color=COLORS["teal"], loc="center", pad=10,
    )

    ax.text(0.3, 9.2, "a = 10\nb = a\nb += 1", fontsize=10.5,
            family="monospace", va="top", color=COLORS["text"], linespacing=1.7)

    # Step 1
    ax.text(0.3, 6.9, "after  b = a", fontsize=9, style="italic",
            color=COLORS["muted"])
    rounded_box(ax, (0.3, 5.6), 0.95, 0.6, "a", edgecolor=COLORS["primary"],
                fontsize=10, fontweight="bold")
    rounded_box(ax, (0.3, 4.7), 0.95, 0.6, "b", edgecolor=COLORS["primary"],
                fontsize=10, fontweight="bold")
    c = Circle((3.6, 5.35), 0.62, facecolor=COLORS["teal"] + "1A",
               edgecolor=COLORS["teal"], linewidth=1.8, zorder=3)
    ax.add_patch(c)
    ax.text(3.6, 5.35, "10", ha="center", va="center", fontsize=13,
            fontweight="bold", color=COLORS["teal"], zorder=4)
    arrow(ax, (1.4, 5.9), (2.9, 5.55), color=COLORS["primary"])
    arrow(ax, (1.4, 5.0), (2.9, 5.15), color=COLORS["primary"])

    # Step 2
    ax.text(5.6, 6.9, "after  b += 1", fontsize=9, style="italic",
            color=COLORS["muted"])
    rounded_box(ax, (5.6, 5.6), 0.95, 0.6, "a", edgecolor=COLORS["primary"],
                fontsize=10, fontweight="bold")
    rounded_box(ax, (5.6, 4.7), 0.95, 0.6, "b", edgecolor=COLORS["primary"],
                fontsize=10, fontweight="bold")
    c = Circle((8.7, 6.0), 0.6, facecolor=COLORS["teal"] + "1A",
               edgecolor=COLORS["teal"], linewidth=1.8, zorder=3)
    ax.add_patch(c)
    ax.text(8.7, 6.0, "10", ha="center", va="center", fontsize=12,
            fontweight="bold", color=COLORS["teal"], zorder=4)
    c = Circle((8.7, 4.35), 0.6, facecolor=COLORS["accent"] + "1A",
               edgecolor=COLORS["accent"], linewidth=1.8, zorder=3)
    ax.add_patch(c)
    ax.text(8.7, 4.35, "11", ha="center", va="center", fontsize=12,
            fontweight="bold", color=COLORS["accent"], zorder=4)
    arrow(ax, (6.7, 5.9), (8.05, 6.0), color=COLORS["primary"])
    arrow(ax, (6.7, 5.0), (8.05, 4.5), color=COLORS["accent"])

    ax.text(5.0, 3.4,
            "`+=` could not modify 10 in place,\nso it created a NEW object "
            "and rebound b.",
            ha="center", fontsize=9, color=COLORS["text"], linespacing=1.6)
    rounded_box(ax, (1.55, 1.55), 6.9, 1.15,
                "a is unchanged  →  a = 10,  b = 11",
                facecolor=COLORS["teal"] + "12", edgecolor=COLORS["teal"],
                fontsize=10, fontweight="bold", textcolor=COLORS["teal"])

    # ================= RIGHT: mutable =================
    ax = axes[1]
    ax.set_title(
        "MUTABLE  (list, dict, set, DataFrame)",
        fontsize=11, fontweight="bold", color=COLORS["danger"], loc="center", pad=10,
    )

    ax.text(0.3, 9.2, "a = [1, 2]\nb = a\nb.append(3)", fontsize=10.5,
            family="monospace", va="top", color=COLORS["text"], linespacing=1.7)

    ax.text(0.3, 6.9, "after  b = a", fontsize=9, style="italic",
            color=COLORS["muted"])
    rounded_box(ax, (0.3, 5.6), 0.95, 0.6, "a", edgecolor=COLORS["primary"],
                fontsize=10, fontweight="bold")
    rounded_box(ax, (0.3, 4.7), 0.95, 0.6, "b", edgecolor=COLORS["primary"],
                fontsize=10, fontweight="bold")
    rounded_box(ax, (2.75, 4.95), 1.75, 0.8, "[1, 2]",
                facecolor=COLORS["danger"] + "12", edgecolor=COLORS["danger"],
                fontsize=11, fontweight="bold", textcolor=COLORS["danger"])
    arrow(ax, (1.4, 5.9), (2.65, 5.55), color=COLORS["primary"])
    arrow(ax, (1.4, 5.0), (2.65, 5.15), color=COLORS["primary"])

    ax.text(5.6, 6.9, "after  b.append(3)", fontsize=9, style="italic",
            color=COLORS["muted"])
    rounded_box(ax, (5.6, 5.6), 0.95, 0.6, "a", edgecolor=COLORS["primary"],
                fontsize=10, fontweight="bold")
    rounded_box(ax, (5.6, 4.7), 0.95, 0.6, "b", edgecolor=COLORS["primary"],
                fontsize=10, fontweight="bold")
    rounded_box(ax, (7.95, 4.95), 1.95, 0.8, "[1, 2, 3]",
                facecolor=COLORS["danger"] + "12", edgecolor=COLORS["danger"],
                fontsize=11, fontweight="bold", textcolor=COLORS["danger"])
    arrow(ax, (6.7, 5.9), (7.85, 5.55), color=COLORS["primary"])
    arrow(ax, (6.7, 5.0), (7.85, 5.15), color=COLORS["primary"])

    ax.text(5.0, 3.4,
            "`.append()` modified the object IN PLACE.\n"
            "Both names still point at that same object.",
            ha="center", fontsize=9, color=COLORS["text"], linespacing=1.6)
    rounded_box(ax, (1.15, 1.55), 7.7, 1.15,
                "a changed too!  →  a = [1, 2, 3]",
                facecolor=COLORS["danger"] + "12", edgecolor=COLORS["danger"],
                fontsize=10, fontweight="bold", textcolor=COLORS["danger"])

    fig.suptitle(
        "The #1 Source of Silent Bugs: Mutability",
        fontsize=14.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    save(fig, "mutable_vs_immutable.png")


# ---------------------------------------------------------------------------
# Figure 5 — Choosing a built-in data structure
# ---------------------------------------------------------------------------
def fig_data_structures() -> None:
    """Comparison table of the four core built-in collections."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")

    columns = ["", "list", "tuple", "dict", "set"]
    rows = [
        ["Ordered", "yes", "yes", "yes (insertion)", "no"],
        ["Mutable", "yes", "no", "yes", "yes"],
        ["Duplicates", "yes", "yes", "keys: no", "no"],
        ["Membership  x in c", "O(n)", "O(n)", "O(1)", "O(1)"],
        ["Index / key access", "O(1)", "O(1)", "O(1)", "n/a"],
        ["Append / add", "O(1)*", "n/a", "O(1)", "O(1)"],
        ["Can be a dict key", "no", "yes**", "no", "no"],
        ["Literal syntax", "[1, 2]", "(1, 2)", "{'a': 1}", "{1, 2}"],
    ]

    accent_cols = {
        1: COLORS["primary"],
        2: COLORS["accent"],
        3: COLORS["teal"],
        4: COLORS["purple"],
    }

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.26, 0.16, 0.16, 0.21, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.75)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#E4E4E4")
        cell.set_linewidth(0.9)

        if row == 0:  # header
            cell.set_facecolor(accent_cols.get(col, "#FFFFFF"))
            cell.set_text_props(
                color="white" if col > 0 else COLORS["text"],
                fontweight="bold", fontsize=11,
            )
            if col == 0:
                cell.set_facecolor("#FFFFFF")
        elif col == 0:  # row labels
            cell.set_facecolor("#FAFAFA")
            cell.set_text_props(
                ha="left", fontweight="bold", color=COLORS["text"], fontsize=9,
            )
            cell.PAD = 0.04
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#FBFBFC")
            text = rows[row - 1][col]
            # Colour-code the answer so the table can be skimmed.
            if text == "yes":
                cell.set_text_props(color=COLORS["teal"], fontweight="bold")
            elif text == "no":
                cell.set_text_props(color=COLORS["danger"])
            elif text.startswith("O(1)"):
                cell.set_text_props(color=COLORS["teal"], fontweight="bold",
                                    family="monospace")
            elif text.startswith("O(n)"):
                cell.set_text_props(color=COLORS["danger"], fontweight="bold",
                                    family="monospace")
            elif text.startswith(("[", "(", "{")):
                cell.set_text_props(family="monospace", color=COLORS["text"])

    ax.set_title(
        "Choosing a Built-in Collection",
        fontsize=14, fontweight="bold", loc="center", pad=22,
    )
    ax.text(
        0.5, -0.08,
        "*  amortised — an occasional resize costs O(n)        "
        "**  only if every element it contains is itself hashable",
        transform=ax.transAxes, ha="center", fontsize=8, color=COLORS["muted"],
    )
    ax.text(
        0.5, -0.16,
        "Rule of thumb: reach for a set or dict the moment you find yourself "
        "writing `if x in my_list` inside a loop.",
        transform=ax.transAxes, ha="center", fontsize=9,
        color=COLORS["text"], style="italic",
    )

    save(fig, "data_structures_comparison.png")


# ---------------------------------------------------------------------------
# Figure 6 — LEGB scope resolution
# ---------------------------------------------------------------------------
def fig_legb_scope() -> None:
    """How Python resolves a name: Local → Enclosing → Global → Built-in."""
    fig, ax = blank_axes((9.4, 6.6), xlim=(0, 9.4), ylim=(0, 6.6))

    # (label, description, colour, x, y, w, h, search-order badge)
    # Each ring is inset by 0.55 horizontally and leaves a 0.72-tall header
    # strip at the top, so no label ever collides with an inner border.
    scopes = [
        ("B — Built-in", "print, len, sum, ValueError …",
         COLORS["purple"], 0.30, 0.30, 8.80, 5.95, "4"),
        ("G — Global (module)", "names defined at file top level",
         COLORS["primary"], 0.85, 0.85, 7.70, 4.68, "3"),
        ("E — Enclosing function", "names in an outer def wrapping this one",
         COLORS["teal"], 1.40, 1.40, 6.60, 3.41, "2"),
        ("L — Local", "names assigned inside the current function",
         COLORS["accent"], 1.95, 1.95, 5.50, 2.14, "1"),
    ]

    for title, subtitle, color, x, y, w, h, badge in scopes:
        rounded_box(
            ax, (x, y), w, h, "",
            facecolor=color + "0F", edgecolor=color, linewidth=1.8,
        )
        # Title over description, stacked so neither can collide with the other
        # regardless of how long the title is.
        ax.text(x + 0.66, y + h - 0.26, title, fontsize=9.5,
                fontweight="bold", color=color, va="center")
        ax.text(x + 0.66, y + h - 0.53, subtitle, fontsize=7.6,
                color=COLORS["muted"], va="center")

        # Search-order badge pinned to the left of the header strip.
        badge_circle = Circle(
            (x + 0.34, y + h - 0.39), 0.19,
            facecolor=color, edgecolor=color, zorder=5,
        )
        ax.add_patch(badge_circle)
        ax.text(x + 0.34, y + h - 0.39, badge, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=6)

    # Explanation sits in the empty interior of the innermost ring.
    ax.text(
        4.70, 2.72,
        "Python resolves a name by searching\n"
        "①  →  ②  →  ③  →  ④\n"
        "and stops at the FIRST match.",
        ha="center", va="center", fontsize=9.5,
        color=COLORS["text"], linespacing=1.8,
    )

    ax.set_title(
        "The LEGB Rule — How Python Resolves a Name",
        fontsize=13.5, fontweight="bold", loc="center", pad=12,
    )
    save(fig, "legb_scope.png")


# ---------------------------------------------------------------------------
# Figure 7 — Exception hierarchy
# ---------------------------------------------------------------------------
def fig_exception_hierarchy() -> None:
    """The slice of Python's exception tree a data scientist meets daily."""
    fig, ax = blank_axes((11, 5.6), xlim=(0, 11), ylim=(0, 5.6))

    rounded_box(
        ax, (4.05, 4.65), 2.9, 0.62, "BaseException",
        edgecolor=COLORS["muted"], fontsize=10, fontweight="bold",
        textcolor=COLORS["muted"],
    )
    ax.text(7.15, 4.96, "← never catch this one", fontsize=8.5,
            color=COLORS["danger"], va="center", style="italic")

    rounded_box(
        ax, (4.15, 3.5), 2.7, 0.62, "Exception",
        edgecolor=COLORS["primary"], fontsize=10, fontweight="bold",
        textcolor=COLORS["primary"],
    )
    ax.text(7.05, 3.81, "← the base of everything you should catch",
            fontsize=8.5, color=COLORS["primary"], va="center", style="italic")
    arrow(ax, (5.5, 4.62), (5.5, 4.15), color=COLORS["muted"])

    leaves = [
        ("ValueError", "right type,\nwrong value", COLORS["accent"], 0.15),
        ("TypeError", "wrong type\nentirely", COLORS["accent"], 2.32),
        ("KeyError", "missing dict\nkey", COLORS["teal"], 4.49),
        ("IndexError", "sequence index\nout of range", COLORS["teal"], 6.66),
        ("FileNotFoundError", "path does\nnot exist", COLORS["purple"], 8.83),
    ]

    for name, desc, color, x in leaves:
        w = 2.02
        rounded_box(
            ax, (x, 1.72), w, 0.6, name,
            edgecolor=color, fontsize=9, fontweight="bold", textcolor=color,
        )
        ax.text(x + w / 2, 1.28, desc, ha="center", va="center",
                fontsize=7.8, color=COLORS["muted"], linespacing=1.5)
        # Orthogonal connector: down from Exception, across the bus, into the leaf.
        elbow_connector(ax, (5.5, 3.47), (x + w / 2, 2.36), bus_y=2.92)

    ax.text(
        5.5, 0.42,
        "Catch the most specific exception that you can actually handle. "
        "`except Exception:` hides bugs; bare `except:` also swallows Ctrl-C.",
        ha="center", fontsize=9, color=COLORS["text"],
    )

    ax.set_title(
        "Python Exceptions You Will Meet Weekly",
        fontsize=13.5, fontweight="bold", loc="center", pad=12,
    )
    save(fig, "exception_hierarchy.png")


# ===========================================================================
# MODULE 02 — NumPy & Numerical Computing
# ===========================================================================


def fig_numpy_memory_layout() -> None:
    """Why a Python list of numbers is slow: pointers versus a packed buffer."""
    fig, ax = blank_axes((12, 5.6), xlim=(0, 12), ylim=(0, 5.6))

    # ---------------- LEFT: Python list -----------------------------------
    ax.text(3.0, 5.25, "Python  list", ha="center", fontsize=12.5,
            fontweight="bold", color=COLORS["danger"], family="monospace")
    ax.text(3.0, 4.86, "an array of POINTERS to scattered objects",
            ha="center", fontsize=8.5, color=COLORS["muted"], style="italic")

    # The pointer array itself.
    for i in range(4):
        x = 0.72 + i * 1.14
        rounded_box(ax, (x, 3.85), 1.02, 0.62, "ptr",
                    edgecolor=COLORS["danger"], fontsize=8.5,
                    textcolor=COLORS["danger"])

    # The heap objects the pointers reach, deliberately at uneven heights to
    # convey that they are scattered rather than adjacent.
    # Box width (1.02) must stay below the 1.14 column pitch, otherwise
    # neighbours collide whenever their y ranges happen to overlap.
    heap_y = [2.30, 1.50, 2.55, 1.24]
    for i, (value, y) in enumerate(zip([10, 11, 12, 13], heap_y, strict=True)):
        x = 0.72 + i * 1.14
        rounded_box(ax, (x, y), 1.02, 0.72, f"PyObject\n{value}",
                    facecolor=COLORS["danger"] + "10",
                    edgecolor=COLORS["danger"], fontsize=7.4,
                    textcolor=COLORS["text"])
        arrow(ax, (x + 0.51, 3.82), (x + 0.51, y + 0.76),
              color=COLORS["danger"], linewidth=1.2)

    ax.text(3.0, 0.86, "≈ 8 bytes per pointer  +  ≈ 28 bytes per int object",
            ha="center", fontsize=8.5, color=COLORS["danger"], fontweight="bold")
    ax.text(3.0, 0.46,
            "every element costs a pointer hop; types are checked one by one",
            ha="center", fontsize=8, color=COLORS["muted"])

    # ---------------- RIGHT: ndarray --------------------------------------
    ax.text(9.0, 5.25, "NumPy  ndarray", ha="center", fontsize=12.5,
            fontweight="bold", color=COLORS["teal"], family="monospace")
    ax.text(9.0, 4.86, "ONE contiguous, typed block of memory",
            ha="center", fontsize=8.5, color=COLORS["muted"], style="italic")

    # A small header block, then the packed data buffer.
    rounded_box(ax, (6.35, 3.85), 1.5, 0.62, "header\nshape, dtype",
                edgecolor=COLORS["teal"], fontsize=7,
                textcolor=COLORS["muted"])

    for i, value in enumerate([10, 11, 12, 13]):
        # No gaps between cells: adjacency is the whole point of the figure.
        x = 8.05 + i * 0.92
        rounded_box(ax, (x, 3.85), 0.92, 0.62, str(value),
                    facecolor=COLORS["teal"] + "18",
                    edgecolor=COLORS["teal"], fontsize=10,
                    fontweight="bold", textcolor=COLORS["text"])

    # Brace-like span marker under the buffer.
    ax.plot([8.05, 11.73], [3.55, 3.55], color=COLORS["teal"], linewidth=1.4)
    for x in (8.05, 11.73):
        ax.plot([x, x], [3.55, 3.70], color=COLORS["teal"], linewidth=1.4)
    ax.text(9.89, 3.24, "4 × 8 bytes, back to back",
            ha="center", fontsize=8, color=COLORS["teal"])

    ax.text(9.0, 0.86, "= 8 bytes per element,  nothing else",
            ha="center", fontsize=8.5, color=COLORS["teal"], fontweight="bold")
    ax.text(9.0, 0.46,
            "one dtype for the whole block → the CPU can vectorise the loop",
            ha="center", fontsize=8, color=COLORS["muted"])

    ax.plot([6.0, 6.0], [0.3, 5.4], color="#DDDDDD", linewidth=1.2, zorder=1)

    ax.set_title("Why NumPy Is Faster: It Is About Memory Layout",
                 fontsize=14, fontweight="bold", loc="center", pad=14)
    save(fig, "numpy_memory_layout.png")


def fig_broadcasting_rules() -> None:
    """The right-align-and-stretch rule, worked through three shape pairs."""
    fig, ax = blank_axes((11.5, 6.4), xlim=(0, 11.5), ylim=(0, 6.4))

    ax.text(5.75, 6.0,
            "Align shapes from the RIGHT.  A dimension is compatible when it "
            "matches — or is 1.",
            ha="center", fontsize=10, color=COLORS["text"])

    # (label, shape A, shape B, result or None when incompatible)
    cases = [
        ("(3, 4)  +  (4,)", ["3", "4"], ["1*", "4"], ["3", "4"], True),
        ("(3, 1)  +  (1, 4)", ["3", "1"], ["1", "4"], ["3", "4"], True),
        ("(3, 4)  +  (3,)", ["3", "4"], ["1*", "3"], None, False),
    ]

    y = 4.55
    for label, shape_a, shape_b, result, ok in cases:
        color = COLORS["teal"] if ok else COLORS["danger"]

        ax.text(0.25, y + 0.30, label, fontsize=10.5, fontweight="bold",
                family="monospace", color=COLORS["text"], va="center")

        # Two stacked shape rows, right-aligned against a common edge.
        for row, (shape, tag) in enumerate([(shape_a, "A"), (shape_b, "B")]):
            row_y = y - row * 0.52
            ax.text(3.42, row_y + 0.21, tag, fontsize=8.5,
                    color=COLORS["muted"], ha="right", va="center")
            for j, dim in enumerate(shape):
                x = 3.62 + j * 0.78
                # A starred entry is a dimension NumPy padded on the left.
                padded = dim.endswith("*")
                rounded_box(
                    ax, (x, row_y), 0.68, 0.42, dim.rstrip("*"),
                    facecolor="#FFFFFF" if not padded else COLORS["muted"] + "18",
                    edgecolor=COLORS["muted"] if padded else COLORS["primary"],
                    fontsize=9.5, fontweight="bold",
                    textcolor=COLORS["muted"] if padded else COLORS["text"],
                )

        # Result, or the reason it fails.
        ax.text(5.55, y - 0.05, "→", fontsize=15, color=color,
                ha="center", va="center")

        if ok:
            for j, dim in enumerate(result):
                x = 6.05 + j * 0.78
                rounded_box(ax, (x, y - 0.26), 0.68, 0.42, dim,
                            facecolor=color + "18", edgecolor=color,
                            fontsize=9.5, fontweight="bold",
                            textcolor=COLORS["text"])
            ax.text(7.75, y - 0.05, "✔  broadcasts", fontsize=9.5,
                    color=color, va="center", fontweight="bold")
        else:
            ax.text(6.05, y - 0.05, "✘  4 vs 3 — neither is 1",
                    fontsize=9.5, color=color, va="center", fontweight="bold")

        y -= 1.45

    ax.text(0.25, 0.86,
            "*  a shape shorter than the other is padded with 1s on the LEFT "
            "before comparing",
            fontsize=8.5, color=COLORS["muted"])
    ax.text(0.25, 0.46,
            "To fix the failing case, make the intent explicit:  "
            "b.reshape(3, 1)  →  (3, 4)",
            fontsize=9, color=COLORS["text"], style="italic")

    ax.set_title("Broadcasting: the Two Rules",
                 fontsize=14, fontweight="bold", loc="center", pad=14)
    save(fig, "broadcasting_rules.png")


def fig_axis_semantics() -> None:
    """`axis=` names the dimension that disappears."""
    fig, ax = blank_axes((11, 5.8), xlim=(0, 11), ylim=(0, 5.8))

    values = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])
    n_rows, n_cols = values.shape

    cell = 0.72
    grid_x, grid_y = 2.55, 2.35

    # The array itself.
    for i in range(n_rows):
        for j in range(n_cols):
            x = grid_x + j * cell
            # Row 0 at the top: matrix order, not plot order.
            y = grid_y + (n_rows - 1 - i) * cell
            rounded_box(ax, (x, y), cell, cell, str(values[i, j]),
                        facecolor="#FFFFFF", edgecolor="#C8C8C8",
                        fontsize=9.5, linewidth=1.1)

    # Sits just above the grid (top edge at grid_y + 3*cell = 4.51) and clear of
    # the rule banner, which starts at 5.02.
    ax.text(grid_x + n_cols * cell / 2, grid_y + n_rows * cell + 0.26,
            "arr.shape == (3, 4)", ha="center", fontsize=10,
            family="monospace", fontweight="bold", color=COLORS["text"])

    # --- axis=0: collapse DOWN the rows ------------------------------------
    for j in range(n_cols):
        x = grid_x + j * cell + cell / 2
        arrow(ax, (x, grid_y - 0.12), (x, grid_y - 0.72),
              color=COLORS["accent"], linewidth=1.6)
    for j in range(n_cols):
        x = grid_x + j * cell
        rounded_box(ax, (x, grid_y - 1.5), cell, cell,
                    str(values.sum(axis=0)[j]),
                    facecolor=COLORS["accent"] + "18",
                    edgecolor=COLORS["accent"], fontsize=9.5,
                    fontweight="bold")

    ax.text(grid_x + n_cols * cell / 2, grid_y - 1.95,
            "arr.sum(axis=0)  →  shape (4,)", ha="center", fontsize=9.5,
            family="monospace", fontweight="bold", color=COLORS["accent"])
    ax.text(grid_x + n_cols * cell / 2, grid_y - 2.32,
            "axis 0 (the rows) is gone", ha="center", fontsize=8.5,
            color=COLORS["muted"])

    # --- axis=1: collapse ACROSS the columns -------------------------------
    right_edge = grid_x + n_cols * cell
    for i in range(n_rows):
        y = grid_y + (n_rows - 1 - i) * cell + cell / 2
        arrow(ax, (right_edge + 0.12, y), (right_edge + 0.72, y),
              color=COLORS["purple"], linewidth=1.6)
        rounded_box(ax, (right_edge + 0.86, y - cell / 2), cell, cell,
                    str(values.sum(axis=1)[i]),
                    facecolor=COLORS["purple"] + "18",
                    edgecolor=COLORS["purple"], fontsize=9.5,
                    fontweight="bold")

    ax.text(right_edge + 0.86 + cell / 2, grid_y - 0.55,
            "arr.sum(axis=1)", ha="center", fontsize=9.5,
            family="monospace", fontweight="bold", color=COLORS["purple"])
    ax.text(right_edge + 0.86 + cell / 2, grid_y - 0.92,
            "→ shape (3,)", ha="center", fontsize=9,
            family="monospace", color=COLORS["purple"])

    # The rule, stated once, in a banner above everything else.
    rounded_box(ax, (0.3, 5.02), 10.4, 0.68, "",
                facecolor=COLORS["primary"] + "10",
                edgecolor=COLORS["primary"], linewidth=1.6)
    ax.text(5.5, 5.36,
            "`axis` names the dimension that DISAPPEARS — not the one you keep.",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=COLORS["primary"])

    ax.set_title("Reading `axis=` Correctly",
                 fontsize=14, fontweight="bold", loc="center", pad=12)
    save(fig, "axis_semantics.png")


def fig_views_vs_copies() -> None:
    """Slicing a list copies; slicing an array does not."""
    fig, ax = blank_axes((12, 5.8), xlim=(0, 12), ylim=(0, 5.8))

    # ---------------- LEFT: list slicing copies ---------------------------
    ax.text(3.0, 5.35, "list  —  slicing COPIES", ha="center", fontsize=11.5,
            fontweight="bold", color=COLORS["teal"])
    ax.text(3.0, 4.95, "lst = [1, 2, 3, 4]\npart = lst[1:3]\npart[0] = 99",
            ha="center", fontsize=9, family="monospace",
            color=COLORS["text"], linespacing=1.5, va="top")

    for i, value in enumerate([1, 2, 3, 4]):
        rounded_box(ax, (0.9 + i * 0.85, 2.72), 0.85, 0.6, str(value),
                    facecolor="#FFFFFF", edgecolor="#C8C8C8", fontsize=9.5)
    ax.text(0.55, 3.02, "lst", fontsize=9.5, fontweight="bold",
            family="monospace", color=COLORS["text"], ha="right", va="center")

    for i, value in enumerate([99, 3]):
        rounded_box(ax, (1.75 + i * 0.85, 1.62), 0.85, 0.6, str(value),
                    facecolor=COLORS["teal"] + "18",
                    edgecolor=COLORS["teal"], fontsize=9.5, fontweight="bold")
    ax.text(1.4, 1.92, "part", fontsize=9.5, fontweight="bold",
            family="monospace", color=COLORS["teal"], ha="right", va="center")

    ax.text(3.0, 1.05, "a SECOND buffer — lst is untouched",
            ha="center", fontsize=9, color=COLORS["teal"], fontweight="bold")
    ax.text(3.0, 0.62, "lst  →  [1, 2, 3, 4]", ha="center", fontsize=9.5,
            family="monospace", color=COLORS["text"])

    # ---------------- RIGHT: array slicing views --------------------------
    ax.text(9.0, 5.35, "ndarray  —  slicing gives a VIEW", ha="center",
            fontsize=11.5, fontweight="bold", color=COLORS["danger"])
    ax.text(9.0, 4.95,
            "arr = np.array([1, 2, 3, 4])\npart = arr[1:3]\npart[0] = 99",
            ha="center", fontsize=9, family="monospace",
            color=COLORS["text"], linespacing=1.5, va="top")

    for i, value in enumerate([1, 99, 3, 4]):
        highlighted = i in (1, 2)
        rounded_box(
            ax, (6.9 + i * 0.85, 2.72), 0.85, 0.6, str(value),
            facecolor=COLORS["danger"] + "18" if highlighted else "#FFFFFF",
            edgecolor=COLORS["danger"] if highlighted else "#C8C8C8",
            fontsize=9.5, fontweight="bold" if highlighted else "normal",
        )
    ax.text(6.55, 3.02, "arr", fontsize=9.5, fontweight="bold",
            family="monospace", color=COLORS["text"], ha="right", va="center")

    # `part` is drawn as a bracket over the shared cells, not as a new buffer.
    ax.plot([7.75, 9.45], [2.35, 2.35], color=COLORS["danger"], linewidth=1.6)
    for x in (7.75, 9.45):
        ax.plot([x, x], [2.35, 2.55], color=COLORS["danger"], linewidth=1.6)
    ax.text(8.6, 1.98, "part  — same memory, no copy", ha="center",
            fontsize=9, family="monospace", color=COLORS["danger"],
            fontweight="bold")

    ax.text(9.0, 1.05, "ONE buffer — writing through the view changes arr",
            ha="center", fontsize=9, color=COLORS["danger"], fontweight="bold")
    ax.text(9.0, 0.62, "arr  →  [1, 99, 3, 4]", ha="center", fontsize=9.5,
            family="monospace", color=COLORS["danger"], fontweight="bold")

    ax.plot([6.0, 6.0], [0.4, 5.5], color="#DDDDDD", linewidth=1.2, zorder=1)

    ax.text(6.0, 0.16,
            "Check with  arr.base is not None  ·  force a copy with  arr[1:3].copy()",
            ha="center", fontsize=8.5, color=COLORS["muted"], style="italic")

    ax.set_title("The Rule That Reverses from Module 01",
                 fontsize=14, fontweight="bold", loc="center", pad=12)
    save(fig, "views_vs_copies.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
FIGURES = [
    fig_ds_lifecycle,
    fig_stack_map,
    fig_names_and_objects,
    fig_mutable_vs_immutable,
    fig_data_structures,
    fig_legb_scope,
    fig_exception_hierarchy,
    # --- Module 02 ---
    fig_numpy_memory_layout,
    fig_broadcasting_rules,
    fig_axis_semantics,
    fig_views_vs_copies,
]


def main() -> None:
    setup_plot_style()
    print(f"Generating {len(FIGURES)} figures into {OUTPUT_DIR.relative_to(REPO_ROOT)}/\n")
    for make_figure in FIGURES:
        make_figure()
    print(f"\nDone — {len(FIGURES)} figures written.")


if __name__ == "__main__":
    main()
