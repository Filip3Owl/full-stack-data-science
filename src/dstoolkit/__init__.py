"""dstoolkit — reusable helpers for the Full-Stack Data Science course.

Code that proves useful inside a notebook gets *promoted* into this package,
covered by a test in ``tests/``, and then imported back into later notebooks.
That promotion loop is how real data science projects avoid drowning in
un-reusable notebook code.

Usage from any notebook::

    import sys; sys.path.append("../../../src")   # or install with `pip install -e .`
    from dstoolkit import setup_plot_style
"""

from dstoolkit.plotting import COLORS, setup_plot_style
from dstoolkit.utils import format_bytes, memory_footprint, timer

__version__ = "0.1.0"

__all__ = [
    "COLORS",
    "setup_plot_style",
    "format_bytes",
    "memory_footprint",
    "timer",
]
