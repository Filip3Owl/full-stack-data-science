"""Render the course diagrams in Portuguese, for the web version only.

The drawing code stays in ``generate_figures.py``. This script runs it with every
piece of text swapped through ``web/i18n/pt/figures.json`` and writes the result to
``web/i18n/pt/images/``, leaving ``assets/images/`` and the notebooks untouched.

    # render the translated figures
    .venv/bin/python scripts/translate_figures.py

    # list the English strings, ready to translate
    .venv/bin/python scripts/translate_figures.py --list

A string with no translation stops the run, so a changed diagram cannot quietly
ship with English left in it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import generate_figures  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from dstoolkit.plotting import setup_plot_style  # noqa: E402

CATALOG = generate_figures.REPO_ROOT / "web" / "i18n" / "pt" / "figures.json"
OUTPUT = generate_figures.REPO_ROOT / "web" / "i18n" / "pt" / "images"


def run_figures(translate, output: Path) -> None:
    """Draw every figure into `output`, passing all text through `translate`."""
    original = Text.set_text

    def set_text(self: Text, value: object) -> None:
        original(self, translate(value))

    Text.set_text = set_text
    generate_figures.OUTPUT_DIR = output
    generate_figures.REPO_ROOT = output.parent  # save() prints paths relative to it
    try:
        setup_plot_style()
        for make_figure in generate_figures.FIGURES:
            make_figure()
    finally:
        Text.set_text = original


def english_strings(scratch: Path) -> list[str]:
    """Collect every string the figures draw, in drawing order."""
    seen: dict[str, None] = {}

    def record(value: object) -> object:
        if isinstance(value, str) and value.strip():
            seen[value] = None
        return value

    run_figures(record, scratch)
    return list(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list", action="store_true", help="print the English strings as JSON"
    )
    args = parser.parse_args()

    if args.list:
        import tempfile

        with tempfile.TemporaryDirectory() as scratch:
            strings = english_strings(Path(scratch))
        print(json.dumps(strings, ensure_ascii=False, indent=1))
        return 0

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    missing: set[str] = set()

    def translate(value: object) -> object:
        if not isinstance(value, str) or not re.search(r"[^\W\d_]", value):
            return value  # axis ticks and other text with no letters
        if value not in catalog:
            missing.add(value)
            return value
        return catalog[value]

    run_figures(translate, OUTPUT)
    if missing:
        print("No translation for:", *sorted(missing), sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
