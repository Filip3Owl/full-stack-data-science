"""Build executable notebooks from their plain-text `.py` sources.

The course authors lessons as jupytext *percent-format* `.py` files, which diff
cleanly in git, and generates the `.ipynb` files from them. Each notebook is then
executed so its outputs and figures are committed alongside the code — that way a
reader can browse a lesson on GitHub, and we get a guarantee that every cell in
the course actually runs.

    # build everything
    .venv/bin/python scripts/build_notebooks.py

    # build one module
    .venv/bin/python scripts/build_notebooks.py --module 01

    # convert without executing (fast, for a syntax check)
    .venv/bin/python scripts/build_notebooks.py --no-execute
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = REPO_ROOT / "modules"
PYTHON = sys.executable

# Cells that hit the network or take minutes would make the build fragile; the
# course avoids them, so a five-minute ceiling per notebook is generous.
CELL_TIMEOUT_SECONDS = 600

GREEN, RED, YELLOW, BOLD, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"


def discover(module_filter: str | None) -> list[Path]:
    """Find every notebook source, optionally restricted to one module."""
    pattern = f"{module_filter}-*/scripts/*.py" if module_filter else "*/scripts/*.py"
    return sorted(MODULES_DIR.glob(pattern))


def run(cmd: list[str]) -> tuple[bool, str]:
    """Run a subprocess, returning (succeeded, combined output)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr


def convert(source: Path, target: Path) -> tuple[bool, str]:
    """Convert a percent-format .py into a .ipynb."""
    target.parent.mkdir(parents=True, exist_ok=True)
    return run([PYTHON, "-m", "jupytext", "--to", "ipynb",
                "--output", str(target), str(source)])


def execute(notebook: Path) -> tuple[bool, str]:
    """Execute a notebook in place so outputs are embedded."""
    return run([
        PYTHON, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={CELL_TIMEOUT_SECONDS}",
        str(notebook),
    ])


def extract_error(output: str) -> str:
    """Pull the useful part out of nbconvert's very verbose failure output."""
    lines = output.splitlines()
    for i, line in enumerate(lines):
        # nbconvert prints the offending cell then a traceback; the last
        # traceback line is almost always the actual exception.
        if "Error" in line and ("Traceback" in output or ":" in line):
            tail = [ln for ln in lines[i:] if ln.strip()][:6]
            return "\n      ".join(tail)
    return "\n      ".join(lines[-6:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", help="module number prefix, e.g. 01")
    parser.add_argument("--no-execute", action="store_true",
                        help="convert only; do not run the notebooks")
    args = parser.parse_args()

    sources = discover(args.module)
    if not sources:
        print(f"{RED}No notebook sources found.{RESET}")
        return 1

    print(f"\n{BOLD}Building {len(sources)} notebook(s)"
          f"{' (convert only)' if args.no_execute else ''}{RESET}\n")

    failures: list[tuple[str, str]] = []

    for source in sources:
        # modules/<mod>/scripts/x.py  →  modules/<mod>/notebooks/x.ipynb
        target = source.parent.parent / "notebooks" / f"{source.stem}.ipynb"
        label = f"{source.parent.parent.name}/{source.stem}"

        print(f"  {label:<52}", end="", flush=True)
        start = time.perf_counter()

        ok, output = convert(source, target)
        if not ok:
            print(f"{RED}CONVERT FAILED{RESET}")
            failures.append((label, extract_error(output)))
            continue

        if not args.no_execute:
            ok, output = execute(target)
            if not ok:
                print(f"{RED}EXEC FAILED{RESET}")
                failures.append((label, extract_error(output)))
                continue

        print(f"{GREEN}ok{RESET}  ({time.perf_counter() - start:5.1f}s)")

    print()
    if failures:
        print(f"{RED}{BOLD}{len(failures)} notebook(s) failed:{RESET}\n")
        for label, error in failures:
            print(f"  {YELLOW}{label}{RESET}\n      {error}\n")
        return 1

    print(f"{GREEN}{BOLD}All {len(sources)} notebook(s) built successfully.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
