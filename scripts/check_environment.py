"""Verify that the course environment is correctly installed.

Run this right after setting up your virtual environment:

    .venv/bin/python scripts/check_environment.py

It checks the Python version and imports every core package, reporting the
version actually installed. Exit code is 0 when everything is fine, 1 otherwise
so the script can be used in CI.
"""

from __future__ import annotations

import importlib
import platform
import sys
from collections.abc import Sequence

# Minimum Python version the course is built and tested against.
REQUIRED_PYTHON = (3, 12)

# (import name, human-readable name) — import name differs from the PyPI name
# for a few packages, e.g. `sklearn` is installed as `scikit-learn`.
CORE_PACKAGES: list[tuple[str, str]] = [
    ("numpy", "NumPy"),
    ("pandas", "pandas"),
    ("pyarrow", "PyArrow"),
    ("scipy", "SciPy"),
    ("statsmodels", "statsmodels"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "Plotly"),
    ("jupyterlab", "JupyterLab"),
    ("jupytext", "Jupytext"),
]

# Optional extras, grouped by the modules that need them. Missing ones are
# reported as informational, never as failures.
OPTIONAL_PACKAGES: list[tuple[str, str, str]] = [
    ("xgboost", "XGBoost", "advanced-ml"),
    ("lightgbm", "LightGBM", "advanced-ml"),
    ("torch", "PyTorch", "deep-learning"),
    ("transformers", "Transformers", "deep-learning"),
    ("pyspark", "PySpark", "big-data"),
    ("duckdb", "DuckDB", "big-data"),
    ("mlflow", "MLflow", "mlops"),
    ("fastapi", "FastAPI", "mlops"),
]

# ANSI colours; disabled automatically when output is piped to a file.
_TTY = sys.stdout.isatty()
GREEN = "\033[92m" if _TTY else ""
YELLOW = "\033[93m" if _TTY else ""
RED = "\033[91m" if _TTY else ""
BOLD = "\033[1m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""

PASS = f"{GREEN}✔{RESET}"
WARN = f"{YELLOW}○{RESET}"
FAIL = f"{RED}✘{RESET}"


def _version_of(module) -> str:
    """Return a module's version string, tolerating packages without __version__."""
    return getattr(module, "__version__", "unknown")


def check_python() -> bool:
    """Report the running interpreter and whether it meets the course minimum."""
    actual = sys.version_info[:2]
    ok = actual >= REQUIRED_PYTHON

    marker = PASS if ok else FAIL
    required = ".".join(map(str, REQUIRED_PYTHON))
    print(f"{BOLD}Interpreter{RESET}")
    print(f"  {marker} Python {platform.python_version()} (need >= {required})")
    print(f"    executable: {sys.executable}")

    if not ok:
        print(
            f"    {RED}This is not the course environment.{RESET} "
            f"Activate it with: source .venv/bin/activate"
        )
    return ok


def check_packages(
    packages: Sequence[tuple[str, ...]], *, optional: bool = False
) -> list[str]:
    """Import each package and print its version. Returns the names that failed.

    Entries are ``(import_name, display_name)`` or, for optional packages,
    ``(import_name, display_name, extras_group)`` so we can print an install hint.
    """
    missing: list[str] = []

    for entry in packages:
        import_name, display_name = entry[0], entry[1]
        extra = entry[2] if len(entry) > 2 else None

        try:
            module = importlib.import_module(import_name)
        except ImportError:
            missing.append(display_name)
            hint = f"  → install with: pip install -e '.[{extra}]'" if extra else ""
            print(f"  {WARN if optional else FAIL} {display_name:<16} not installed{hint}")
        else:
            print(f"  {PASS} {display_name:<16} {_version_of(module)}")

    return missing


def main() -> int:
    print(f"\n{BOLD}Full-Stack Data Science — environment check{RESET}\n")

    python_ok = check_python()

    print(f"\n{BOLD}Core packages{RESET} (Modules 01-15)")
    missing_core = check_packages(CORE_PACKAGES)

    print(f"\n{BOLD}Optional extras{RESET} (installed on demand for later modules)")
    check_packages(OPTIONAL_PACKAGES, optional=True)

    print()
    if python_ok and not missing_core:
        print(f"{GREEN}{BOLD}All set — you are ready to start Module 01.{RESET}\n")
        return 0

    if missing_core:
        print(f"{RED}Missing core packages: {', '.join(missing_core)}{RESET}")
        print("Fix with:  pip install -e '.[dev]'\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
