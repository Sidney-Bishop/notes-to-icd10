"""
src/plot_utils.py — Figure Persistence and Traceability
========================================================
Centralised figure saving for all notebooks and scripts.

Every plot generated in this project — whether from a notebook cell or a
script — should be saved via save_figure() rather than plt.show() alone.
This ensures:

  1. Figures survive between sessions
  2. Every figure is logged to outputs/run.log with its full path
  3. Figures are organised by source notebook/script and named
     with a timestamp to prevent overwrites
  4. A paper author can query the log to find all figures for a
     given experiment or notebook

Output layout
-------------
    outputs/visualizations/
        01-EDA_SOAP/
            token_pressure_kde_20260426_031500.png
            icd10_frequency_histogram_20260426_031500.png
        E-009_Balanced_E002Init/
            chapter_accuracy_20260426_040000.png
            calibration_ece_20260426_040100.png

Usage — notebooks
-----------------
    from src.plot_utils import save_figure
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(...)
    save_figure(fig, notebook="01-EDA_SOAP", description="token_pressure_kde")
    plt.show()
    plt.close()

Usage — scripts
---------------
    from src.plot_utils import save_figure
    fig = plt.figure()
    ...
    path = save_figure(fig, notebook="evaluate", description="chapter_accuracy",
                       experiment="E-009_Balanced_E002Init")

Usage — seaborn (no explicit fig object)
-----------------------------------------
    ax = sns.kdeplot(...)
    save_figure(ax.figure, notebook="01-EDA_SOAP", description="token_kde")

Public API
----------
    save_figure(fig, notebook, description, experiment=None, dpi=150, fmt="png")
        Save a matplotlib figure and log the path.
        Returns the saved Path.

    list_figures(notebook=None, experiment=None)
        Return all saved figure paths, optionally filtered.

    figure_report()
        Print a summary table of all saved figures.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root detection — mirrors the pattern used across all src modules
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    current = Path.cwd()
    while current != current.parent:
        if (current / "artifacts.yaml").exists():
            return current.resolve()
        current = current.parent
    raise FileNotFoundError(
        "Could not find artifacts.yaml — run from within the project tree."
    )


def _visualizations_dir() -> Path:
    """Return outputs/visualizations/, creating it if needed."""
    root = _project_root()
    d = root / "outputs" / "visualizations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _figure_index_path() -> Path:
    """Path to the machine-readable figure index JSON."""
    root = _project_root()
    return root / "outputs" / "figure_index.json"


def _run_log_path() -> Path:
    root = _project_root()
    return root / "outputs" / "run.log"


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _load_index() -> list:
    p = _figure_index_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return []
    return []


def _save_index(index: list) -> None:
    p = _figure_index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=2, default=str))


def _append_run_log(line: str) -> None:
    p = _run_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_figure(
    fig,
    notebook: str,
    description: str,
    experiment: Optional[str] = None,
    dpi: int = 150,
    fmt: str = "png",
) -> Path:
    """
    Save a matplotlib figure to outputs/visualizations/ and log the path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save. For seaborn plots without an explicit figure,
        pass `ax.figure`.
    notebook : str
        Name of the notebook or script producing this figure.
        Used as the subdirectory name.
        Examples: "01-EDA_SOAP", "evaluate", "calibrate"
    description : str
        Short description of what the figure shows.
        Used in the filename — use underscores, no spaces.
        Examples: "token_pressure_kde", "chapter_accuracy", "ece_calibration"
    experiment : str, optional
        Experiment name (e.g. "E-009_Balanced_E002Init"). If provided,
        it is recorded in the index and log entry for traceability.
    dpi : int
        Resolution. Default 150 — good for screen and papers.
        Use 300 for publication-quality figures.
    fmt : str
        File format. Default "png". Use "pdf" for vector figures.

    Returns
    -------
    Path
        The full path where the figure was saved.

    Examples
    --------
    >>> save_figure(fig, "01-EDA_SOAP", "token_pressure_kde")
    PosixPath('.../outputs/visualizations/01-EDA_SOAP/token_pressure_kde_20260426_031500.png')

    >>> save_figure(fig, "evaluate", "chapter_accuracy",
    ...             experiment="E-009_Balanced_E002Init", dpi=300)
    """
    # Validate fig
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for save_figure(). "
            "Install with: pip install matplotlib"
        )

    if fig is None:
        raise ValueError("fig cannot be None — pass a matplotlib Figure object.")

    # Sanitise description — replace spaces with underscores
    description = description.strip().replace(" ", "_").replace("/", "-")
    notebook_safe = notebook.strip().replace(" ", "_").replace("/", "-")

    # Build output path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{description}_{ts}.{fmt}"
    out_dir = _visualizations_dir() / notebook_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # Save
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    # Build index entry
    entry = {
        "saved":       datetime.now().isoformat(),
        "path":        str(out_path),
        "notebook":    notebook,
        "description": description,
        "experiment":  experiment or "",
        "dpi":         dpi,
        "fmt":         fmt,
    }

    # Append to figure index
    index = _load_index()
    index.append(entry)
    _save_index(index)

    # Append to run.log
    exp_str = f" | experiment={experiment}" if experiment else ""
    line = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | FIGURE   | "
        f"{notebook:<30} | {description:<30} | {out_path}{exp_str}"
    )
    _append_run_log(line)

    # Also log to config audit trail if config is available
    try:
        from src.config import config
        config.log_event(
            phase="figure_save",
            action="save_figure",
            details={
                "path":        str(out_path),
                "description": description,
                "experiment":  experiment or "",
                "dpi":         dpi,
                "fmt":         fmt,
            },
            notebook=notebook,
        )
    except Exception:
        pass  # Config not available in all contexts — fail silently

    print(f"📊 Figure saved: {out_path.relative_to(_project_root())}")
    return out_path


def list_figures(
    notebook: Optional[str] = None,
    experiment: Optional[str] = None,
) -> list[dict]:
    """
    Return saved figure entries, optionally filtered.

    Parameters
    ----------
    notebook : str, optional
        Filter to figures from this notebook/script.
    experiment : str, optional
        Filter to figures associated with this experiment.

    Returns
    -------
    list[dict]
        List of figure index entries matching the filter.

    Examples
    --------
    >>> list_figures(notebook="01-EDA_SOAP")
    [{'saved': '...', 'path': '...', 'notebook': '01-EDA_SOAP', ...}]

    >>> list_figures(experiment="E-009_Balanced_E002Init")
    [...]
    """
    index = _load_index()
    results = index

    if notebook:
        results = [e for e in results if e.get("notebook") == notebook]
    if experiment:
        results = [e for e in results if e.get("experiment") == experiment]

    return results


def figure_report() -> None:
    """
    Print a summary table of all saved figures.

    Useful for finding figures when writing a paper.

    Example
    -------
    >>> from src.plot_utils import figure_report
    >>> figure_report()

    ============================================================
     Figure Registry — 2026-04-26 09:30:00
    ============================================================
     Notebook                       Description                    Experiment
     ------------------------------ ------------------------------ ----------
     01-EDA_SOAP                    token_pressure_kde             —
     01-EDA_SOAP                    icd10_frequency_histogram      —
     evaluate                       chapter_accuracy               E-009_Balanced_E002Init
    ============================================================
     Total: 3 figures
     Index: outputs/figure_index.json
    ============================================================
    """
    index = _load_index()

    print(f"\n{'='*70}")
    print(f" Figure Registry — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    if not index:
        print(" No figures saved yet.")
        print(f"{'='*70}\n")
        return

    print(f" {'Notebook':<30} {'Description':<30} {'Experiment'}")
    print(f" {'-'*30} {'-'*30} {'-'*20}")

    for entry in sorted(index, key=lambda x: x.get("saved", "")):
        nb  = entry.get("notebook", "—")[:29]
        desc = entry.get("description", "—")[:29]
        exp  = entry.get("experiment", "—") or "—"
        print(f" {nb:<30} {desc:<30} {exp}")

    print(f"{'='*70}")
    print(f" Total: {len(index)} figure(s)")
    print(f" Index: outputs/figure_index.json")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Convenience wrapper for the common notebook pattern
# ---------------------------------------------------------------------------

def show_and_save(
    fig,
    notebook: str,
    description: str,
    experiment: Optional[str] = None,
    dpi: int = 150,
) -> Path:
    """
    Save a figure then display it inline — the standard notebook pattern.

    Replaces the common sequence:
        plt.show()
        plt.close()

    With:
        show_and_save(fig, "01-EDA_SOAP", "token_pressure_kde")

    The figure is saved first (so it's never lost), then displayed,
    then closed to prevent matplotlib memory leaks.

    Returns the saved Path.
    """
    import matplotlib.pyplot as plt
    path = save_figure(fig, notebook=notebook, description=description,
                       experiment=experiment, dpi=dpi)
    plt.show()
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        figure_report()
    elif len(sys.argv) > 2 and sys.argv[1] == "list":
        figures = list_figures(notebook=sys.argv[2] if len(sys.argv) > 2 else None)
        for f in figures:
            print(f"{f['saved'][:19]}  {f['description']:<35}  {f['path']}")
    else:
        print("Usage:")
        print("  uv run python src/plot_utils.py report")
        print("  uv run python src/plot_utils.py list 01-EDA_SOAP")