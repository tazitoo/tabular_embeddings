"""Canonical path resolver for the paper repo.

Generators in scripts/paper/ and scripts/figures/ dual-write their
final artefacts to both output/paper_figures/<section>/ (working copy)
and the paper repo at PAPER_REPO/figures/<section>/.
"""
from pathlib import Path

PAPER_REPO = Path("/Users/brian/src/tabular_embedding_paper")
PAPER_FIGURES = PAPER_REPO / "figures"
PAPER_TABLES = PAPER_REPO / "tables"


def paper_figure_path(section: str, name: str) -> Path:
    """Resolve absolute path for a figure in the paper repo.

    Args:
        section: One of "1_intro", "4_results", "A_appendix",
                 "B_appendix", "C_appendix", "D_appendix",
                 "E_appendix", "F_appendix", "G_observational".
        name: Filename with .pdf extension.

    Returns:
        Absolute Path; parent dir guaranteed to exist.
    """
    p = PAPER_FIGURES / section / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def paper_table_path(name: str) -> Path:
    """Resolve absolute path for a .tex table in the paper repo."""
    p = PAPER_TABLES / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
