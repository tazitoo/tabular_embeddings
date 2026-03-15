"""Canonical PROJECT_ROOT for all scripts, regardless of subdirectory depth."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
