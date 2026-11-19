"""LASER's name in the paper's ELCD run records, mapped to "laser" for every reader of records.

The paper's records (not in the repository) keep the file names and fields they were written with, in which LASER is
PAPER_LASER; runs of this code write "laser". item_name maps the stem of a record file to its item, record_file finds
the file of an item. Standard library only, so that the readers in train/ need neither numpy nor pandas.
"""
from pathlib import Path

PAPER_LASER = "laser-lle"


def item_name(stem: str) -> str:
    """The item of a record file's stem: the stem, with PAPER_LASER at its start replaced by "laser"."""
    return "laser" + stem[len(PAPER_LASER):] if stem.startswith(PAPER_LASER) else stem


def record_file(path) -> Path:
    """path (<cell>/<item><suffix>), or the paper's file of the same LASER item where only that one exists."""
    path = Path(path)
    if path.exists() or not path.name.startswith("laser"):
        return path
    old = path.with_name(PAPER_LASER + path.name[len("laser"):])
    return old if old.exists() else path
