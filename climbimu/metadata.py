"""Labels for each recording: from the file name, from a table, or both.

Three levels, from simple to flexible:

1. Nothing: the file name (without extension) is only an ``id``.
2. A file name pattern, e.g. ``"{climber}_{attempt}"``. You choose the
   field names, so other studies can use ``"{climber}_{boulder}_{condition}"``.
3. A metadata table (CSV or Excel) with one row per file and any columns
   you want. If the table exists, its values override the file name.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd


def pattern_to_regex(pattern: str) -> re.Pattern:
    """Turn ``"{climber}_{attempt}"`` into a regular expression."""
    parts = re.split(r"(\{[A-Za-z_][A-Za-z0-9_]*\})", pattern)
    regex = ""
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            regex += f"(?P<{part[1:-1]}>.+?)"
        else:
            regex += re.escape(part)
    return re.compile(f"^{regex}$")


def parse_filename(stem: str, pattern: str | None) -> dict:
    """Read labels from a file name. Returns an empty dict if it does not match."""
    if not pattern:
        return {}
    match = pattern_to_regex(pattern).match(stem)
    if match is None:
        warnings.warn(
            f"File name '{stem}' does not match the pattern '{pattern}'. "
            "Only the id will be used."
        )
        return {}
    return {key: _to_number(value) for key, value in match.groupdict().items()}


def load_metadata_table(path: str | Path, id_column: str = "file") -> dict:
    """Load a table with one row per recording.

    The ``id_column`` must contain the file name (with or without extension).
    Returns a dict ``{file_stem: {column: value, ...}}``.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        table = pd.read_excel(path)
    else:
        table = pd.read_csv(path)
    if id_column not in table:
        raise ValueError(f"The metadata table has no column '{id_column}'.")
    result = {}
    for _, row in table.iterrows():
        stem = Path(str(row[id_column])).stem
        values = {k: v for k, v in row.items() if k != id_column and pd.notna(v)}
        result[stem] = values
    return result


def get_metadata(file_path: str | Path, config: dict) -> dict:
    """Build the labels of one file following the configuration."""
    stem = Path(file_path).stem
    settings = config.get("metadata", {}) or {}
    meta = {"id": stem}
    meta.update(parse_filename(stem, settings.get("filename_pattern")))
    table_path = settings.get("table")
    if table_path:
        table = load_metadata_table(table_path, settings.get("id_column", "file"))
        meta.update(table.get(stem, {}))
    return meta


def _to_number(value: str):
    """'1' -> 1, '2.5' -> 2.5, 'A' -> 'A'."""
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            pass
    return value
