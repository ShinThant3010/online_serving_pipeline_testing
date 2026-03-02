#!/usr/bin/env python3
"""Analyze interaction counts before/after mock student data preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
BEFORE_METADATA_PATH = BASE_DIR / "data" / "students_1k_metadata.json"
AFTER_STUDENT_DATA_DIR = BASE_DIR / "student_data"


def load_before_counts(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected JSON array in {path}")

    counts: dict[str, int] = {}
    for rec in records:
        student_id = rec.get("student_id")
        if not student_id:
            continue
        interactions = rec.get("interaction", [])
        counts[student_id] = len(interactions) if isinstance(interactions, list) else 0
    return counts


def load_after_counts(root_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}

    if not root_dir.exists():
        return counts

    for student_dir in root_dir.iterdir():
        if not student_dir.is_dir():
            continue

        metadata_path = student_dir / "metadata" / "metadata.json"
        if not metadata_path.exists():
            continue

        with metadata_path.open("r", encoding="utf-8") as f:
            rec = json.load(f)

        student_id = rec.get("student_id", student_dir.name)
        interactions = rec.get("interaction", [])
        counts[student_id] = len(interactions) if isinstance(interactions, list) else 0

    return counts


def build_dataframe(before_counts: dict[str, int], after_counts: dict[str, int]) -> pd.DataFrame:
    student_ids = sorted(set(before_counts) | set(after_counts))
    rows = [
        {
            "student_id": sid,
            "interactions_count_before": before_counts.get(sid, 0),
            "interactions_count_after": after_counts.get(sid, 0),
        }
        for sid in student_ids
    ]
    return pd.DataFrame(rows)


def main() -> None:
    before_counts = load_before_counts(BEFORE_METADATA_PATH)
    after_counts = load_after_counts(AFTER_STUDENT_DATA_DIR)

    df = build_dataframe(before_counts, after_counts)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    print(df)


if __name__ == "__main__":
    main()
