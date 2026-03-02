#!/usr/bin/env python3
"""Prepare per-student mock data directories from source JSON files."""

from __future__ import annotations

import json
from pathlib import Path

HYDE_PATH = Path("test/data/students_1k_hyde.json")
METADATA_PATH = Path("test/data/students_1k_metadata.json")
FEEDS_20K_PATH = Path("test/data/feeds_20k.jsonl")
OUTPUT_ROOT = Path("test/student_data")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_feed_ids(path: Path) -> set[str]:
    feed_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}: {exc}") from exc

            feed_id = record.get("feed_id")
            if feed_id:
                feed_ids.add(str(feed_id))
    return feed_ids


def main() -> None:
    hyde_records = load_json(HYDE_PATH)
    metadata_records = load_json(METADATA_PATH)

    if not isinstance(hyde_records, list) or not isinstance(metadata_records, list):
        raise ValueError("Both input files must be JSON arrays.")

    hyde_by_student = {}
    for rec in hyde_records:
        sid = rec.get("student_id")
        if not sid:
            raise ValueError("Found hyde record without student_id.")
        hyde_by_student[sid] = rec.get("hq", [])

    metadata_by_student = {}
    for rec in metadata_records:
        sid = rec.get("student_id")
        if not sid:
            raise ValueError("Found metadata record without student_id.")
        metadata_by_student[sid] = rec

    hyde_ids = set(hyde_by_student)
    metadata_ids = set(metadata_by_student)

    missing_in_hyde = sorted(metadata_ids - hyde_ids)
    missing_in_metadata = sorted(hyde_ids - metadata_ids)
    if missing_in_hyde or missing_in_metadata:
        details = []
        if missing_in_hyde:
            details.append(
                f"student_id(s) in metadata but missing in hyde: {missing_in_hyde[:10]}"
            )
        if missing_in_metadata:
            details.append(
                f"student_id(s) in hyde but missing in metadata: {missing_in_metadata[:10]}"
            )
        raise ValueError("Input student_id mismatch. " + " | ".join(details))

    valid_feed_ids = load_feed_ids(FEEDS_20K_PATH)

    filtered_metadata_by_student = {}
    ignored_interactions = 0
    students_with_ignored = 0
    for sid, metadata in metadata_by_student.items():
        interactions = metadata.get("interaction", [])
        kept_interactions = []
        for item in interactions:
            feed_id = item.get("feed_id")
            if feed_id and feed_id in valid_feed_ids:
                kept_interactions.append(item)
            else:
                ignored_interactions += 1

        if len(kept_interactions) != len(interactions):
            students_with_ignored += 1

        metadata_copy = dict(metadata)
        metadata_copy["interaction"] = kept_interactions
        filtered_metadata_by_student[sid] = metadata_copy

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for sid in sorted(hyde_ids):
        student_root = OUTPUT_ROOT / sid
        hyde_dir = student_root / "hyde"
        metadata_dir = student_root / "metadata"

        hyde_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        hyde_payload = {"hq": hyde_by_student[sid]}
        metadata_payload = filtered_metadata_by_student[sid]

        with (hyde_dir / "hyde.json").open("w", encoding="utf-8") as f:
            json.dump(hyde_payload, f, ensure_ascii=True)

        with (metadata_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, ensure_ascii=True)

    print(f"Done. Wrote {len(hyde_ids)} student folders under {OUTPUT_ROOT}")
    print(
        "Ignored interaction rows with feed_id not found in test/data/feeds_20k.jsonl: "
        f"{ignored_interactions} row(s) across {students_with_ignored} student(s)"
    )


if __name__ == "__main__":
    main()
