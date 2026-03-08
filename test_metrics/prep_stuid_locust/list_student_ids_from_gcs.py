#!/usr/bin/env python3
"""List student ID folders from a GCS path [hyde-datalake-feeds] and export to CSV for load testing (loscust)"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from google.cloud import storage


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a GCS URI into bucket name and normalized prefix (with trailing slash)."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")

    raw = gcs_uri[5:]
    bucket, _, prefix = raw.partition("/")
    if not bucket:
        raise ValueError("GCS URI must include bucket name")

    normalized_prefix = prefix.strip("/")
    if normalized_prefix:
        normalized_prefix = f"{normalized_prefix}/"
    return bucket, normalized_prefix


def list_student_ids(bucket_name: str, prefix: str) -> list[str]:
    """List direct child folder names under a GCS prefix as unique sorted student IDs."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # delimiter='/' returns only direct child folders for the given prefix.
    iterator = bucket.list_blobs(prefix=prefix, delimiter="/")

    student_ids: set[str] = set()
    for page in iterator.pages:
        for folder_prefix in page.prefixes:
            student_id = folder_prefix[len(prefix) :].strip("/")
            if student_id:
                student_ids.add(student_id)

    return sorted(student_ids)


def write_csv(student_ids: list[str], output_path: Path) -> None:
    """Write student IDs to a CSV file with a single `student_id` column."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id"])
        for student_id in student_ids:
            writer.writerow([student_id])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for source GCS URI and output CSV path."""
    parser = argparse.ArgumentParser(
        description=(
            "Read student-id folder names from GCS and save to CSV with column 'student_id'."
        )
    )
    parser.add_argument(
        "--gcs-uri",
        default="gs://hyde-datalake-feeds/",
        help="GCS URI to list direct child folders from",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_metrics/prep_stuid_locust/student_ids.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    """Run the GCS listing flow and export discovered student IDs to CSV."""
    args = parse_args()
    bucket_name, prefix = parse_gcs_uri(args.gcs_uri)

    student_ids = list_student_ids(bucket_name=bucket_name, prefix=prefix)
    write_csv(student_ids, args.output)

    print(f"Saved {len(student_ids)} student IDs to {args.output}")
    print("Sample student IDs (up to 5):")
    for student_id in student_ids[:5]:
        print(f"- {student_id}")
    print(f"Total student IDs: {len(student_ids)}")


if __name__ == "__main__":
    main()
