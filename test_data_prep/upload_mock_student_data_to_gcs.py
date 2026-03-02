#!/usr/bin/env python3
"""Upload a range of mock student folders from test/student_data to GCS."""

from __future__ import annotations

import argparse
from pathlib import Path

from google.cloud import storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload student folders (e.g., stu_p501..stu_p700) from test/student_data "
            "to a GCS bucket, preserving relative paths."
        )
    )
    parser.add_argument(
        "--bucket",
        default="hyde-datalake-feeds",
        help="Destination GCS bucket name",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("test/student_data"),
        help="Local root containing per-student directories",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=501,
        help="Start numeric suffix for stu_pXXX (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=700,
        help="End numeric suffix for stu_pXXX (inclusive)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional GCS key prefix (e.g., 'mock/')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads without uploading",
    )
    return parser.parse_args()


def student_id_from_number(number: int) -> str:
    return f"stu_p{number:03d}"


def list_files_for_student(student_dir: Path) -> list[Path]:
    if not student_dir.exists() or not student_dir.is_dir():
        return []
    return sorted(path for path in student_dir.rglob("*") if path.is_file())


def build_blob_name(*, source_root: Path, file_path: Path, prefix: str) -> str:
    rel = file_path.relative_to(source_root).as_posix()
    normalized_prefix = prefix.strip("/")
    if normalized_prefix:
        return f"{normalized_prefix}/{rel}"
    return rel


def main() -> None:
    args = parse_args()

    if args.start > args.end:
        raise ValueError("--start must be less than or equal to --end")

    source_root = args.source_root
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {source_root}")

    planned: list[tuple[Path, str]] = []
    missing_students: list[str] = []

    for n in range(args.start, args.end + 1):
        print(f"Processing student number: {n}")
        student_id = student_id_from_number(n)
        student_dir = source_root / student_id
        files = list_files_for_student(student_dir)
        if not files:
            missing_students.append(student_id)
            continue

        for file_path in files:
            blob_name = build_blob_name(
                source_root=source_root,
                file_path=file_path,
                prefix=args.prefix,
            )
            planned.append((file_path, blob_name))

    if not planned:
        print("No files found to upload for the selected student range.")
        if missing_students:
            print(f"Missing/empty student folders: {len(missing_students)}")
        return

    print(f"Selected files: {len(planned)}")
    print(f"Bucket: gs://{args.bucket}")
    if args.prefix.strip("/"):
        print(f"Prefix: {args.prefix.strip('/')}")
    print(f"Student range: stu_p{args.start:03d}..stu_p{args.end:03d}")

    if missing_students:
        print(f"Missing/empty student folders: {len(missing_students)}")

    if args.dry_run:
        print("Dry run enabled. First 20 planned uploads:")
        for local_path, blob_name in planned[:20]:
            print(f"  {local_path} -> gs://{args.bucket}/{blob_name}")
        return

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    uploaded = 0
    for local_path, blob_name in planned:
        print(f"Uploading {local_path} to gs://{args.bucket}/{blob_name}...")
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        uploaded += 1

    print(f"Upload complete. Uploaded {uploaded} file(s) to gs://{args.bucket}")


if __name__ == "__main__":
    main()
