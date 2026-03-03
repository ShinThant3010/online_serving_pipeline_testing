#!/usr/bin/env python3
"""Run fixed-count API calls and report runtime statistics for endpoint/index comparison."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import httpx
import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {config_path}")
    return data


def get_nested(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_api_url(cfg: dict[str, Any]) -> str:
    base = str(get_nested(cfg, "api.base", "")).rstrip("/")
    route = str(get_nested(cfg, "api.route", "")).lstrip("/")
    if not base or not route:
        raise ValueError("Config must include api.base and api.route")
    return f"{base}/{route}"


def load_student_ids(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []
    student_ids: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        student_key = next(
            (k for k in reader.fieldnames if isinstance(k, str) and k.strip().lower() == "student_id"),
            None,
        )
        if not student_key:
            return []
        seen: set[str] = set()
        for row in reader:
            student_id = (row.get(student_key) or "").strip()
            if not student_id or student_id in seen:
                continue
            seen.add(student_id)
            student_ids.append(student_id)
    return student_ids


def parse_header_latency_ms(headers: httpx.Headers) -> float | None:
    value = headers.get("x-response-time-seconds")
    if value is None:
        return None
    try:
        return float(value) * 1000.0
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run recommendation API requests and compare average runtime across vertex configs."
    )
    parser.add_argument(
        "--config",
        default="test_metrics/test_metrics_config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=200,
        help="Number of API requests to send",
    )
    parser.add_argument(
        "--index-endpoint",
        default="",
        help="Override vertex.index_endpoint for request payload",
    )
    parser.add_argument(
        "--deployed-index-id",
        default="",
        help="Override vertex.deployed_index_id for request payload",
    )
    parser.add_argument(
        "--output-csv",
        default="test_metrics/data/runtime_comparison_results.csv",
        help="Output CSV path for per-request timings",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.requests <= 0:
        raise ValueError("--requests must be > 0")

    print(f"Using config: {args.config}")
    cfg = load_config(Path(args.config))
    api_url = build_api_url(cfg)
    timeout_seconds = float(get_nested(cfg, "timeout_seconds", 60))

    configured_endpoint = str(get_nested(cfg, "search.endpoint_id", "")).strip()
    configured_deployed_id = str(get_nested(cfg, "search.deployed_index_id", "")).strip()
    index_endpoint = args.index_endpoint.strip() or configured_endpoint
    deployed_index_id = args.deployed_index_id.strip() or configured_deployed_id

    include_vertex = bool(index_endpoint and deployed_index_id)
    if (index_endpoint and not deployed_index_id) or (deployed_index_id and not index_endpoint):
        raise ValueError("Both index_endpoint and deployed_index_id must be provided together")

    groundtruth_csv = Path(str(get_nested(cfg, "data.groundtruth_csv", "")))
    student_ids = load_student_ids(groundtruth_csv) if str(groundtruth_csv) else []
    print(f"Loaded {len(student_ids)} student IDs from {groundtruth_csv}") if student_ids else print("No student IDs loaded from config")
    
    if not student_ids:
        student_ids = [f"student_{i:04d}" for i in range(1, 201)]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "request_no",
        "student_id",
        "status_code",
        "source",
        "latency_ms_client",
        "latency_ms_header",
        "error",
    ]

    rows: list[dict[str, Any]] = []
    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=timeout) as client:
        for i in range(args.requests):
            print(f"Sending request {i + 1}/{args.requests}...", end="\r")
            student_id = student_ids[i % len(student_ids)]
            payload: dict[str, Any] = {"student_id": student_id}
            if include_vertex:
                payload["vertex"] = {
                    "index_endpoint": index_endpoint,
                    "deployed_index_id": deployed_index_id,
                }

            row = {k: "" for k in fieldnames}
            row["request_no"] = i + 1
            row["student_id"] = student_id

            started = time.perf_counter()
            try:
                print(f"Sending request {i + 1}/{args.requests} with payload: {json.dumps(payload)}")  # Debug log
                response = client.post(
                    api_url,
                    headers={"accept": "application/json", "content-type": "application/json"},
                    json=payload,
                )
                print(f"Received response for request {i + 1}/{args.requests}: status={response.status_code} body={response.text[:200]}")  # Debug log
                row["status_code"] = response.status_code
                row["latency_ms_client"] = round((time.perf_counter() - started) * 1000.0, 3)
                header_ms = parse_header_latency_ms(response.headers)
                row["latency_ms_header"] = "" if header_ms is None else round(header_ms, 3)

                if response.status_code >= 400:
                    row["error"] = f"status={response.status_code} body={response.text[:200]}"
                else:
                    try:
                        data = response.json()
                        row["source"] = str(data.get("source", "")) if isinstance(data, dict) else ""
                    except ValueError:
                        row["error"] = "invalid_json_response"
            except Exception as exc:  # noqa: BLE001
                row["latency_ms_client"] = round((time.perf_counter() - started) * 1000.0, 3)
                row["error"] = str(exc)

            rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        print(f"Writing results to {output_csv}...")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful_rows = [r for r in rows if not r.get("error")]
    avg_client_ms = (
        sum(float(r["latency_ms_client"]) for r in successful_rows) / len(successful_rows)
        if successful_rows
        else 0.0
    )
    header_values = [
        float(r["latency_ms_header"])
        for r in successful_rows
        if str(r.get("latency_ms_header", "")).strip() != ""
    ]
    avg_header_ms = sum(header_values) / len(header_values) if header_values else 0.0

    print(
        json.dumps(
            {
                "api_url": api_url,
                "requests_sent": args.requests,
                "successful_requests": len(successful_rows),
                "failed_requests": len(rows) - len(successful_rows),
                "average_latency_ms_client": round(avg_client_ms, 3),
                "average_latency_ms_header": round(avg_header_ms, 3) if header_values else None,
                "index_endpoint": index_endpoint if include_vertex else "(default-from-server-config)",
                "deployed_index_id": deployed_index_id if include_vertex else "(default-from-server-config)",
                "output_csv": str(output_csv),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
