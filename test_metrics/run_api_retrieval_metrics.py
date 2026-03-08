#!/usr/bin/env python3
"""Run API retrieval evaluation and compute MRR@K, Precision@K, Hit@K."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import time
from pathlib import Path
from typing import Any

import httpx
import yaml


def load_test_metrics_config(config_path: Path) -> dict[str, Any]:
    """Load and validate YAML config for API retrieval-metrics evaluation."""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {config_path}")
    print(f"Loaded config from {config_path}: {json.dumps(data, ensure_ascii=True)}")
    return data


def get_nested(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a nested config value by dot-separated key path with a default fallback."""
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def parse_gold_ids(row: dict[str, str]) -> list[str]:
    """Extract gold feed IDs from a ground-truth CSV row."""
    gold_titles = (row.get("gold_titles") or "").strip()
    if gold_titles:
        try:
            parsed = ast.literal_eval(gold_titles)
            if isinstance(parsed, list):
                vals = [str(x).strip() for x in parsed if str(x).strip()]
                if vals:
                    return vals
        except (SyntaxError, ValueError):
            pass

    out: list[str] = []
    for i in range(1, 51):
        key = f"Gold#{i} title"
        if key not in row:
            break
        raw = (row.get(key) or "").strip()
        if not raw:
            continue
        feed_id = raw.split("|", 1)[0].strip()
        if feed_id:
            out.append(feed_id)
    print(f"Parsed gold_ids for student_id={row.get('student_id', '')}: {out}")
    return out


def load_groundtruth(groundtruth_csv: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Load student IDs and their gold feed IDs from the ground-truth CSV file."""
    student_ids: list[str] = []
    student_to_gold: dict[str, list[str]] = {}

    with groundtruth_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing header row in {groundtruth_csv}")

        student_key = None
        for k in reader.fieldnames:
            if isinstance(k, str) and k.strip().lower() == "student_id":
                student_key = k
                break
        if not student_key:
            raise ValueError(f'Missing "student_id" column in {groundtruth_csv}')

        for row in reader:
            student_id = (row.get(student_key) or "").strip()
            if not student_id:
                continue
            if student_id not in student_to_gold:
                student_ids.append(student_id)
            student_to_gold[student_id] = parse_gold_ids(row)
    print(f"Loaded {len(student_ids)} student_ids from ground truth CSV {groundtruth_csv}")
    return student_ids, student_to_gold


def compute_metrics_at_k(predicted_ids: list[str], gold_ids: list[str], k: int) -> dict[str, float | int]:
    """Compute MRR@K, Precision@K, Hit@K, and hit count for one student result."""
    top_k = predicted_ids[:k]
    gold_set = {x for x in gold_ids if x}
    if not gold_set:
        return {"mrr": 0.0, "precision": 0.0, "hit": 0.0, "hit_count": 0}

    first_rank = None
    hit_count = 0
    for idx, pred_id in enumerate(top_k, start=1):
        if pred_id in gold_set:
            hit_count += 1
            if first_rank is None:
                first_rank = idx

    mrr = 0.0 if first_rank is None else 1.0 / float(first_rank)
    precision = float(hit_count) / float(k)
    hit = 1.0 if hit_count > 0 else 0.0
    print(f"Computed metrics at k={k}: MRR={mrr}, Precision={precision}, Hit={hit}, hit_count={hit_count}")
    return {"mrr": mrr, "precision": precision, "hit": hit, "hit_count": hit_count}


def extract_candidates(response_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract recommendation candidate objects from API response payload."""
    raw = response_json.get("recommendations", [])
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def parse_header_latency_ms(headers: httpx.Headers) -> float | None:
    """Parse `x-response-time-seconds` header and return latency in milliseconds."""
    value = headers.get("x-response-time-seconds")
    if value is None:
        return None
    try:
        return float(value) * 1000.0
    except ValueError:
        return None


def build_api_url(cfg: dict[str, Any]) -> str:
    """Build full API URL from config values `api.base` and `api.route`."""
    base = str(get_nested(cfg, "api.base", "")).rstrip("/")
    route = str(get_nested(cfg, "api.route", "")).lstrip("/")
    if not base or not route:
        raise ValueError("Config must include api.base and api.route")
    return f"{base}/{route}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for config path and optional student evaluation limit."""
    parser = argparse.ArgumentParser(
        description="Call recommendation API for each student and compute retrieval metrics."
    )
    parser.add_argument(
        "--config",
        default="test_metrics/test_metrics_config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, evaluate only first N student_ids from ground truth",
    )
    return parser.parse_args()


def main() -> int:
    """Run API evaluation for all target students and write detailed metrics to CSV."""
    args = parse_args()
    cfg = load_test_metrics_config(Path(args.config))

    api_url = build_api_url(cfg)
    timeout_seconds = float(get_nested(cfg, "timeout_seconds", 60))
    top_k = int(get_nested(cfg, "evaluation.top_k", 10))
    if top_k <= 0:
        raise ValueError("evaluation.top_k must be > 0")

    groundtruth_csv = Path(str(get_nested(cfg, "data.groundtruth_csv", "")))
    output_csv = Path(str(get_nested(cfg, "output.csv", "")))
    if not groundtruth_csv:
        raise ValueError("data.groundtruth_csv is required")
    if not output_csv:
        raise ValueError("output.csv is required")

    student_ids, student_to_gold = load_groundtruth(groundtruth_csv)
    if args.limit > 0:
        student_ids = student_ids[: args.limit]
    if not student_ids:
        raise ValueError("No student_id rows found in ground truth CSV")

    fieldnames = [
        "student_id",
        "status_code",
        "source",
        "latency_ms_client",
        "latency_ms_header",
        "error",
        "gold_ids",
        "predicted_ids_top_k",
        f"MRR@{top_k}",
        f"Precision@{top_k}",
        f"Hit@{top_k}",
        "hit_count",
    ]
    for i in range(1, top_k + 1):
        fieldnames.append(f"top_{i}_feed_id")
        fieldnames.append(f"top_{i}_score")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=timeout) as client:
        for student_id in student_ids:
            payload: dict[str, Any] = {"student_id": student_id}

            row = {k: "" for k in fieldnames}
            row["student_id"] = student_id

            started = time.perf_counter()
            try:
                print(f"Calling API for student_id={student_id} with payload={payload}")
                response = client.post(
                    api_url,
                    headers={
                        "accept": "application/json",
                        "content-type": "application/json",
                    },
                    json=payload,
                )
                print(f"Received response for student_id={student_id}: status_code={response.status_code}")
                client_ms = (time.perf_counter() - started) * 1000.0
                row["status_code"] = response.status_code
                row["latency_ms_client"] = round(client_ms, 3)
                header_ms = parse_header_latency_ms(response.headers)
                row["latency_ms_header"] = "" if header_ms is None else round(header_ms, 3)

                if response.status_code >= 400:
                    row["error"] = f"status={response.status_code} body={response.text}"
                    rows.append(row)
                    continue

                response_json = response.json()
                row["source"] = str(response_json.get("source", ""))
                if not isinstance(response_json.get("recommendations"), list):
                    raise ValueError("Invalid response: `recommendations` must be a list")
                candidates = extract_candidates(response_json)
                predicted_ids: list[str] = []
                scores: list[float | None] = []

                print(f"Extracted {len(candidates)} candidates for student_id={student_id}")
                for cand in candidates:
                    feed_id = cand.get("feed_id")
                    score = cand.get("score")
                    predicted_ids.append(str(feed_id) if feed_id is not None else "")
                    try:
                        scores.append(float(score) if score is not None else None)
                    except (TypeError, ValueError):
                        scores.append(None)

                predicted_ids = [x for x in predicted_ids if x]
                gold_ids = student_to_gold.get(student_id, [])
                metrics = compute_metrics_at_k(predicted_ids, gold_ids, top_k)

                print(f"Metrics for student_id={student_id}: {json.dumps(metrics, ensure_ascii=True)}")

                row["gold_ids"] = " | ".join(gold_ids)
                row["predicted_ids_top_k"] = " | ".join(predicted_ids[:top_k])
                row[f"MRR@{top_k}"] = round(float(metrics["mrr"]), 6)
                row[f"Precision@{top_k}"] = round(float(metrics["precision"]), 6)
                row[f"Hit@{top_k}"] = round(float(metrics["hit"]), 6)
                row["hit_count"] = int(metrics["hit_count"])

                for i in range(1, top_k + 1):
                    idx = i - 1
                    row[f"top_{i}_feed_id"] = predicted_ids[idx] if idx < len(predicted_ids) else ""
                    score_val = scores[idx] if idx < len(scores) else None
                    row[f"top_{i}_score"] = "" if score_val is None else score_val

            except Exception as exc:  # noqa: BLE001
                client_ms = (time.perf_counter() - started) * 1000.0
                row["latency_ms_client"] = round(client_ms, 3)
                row["error"] = str(exc)

            print(f"Completed evaluation for student_id={student_id}: {json.dumps(row, ensure_ascii=True)}")
            rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [r for r in rows if not r.get("error")]
    if valid_rows:
        mrr_key = f"MRR@{top_k}"
        p_key = f"Precision@{top_k}"
        h_key = f"Hit@{top_k}"
        mean_mrr = sum(float(r[mrr_key]) for r in valid_rows) / len(valid_rows)
        mean_precision = sum(float(r[p_key]) for r in valid_rows) / len(valid_rows)
        mean_hit = sum(float(r[h_key]) for r in valid_rows) / len(valid_rows)
        print(
            json.dumps(
                {
                    "evaluated_students": len(rows),
                    "successful_students": len(valid_rows),
                    f"mean_{mrr_key}": round(mean_mrr, 6),
                    f"mean_{p_key}": round(mean_precision, 6),
                    f"mean_{h_key}": round(mean_hit, 6),
                    "output_csv": str(output_csv),
                },
                ensure_ascii=True,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "evaluated_students": len(rows),
                    "successful_students": 0,
                    "output_csv": str(output_csv),
                },
                ensure_ascii=True,
            )
        )

    print(f"Saved detailed results to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
