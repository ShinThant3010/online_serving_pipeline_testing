import os
import random
import csv
from pathlib import Path

from locust import HttpUser, between, task


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _load_student_ids_from_csv(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        student_key = None
        for key in reader.fieldnames:
            if isinstance(key, str) and key.strip().lower() == "student_id":
                student_key = key
                break
        if not student_key:
            return []

        values = []
        for row in reader:
            student_id = (row.get(student_key) or "").strip()
            if student_id:
                values.append(student_id)
        return _dedupe_keep_order(values)


def _student_ids() -> list[str]:
    locust_student_ids = os.getenv("LOCUST_STUDENT_IDS", "").strip()
    if locust_student_ids:
        values = [value.strip() for value in locust_student_ids.split(",") if value.strip()]
        if values:
            return _dedupe_keep_order(values)

    csv_path = Path(
        os.getenv(
            "LOCUST_STUDENT_IDS_CSV",
            "test_metrics/data/ground_truth_gold_top10.csv",
        )
    )
    values = _load_student_ids_from_csv(csv_path)
    if values:
        return values

    return [f"student_{i:04d}" for i in range(1, 101)]


STUDENT_IDS = _student_ids()
RECOMMEND_PATH = "/recommendations"


class RecommendationUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task(5)
    def recommend(self) -> None:
        student_id = random.choice(STUDENT_IDS)
        payload = {"student_id": student_id}
        with self.client.post(
            RECOMMEND_PATH,
            json=payload,
            name="POST /recommendations",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code} body={response.text[:200]}")
                return
            try:
                data = response.json()
            except ValueError:
                response.failure("invalid_json_response")
                return
            if not isinstance(data, dict) or "recommendations" not in data:
                response.failure("missing_recommendations_field")
