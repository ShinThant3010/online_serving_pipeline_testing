import random
import csv
from pathlib import Path

from locust import HttpUser, between, task


def _load_student_ids_from_csv(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Student ID CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        values: list[str] = []
        for row in reader:
            student_id = (row.get("student_id") or "").strip()
            if student_id:
                values.append(student_id)
        if not values:
            raise ValueError(f"No student_id found in {csv_path}")
        return values


STUDENT_IDS = _load_student_ids_from_csv(Path("test_data_prep/prep_students/student_ids.csv"))
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
