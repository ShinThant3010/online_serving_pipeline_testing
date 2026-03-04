import os
import random
import csv
from pathlib import Path

from locust import HttpUser, between, task
from locust.exception import CatchResponseError


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
CATCH_RESPONSE_ERROR_MODE = (os.getenv("LOCUST_CATCH_RESPONSE_ERROR_MODE", "mark") or "mark").strip().lower()


def _handle_catch_response_error(exc: CatchResponseError) -> None:
    if CATCH_RESPONSE_ERROR_MODE == "ignore":
        return
    if CATCH_RESPONSE_ERROR_MODE == "raise":
        raise exc
    # default: "mark" (response already marked failed)


class RecommendationUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task(5)
    def recommend(self) -> None:
        student_id = random.choice(STUDENT_IDS)
        payload = {"student_id": student_id}
        try:
            with self.client.post(
                RECOMMEND_PATH,
                json=payload,
                name="POST /recommendations",
                catch_response=True,
            ) as response:
                if response.status_code != 200:
                    message = f"status={response.status_code} body={response.text[:200]}"
                    response.failure(message)
                    if CATCH_RESPONSE_ERROR_MODE == "raise":
                        raise CatchResponseError(message)
                    return
                try:
                    data = response.json()
                except ValueError:
                    message = "invalid_json_response"
                    response.failure(message)
                    if CATCH_RESPONSE_ERROR_MODE == "raise":
                        raise CatchResponseError(message)
                    return
                if not isinstance(data, dict) or "recommendations" not in data:
                    message = "missing_recommendations_field"
                    response.failure(message)
                    if CATCH_RESPONSE_ERROR_MODE == "raise":
                        raise CatchResponseError(message)
        except CatchResponseError as exc:
            _handle_catch_response_error(exc)
