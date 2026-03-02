import os
import random

from locust import HttpUser, between, task


def _student_ids() -> list[str]:
    LOCUST_STUDENT_IDS="stu_p000,stu_p001,stu_p002,stu_p003,stu_p004,stu_p005,stu_p006,stu_p007,stu_p008,stu_p009,stu_p010,stu_p100"
    if LOCUST_STUDENT_IDS:
        values = [value.strip() for value in LOCUST_STUDENT_IDS.split(",") if value.strip()]
        if values:
            return values
    return [f"student_{i:04d}" for i in range(1, 101)]


STUDENT_IDS = _student_ids()
RECOMMEND_PATH = os.getenv("LOCUST_RECOMMEND_PATH", "/recommendations")
HEALTH_PATH = os.getenv("LOCUST_HEALTH_PATH", "/health")


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

    @task(1)
    def health(self) -> None:
        self.client.get(HEALTH_PATH, name="GET /health")
