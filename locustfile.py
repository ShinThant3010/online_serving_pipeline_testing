import random
import csv
from pathlib import Path

from locust import HttpUser, between, task
from pydantic import BaseModel


def _load_student_ids_from_csv(csv_path: Path) -> list[str]:
    """Load non-empty `student_id` values from a CSV file and return them as a list."""
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


STUDENT_IDS = _load_student_ids_from_csv(Path("test_metrics/prep_stuid_locust/student_ids.csv"))
RECOMMEND_PATH = "/recommendations"

LOCUST_VERTEX_INDEX_ENDPOINT = "projects/810737581373/locations/asia-southeast1/indexEndpoints/6458557689835290624"
LOCUST_VERTEX_DEPLOYED_INDEX_ID = "feeds_20k_4replicas_deployed"


class VertexRequest(BaseModel):
    index_endpoint: str | None = None
    deployed_index_id: str | None = None


def _build_vertex_payload() -> dict[str, str] | None:
    """Build optional Vertex override payload for recommendation requests."""
    if not LOCUST_VERTEX_INDEX_ENDPOINT and not LOCUST_VERTEX_DEPLOYED_INDEX_ID:
        return None
    vertex = VertexRequest(
        index_endpoint=LOCUST_VERTEX_INDEX_ENDPOINT or None,
        deployed_index_id=LOCUST_VERTEX_DEPLOYED_INDEX_ID or None,
    )
    return vertex.model_dump(exclude_none=True)


class RecommendationUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task(5)
    def recommend(self) -> None:
        """Send a recommendation request for a random student and validate response shape."""
        student_id = random.choice(STUDENT_IDS)
        payload = {"student_id": student_id}
        vertex_payload = _build_vertex_payload()
        if vertex_payload:
            payload["vertex"] = vertex_payload
        with self.client.post(
            RECOMMEND_PATH,
            json=payload,
            name="POST /recommendations",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                message = f"status={response.status_code} body={response.text[:200]}"
                response.failure(message)
                return
            try:
                data = response.json()
            except ValueError:
                print(
                    "DEBUG validation_fail invalid_json "
                    f"status={response.status_code} "
                    f"content_type={response.headers.get('content-type')} "
                    f"body={response.text[:300]}"
                )
                response.failure("invalid_json_response")
                return
            if not isinstance(data, dict) or "recommendations" not in data:
                print(
                    "DEBUG validation_fail missing_recommendations "
                    f"status={response.status_code} "
                    f"content_type={response.headers.get('content-type')} "
                    f"body={response.text[:300]}"
                )
                response.failure("missing_recommendations_field")
