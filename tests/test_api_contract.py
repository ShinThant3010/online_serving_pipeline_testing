from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.app import app, get_recommendation_service
from api.schema import FeedsRecommendation, RecommendationResponse


class StubRecommendationService:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(app=SimpleNamespace(perf_log_sample_rate=0.0))

    def recommend(self, student_id: str):
        response = RecommendationResponse(
            student_id=student_id,
            source="redis_cache",
            recommendations=[FeedsRecommendation(feed_id="feed-1", score=0.99)],
            num_recommendations=1,
        )
        diagnostics = SimpleNamespace(
            cache_hit=True,
            t_cache_get=0.0,
            t_vector_search=0.0,
            t_postprocess=0.0,
            t_fallback_prepare=0.0,
            t_rerank=0.0,
            t_metadata_fetch=0.0,
            t_format_response=0.0,
            t_top_up_merge=0.0,
            num_recommendations=[],
        )
        return response, diagnostics


def test_health_includes_api_version_and_generated_correlation_id_headers():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-API-Version"] == app.version
    assert response.headers["X-Correlation-Id"].startswith("corr_")


def test_recommendations_accept_camel_case_request_and_return_camel_case_response():
    app.dependency_overrides[get_recommendation_service] = lambda: StubRecommendationService()
    client = TestClient(app)

    try:
        response = client.post(
            "/recommendations",
            json={"studentId": "student-123"},
            headers={"X-Correlation-Id": "corr-test-123"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["X-API-Version"] == app.version
    assert response.headers["X-Correlation-Id"] == "corr-test-123"
    assert response.json() == {
        "studentId": "student-123",
        "source": "redis_cache",
        "recommendations": [{"feedId": "feed-1", "score": 0.99, "metadata": None}],
        "numRecommendations": 1,
    }
