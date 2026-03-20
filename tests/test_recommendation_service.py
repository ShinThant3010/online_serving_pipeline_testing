from types import SimpleNamespace
from unittest.mock import MagicMock

from api.schema import FeedsMetadata, FeedsRecommendation, RecommendationResponse
from modules.core.recommend_feeds import PostprocessTimings, RecommendationService


def _make_settings(minimum_recommendation: int = 3, fallback_limit: int = 5, ttl_seconds: int = 60):
    return SimpleNamespace(
        recommendation=SimpleNamespace(minimum_recommendation=minimum_recommendation),
        bigquery=SimpleNamespace(fallback_limit=fallback_limit, fallback_table="project.dataset.table"),
        cache=SimpleNamespace(ttl_seconds=ttl_seconds),
    )


def _make_service(settings=None) -> RecommendationService:
    service = RecommendationService.__new__(RecommendationService)
    service.settings = settings or _make_settings()
    service.redis_cache = MagicMock()
    service.embedding_store = MagicMock()
    service.vector_search = MagicMock()
    service.bigquery_client = MagicMock()
    service.trigger_hyde_generation_service = MagicMock()
    return service


def test_get_cached_response_overrides_source_with_redis_cache():
    service = _make_service()
    service.redis_cache.get_one.return_value = {
        "student_id": "s-1",
        "source": "vertex_vector_search",
        "recommendations": [{"feed_id": "f-1", "score": 0.8}],
    }

    response = service._get_cached_response("recommendations:s-1")

    assert response is not None
    assert response.student_id == "s-1"
    assert response.source == "redis_cache"
    assert len(response.recommendations) == 1


def test_recommend_returns_cached_response_when_available():
    service = _make_service()
    cached = RecommendationResponse(
        student_id="s-1",
        source="redis_cache",
        recommendations=[FeedsRecommendation(feed_id="f-1", score=0.9)],
        num_recommendations=1,
    )
    service._get_cached_response = MagicMock(return_value=cached)

    response, diagnostics = service.recommend("s-1")

    assert response == cached
    assert diagnostics.cache_hit is True
    service.embedding_store.load_embeddings.assert_not_called()
    service.redis_cache.set_one.assert_not_called()


def test_recommend_falls_back_and_triggers_refresh_when_embeddings_missing():
    service = _make_service()
    service._get_cached_response = MagicMock(return_value=None)
    service.embedding_store.load_embeddings.return_value = []
    fallback = RecommendationResponse(
        student_id="s-2",
        source="bigquery_fallback",
        recommendations=[],
        num_recommendations=0,
    )
    service._build_fallback_response = MagicMock(return_value=(fallback, PostprocessTimings(t_fallback_prepare=0.1)))

    response, diagnostics = service.recommend("s-2")

    assert response == fallback
    assert diagnostics.cache_hit is False
    assert diagnostics.t_fallback_prepare == 0.1
    service._build_fallback_response.assert_called_once_with(student_id="s-2", trigger_refresh=True)
    service.redis_cache.set_one.assert_not_called()


def test_recommend_tops_up_with_fallback_when_vector_results_below_minimum():
    service = _make_service(settings=_make_settings(minimum_recommendation=3))
    service._get_cached_response = MagicMock(return_value=None)
    service.embedding_store.load_embeddings.return_value = [[0.1, 0.2]]

    vector_response = RecommendationResponse(
        student_id="s-3",
        source="vertex_vector_search",
        recommendations=[
            FeedsRecommendation(feed_id="f-1", score=0.8),
            FeedsRecommendation(feed_id="f-2", score=0.7),
        ],
        num_recommendations=2,
    )
    fallback_response = RecommendationResponse(
        student_id="s-3",
        source="bigquery_fallback",
        recommendations=[
            FeedsRecommendation(feed_id="f-2", score=0.2),
            FeedsRecommendation(feed_id="f-3", score=0.1),
        ],
        num_recommendations=2,
    )

    service._build_vector_response = MagicMock(
        return_value=(vector_response, 0.01, PostprocessTimings(t_rerank=0.02, t_format_response=0.03), [2])
    )
    service._build_fallback_response = MagicMock(
        return_value=(fallback_response, PostprocessTimings(t_fallback_prepare=0.04, t_rerank=0.05))
    )

    response, diagnostics = service.recommend("s-3")

    assert diagnostics.cache_hit is False
    assert response.source == "vertex_vector_search+bigquery_fallback"
    assert [item.feed_id for item in response.recommendations] == ["f-1", "f-2", "f-3"]
    assert response.num_recommendations == 3
    assert diagnostics.t_rerank == 0.07
    assert diagnostics.t_fallback_prepare == 0.04
    assert diagnostics.t_format_response == 0.03
    assert diagnostics.t_top_up_merge >= 0.0
    service._build_fallback_response.assert_called_once_with(student_id="s-3", trigger_refresh=False)
    service.redis_cache.set_one.assert_called_once()


def test_build_fallback_response_uses_redis_cache_when_sufficient_feed_keys(monkeypatch):
    service = _make_service(settings=_make_settings(fallback_limit=2))
    service.redis_cache.get_many_by_prefix.return_value = ["feeds:f-1", "feeds:f-2", "feeds:f-3"]
    service.redis_cache.get_many.return_value = {
        "feeds:f-1": {"title": "A"},
        "feeds:f-2": {"title": "B"},
    }

    def fake_rerank_with_subscore(*, student_id, feed_matrix, embedding_store, score_matrix=None):
        assert student_id == "s-4"
        assert feed_matrix == [["f-1", "f-2"]]
        return [
            {"feed_id": "f-2", "final_score": 0.9},
            {"feed_id": "f-1", "final_score": 0.5},
        ]

    monkeypatch.setattr("modules.core.recommend_feeds.rerank_with_subscore", fake_rerank_with_subscore)

    response, timings = service._build_fallback_response(student_id="s-4", trigger_refresh=False)

    assert response.source == "redis_fallback"
    assert [item.feed_id for item in response.recommendations] == ["f-2", "f-1"]
    assert all(isinstance(item.metadata, FeedsMetadata) for item in response.recommendations)
    assert timings.t_fallback_prepare >= 0.0
    assert timings.t_rerank >= 0.0
    assert timings.t_format_response >= 0.0
    service.trigger_hyde_generation_service.trigger_hyde_generation.assert_not_called()
