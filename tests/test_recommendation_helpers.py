import asyncio
from unittest.mock import MagicMock

from api.schema import FeedsMetadata
from modules.services.recommend_with_subscore import format_recommendations
from modules.services.vector_recommendation import rerank_neighbors, search_neighbors_async


def test_format_recommendations_sorts_scores_and_parses_metadata():
    reranked = [
        {"feed_id": "f-2", "final_score": 0.123456789},
        {"feed_id": "f-1", "final_score": 0.9},
        {"feed_id": "", "final_score": 0.7},
        {"feed_id": "f-3", "final_score": None},
    ]
    metadata_map = {
        "f-1": {"title": "First"},
        "f-2": FeedsMetadata(topic="math"),
    }

    recommendations = format_recommendations(reranked, metadata_by_feed_id=metadata_map)

    assert [item.feed_id for item in recommendations] == ["f-1", "f-2"]
    assert recommendations[0].score == 0.9
    assert recommendations[1].score == 0.123457
    assert isinstance(recommendations[0].metadata, FeedsMetadata)
    assert recommendations[0].metadata.model_dump()["title"] == "First"


def test_rerank_neighbors_builds_matrices_and_calls_subscore(monkeypatch):
    captured = {}

    def fake_rerank_with_subscore(*, student_id, score_matrix, feed_matrix, embedding_store):
        captured["student_id"] = student_id
        captured["score_matrix"] = score_matrix
        captured["feed_matrix"] = feed_matrix
        captured["embedding_store"] = embedding_store
        return [{"feed_id": "f-1", "final_score": 0.7}]

    monkeypatch.setattr("modules.services.vector_recommendation.rerank_with_subscore", fake_rerank_with_subscore)

    store = MagicMock()
    search_results = [
        [{"feed_id": "f-1", "score": 0.1}, {"feed_id": "f-2", "score": 0.2}],
        [{"feed_id": "f-3", "score": 0.3}, {"feed_id": None, "score": 0.4}],
        "bad-row",
    ]

    reranked = rerank_neighbors("student-1", search_results, embedding_store=store)

    assert reranked == [{"feed_id": "f-1", "final_score": 0.7}]
    assert captured["student_id"] == "student-1"
    assert captured["score_matrix"] == [[0.1, 0.2], [0.3]]
    assert captured["feed_matrix"] == [["f-1", "f-2"], ["f-3"]]
    assert captured["embedding_store"] is store


def test_search_neighbors_async_falls_back_to_sync_when_runtime_error(monkeypatch):
    vector_search = MagicMock()
    vector_search.search.side_effect = [
        [{"feed_id": "f-1", "score": 0.1}],
        [{"feed_id": "f-2", "score": 0.2}],
    ]

    def raise_runtime_error(coro):
        coro.close()
        raise RuntimeError("event loop unavailable")

    monkeypatch.setattr(asyncio, "run", raise_runtime_error)

    results = search_neighbors_async([[0.1], [], [0.2]], vector_search=vector_search)

    assert results == [
        [{"feed_id": "f-1", "score": 0.1}],
        [{"feed_id": "f-2", "score": 0.2}],
    ]
    assert vector_search.search.call_count == 2
