import json
from pathlib import Path
from typing import Any

from api.schema import FeedsMetadata, FeedsRecommendation
from modules.functions.hyde_embedding import HydeEmbeddingStore
from modules.services.calc_subscore import RerankItem, calc_subscore
from modules.utils.redis import RedisCache


def rerank_with_subscore(
    *,
    student_id: str,
    score_matrix: list[list[float]] | None = None,
    feed_matrix: list[list[str]],
    embedding_store: HydeEmbeddingStore,
) -> list[RerankItem]:
    hyde_query, metadata = prepare_subscore_context(
        student_id=student_id,
        embedding_store=embedding_store,
    )
    if not hyde_query or not score_matrix or not feed_matrix:
        print(f"No hyde_query nor vector search score found for {student_id}, reranking with metadata only.")
        # _save_calc_subscore_params(
        #     student_id=student_id,
        #     params={
        #         "student_id": student_id,
        #         "feed": feed_matrix or [],
        #         "metadata": metadata,
        #     },
        # )
        return calc_subscore(
            student_id=student_id,
            feed=feed_matrix or [],
            metadata=metadata,
        )
    # _save_calc_subscore_params(
    #     student_id=student_id,
    #     params={
    #         "student_id": student_id,
    #         "score": score_matrix,
    #         "feed": feed_matrix,
    #         "hyde_query": hyde_query,
    #         "metadata": metadata,
    #     },
    # )
    return calc_subscore(
        student_id=student_id,
        score=score_matrix,
        feed=feed_matrix,
        hyde_query=hyde_query,
        metadata=metadata,
    )


def format_recommendations(
    reranked: list[RerankItem],
    *,
    redis_cache: RedisCache | None = None,
    metadata_by_feed_id: dict[str, FeedsMetadata | dict[str, Any] | None] | None = None,
) -> list[FeedsRecommendation]:
    """Convert reranked items to feed recommendations, resolving metadata from cache if needed."""
    if not reranked:
        return []

    resolved_metadata_by_feed_id = metadata_by_feed_id
    if resolved_metadata_by_feed_id is None and redis_cache is not None:
        feed_ids = [str(item["feed_id"]) for item in reranked if item.get("feed_id")]
        resolved_metadata_by_feed_id = redis_cache.get_many(feed_ids) if feed_ids else {}
    elif resolved_metadata_by_feed_id is None:
        resolved_metadata_by_feed_id = {}

    recommendations: list[FeedsRecommendation] = []
    for item in reranked:
        feed_id = item.get("feed_id")
        score_value = item.get("final_score")
        if not feed_id or score_value is None:
            continue

        key = str(feed_id)
        metadata_payload = resolved_metadata_by_feed_id.get(key)
        metadata = (
            metadata_payload
            if isinstance(metadata_payload, FeedsMetadata)
            else FeedsMetadata(**metadata_payload)
            if isinstance(metadata_payload, dict)
            else None
        )

        recommendations.append(
            FeedsRecommendation(
                feed_id=key,
                score=round(float(score_value), 6),
                metadata=metadata,
            )
        )

    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations


def prepare_subscore_context(
    *,
    student_id: str,
    embedding_store: HydeEmbeddingStore,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hyde_query_payload = embedding_store.load_hyde_queries(student_id)
    hyde_query = normalize_hyde_query_for_calc_subscore(hyde_query_payload)
    metadata_payload = embedding_store.load_metadata(student_id)
    return hyde_query, metadata_payload


def normalize_hyde_query_for_calc_subscore(
    hyde_query_payload: Any,
) -> list[dict[str, Any]]:
    if isinstance(hyde_query_payload, dict):
        hq = hyde_query_payload.get("hq")
        if isinstance(hq, list):
            return [q for q in hq if isinstance(q, dict)]

    if isinstance(hyde_query_payload, list) and hyde_query_payload:
        parsed = [q for q in hyde_query_payload if isinstance(q, dict)]
        if parsed:
            return parsed

    return []


def _save_calc_subscore_params(*, student_id: str, params: dict[str, Any]) -> None:
    output_dir = Path("parameters_to_calcSubScore")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_student_id = student_id.replace("/", "_")
    output_path = output_dir / f"{safe_student_id}.txt"
    output_path.write_text(
        json.dumps(params, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
