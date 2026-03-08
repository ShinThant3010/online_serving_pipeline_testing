import json
from pathlib import Path
from typing import Any

from api.schema import FeedsMetadata, FeedsRecommendation
from modules.functions.hyde_embedding import HydeEmbeddingStore
from modules.services.calc_subscore import RerankItem, calc_subscore


# ---------------------------------------------------------------------------------------------
# Compute reranked feed candidates by combining vector-search scores with subscore logic
# ---------------------------------------------------------------------------------------------
def rerank_with_subscore(
    *,
    student_id: str,
    score_matrix: list[list[float]] | None = None,
    feed_matrix: list[list[str]],
    embedding_store: HydeEmbeddingStore,
) -> list[RerankItem]:
    """
    Compute reranked feed candidates by combining vector-search scores with subscore logic.

    If HyDE queries or score inputs are missing, falls back to metadata/feed-based scoring only.
    Returns a list of rerank items with final scores.
    """
    
    ### ----------------- prep hyde & metadata for calc subscore ----------------- ###
    hyde_query = embedding_store.load_hyde_queries(student_id)
    metadata = embedding_store.load_metadata(student_id)

    ### ------------------------- calc subscore w/o hyde ------------------------- ###
    if not hyde_query or not score_matrix or not feed_matrix:
        # _save_calc_subscore_params(student_id=student_id, 
        # params={"student_id": student_id, "feed": feed_matrix or [],"metadata": metadata})
        return calc_subscore(
            student_id=student_id,
            feed=feed_matrix or [],
            metadata=metadata,
        )

    ### ------------------------- calc subscore w/ hyde ------------------------- ###
    # _save_calc_subscore_params(student_id=student_id, 
    # params={"student_id": student_id, "score": score_matrix, "feed": feed_matrix, "hyde_query": hyde_query, "metadata": metadata})
    return calc_subscore(
        student_id=student_id,
        score=score_matrix,
        feed=feed_matrix,
        hyde_query=hyde_query,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------------------------
# Convert reranked items to feed recommendations
# ---------------------------------------------------------------------------------------------
def format_recommendations(
    reranked: list[RerankItem],
    *,
    metadata_by_feed_id: dict[str, FeedsMetadata | dict[str, Any] | None] | None = None,
) -> list[FeedsRecommendation]:
    """Convert reranked items to feed recommendations"""
    if not reranked:
        return []

    resolved_metadata_by_feed_id = metadata_by_feed_id
    if resolved_metadata_by_feed_id is None:
        resolved_metadata_by_feed_id = {}

    recommendations: list[FeedsRecommendation] = []
    for item in reranked:
        if not isinstance(item, dict):
            continue
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


# ---------------------------------------------------------------------------------------------
# Test function to save example input data for calc-subscore function
# ---------------------------------------------------------------------------------------------
def _save_calc_subscore_params(*, student_id: str, params: dict[str, Any]) -> None:
    """Persist calc_subscore input params to a per-student local debug file."""
    output_dir = Path("parameters_to_calcSubScore")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_student_id = student_id.replace("/", "_")
    output_path = output_dir / f"{safe_student_id}.txt"
    output_path.write_text(
        json.dumps(params, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
