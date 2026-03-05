import asyncio
from typing import Any

from modules.functions.hyde_embedding import HydeEmbeddingStore
from modules.services.calc_subscore import RerankItem
from modules.services.recommend_with_subscore import rerank_with_subscore
from modules.functions.vector_search import VectorSearchClient


def search_neighbors_async(
    embeddings: list[list[float]],
    *,
    vector_search: VectorSearchClient,
) -> list[list[dict[str, Any]]]:
    """Search neighbors asynchronously, falling back to sync when needed."""

    async def _run() -> list[list[dict[str, Any]]]:
        tasks = [
            asyncio.to_thread(vector_search.search, [embedding])
            for embedding in embeddings
            if embedding
        ]
        if not tasks:
            return []
        return await asyncio.gather(*tasks)

    try:
        # print("Attempting asynchronous vector search...")
        return asyncio.run(_run())

    except RuntimeError:
        # print("Async vector search failed, falling back to synchronous search.")
        return [vector_search.search([embedding]) for embedding in embeddings if embedding]


def rerank_neighbors(
    student_id: str,
    search_results: list[list[dict[str, Any]]],
    *,
    embedding_store: HydeEmbeddingStore,
) -> list[RerankItem]:
    """Rerank search results with deterministic sub-scoring."""
    score_matrix: list[list[float]] = []
    feed_matrix: list[list[str]] = []

    for neighbors in search_results:
        if not isinstance(neighbors, list):
            continue
        row_scores: list[float] = []
        row_feeds: list[str] = []
        for item in neighbors:
            if not isinstance(item, dict):
                continue
            feed_id = item.get("feed_id")
            score = item.get("score")
            if not feed_id or score is None:
                continue
            row_feeds.append(str(feed_id))
            row_scores.append(float(score))

        if not row_feeds:
            continue
        score_matrix.append(row_scores)
        feed_matrix.append(row_feeds)

    return rerank_with_subscore(
        student_id=student_id,
        score_matrix=score_matrix,
        feed_matrix=feed_matrix,
        embedding_store=embedding_store,
    )
