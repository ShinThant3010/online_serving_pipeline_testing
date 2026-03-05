import time
from dataclasses import dataclass

from google.cloud import bigquery

from api.schema import FeedsMetadata, FeedsRecommendation, RecommendationResponse

from modules.functions.bigquery_fallback import fetch_fallback_recommendations
from modules.functions.hyde_embedding import HydeEmbeddingStore
from modules.functions.trigger_hyde_generation import TriggerHydeGenerationService
from modules.functions.vector_search import VectorSearchClient
from modules.services.recommend_with_subscore import (
    format_recommendations,
    rerank_with_subscore,
)
from modules.services.vector_recommendation import (
    rerank_neighbors,
    search_neighbors_async,
)
from modules.utils.load_config import Settings
from modules.utils.redis import RedisCache


# ---------------------------------------------------------------------------------------------
# Logging and diagnostics dataclass
# ---------------------------------------------------------------------------------------------
@dataclass
class RecommendationDiagnostics:
    cache_hit: bool         # Whether the recommendation was served from cache
    t_total: float          # Total time taken to process the recommendation request
    t_cache_get: float      # Time taken to attempt cache retrieval
    t_vector_search: float  # Time taken to perform vector search (if cache miss)
    t_postprocess: float    # Time taken for post-processing steps like reranking and formatting the response


# ---------------------------------------------------------------------------------------------
# Recommendation Service
# ---------------------------------------------------------------------------------------------
class RecommendationService:
    def __init__(self, settings: Settings) -> None:
        """Initialize the recommendation service with necessary clients and configurations."""
        self.settings = settings

        self.redis_cache = RedisCache(
            host=self.settings.cache.redis_host,
            port=self.settings.cache.redis_port,
            timeout_seconds=self.settings.cache.redis_timeout_seconds,
        )

        self.embedding_store = HydeEmbeddingStore(
            bucket=self.settings.hyde_data.bucket,
            embedding_prefix=self.settings.hyde_data.embedding_prefix,
            query_prefix=self.settings.hyde_data.query_prefix,
            metadata_prefix=self.settings.hyde_data.metadata_prefix,
        )

        self.vector_search = VectorSearchClient(
            index_endpoint=self.settings.vertex.index_endpoint,
            deployed_index_id=self.settings.vertex.deployed_index_id,
            neighbor_count=self.settings.vertex.neighbor_count,
            return_full_datapoint=self.settings.vertex.return_full_datapoint,
            restricts_list=self.settings.vertex.restricts_list,
        )

        self.bigquery_client = bigquery.Client()
        self.trigger_hyde_generation_service = TriggerHydeGenerationService(
            config=self.settings.trigger_hyde_generation
        )

    @staticmethod
    def _key(student_id: str) -> str:
        """Generate a Redis cache key for a given student ID."""
        return f"recommendations:{student_id}"


# ---------------------------------------------------------------------------------------------
# Main recommendation method with cache retrieval, vector search, and fallback logic
# ---------------------------------------------------------------------------------------------
    def recommend(
        self,
        student_id: str,
    ) -> tuple[RecommendationResponse, RecommendationDiagnostics]:
        """Get feed recommendations from cache, vector search, or fallback."""
        started = time.perf_counter()
        t_cache_get = 0.0
        t_vector_search = 0.0
        t_postprocess = 0.0
        cache_hit = False

        cache_key = self._key(student_id)
        cache_started = time.perf_counter()
        ### --------------------- Attempt to retrieve cached response --------------------- ###
        cached_response = self._get_cached_response(cache_key)
        t_cache_get = time.perf_counter() - cache_started
        
        if cached_response:
            # print(f"Cache hit for {student_id}, returning cached recommendations.")
            cache_hit = True
            diagnostics = RecommendationDiagnostics(
                cache_hit=cache_hit,
                t_total=time.perf_counter() - started,
                t_cache_get=t_cache_get,
                t_vector_search=t_vector_search,
                t_postprocess=t_postprocess,
            )
            return cached_response, diagnostics
        ### --------------------------- return cached response --------------------------- ###

        # print(f"No cache found for {student_id}, retrieving embedding for vector search...")
        embeddings = self.embedding_store.load_embeddings(student_id)

        if not embeddings:
            # print(f"No embeddings found for {student_id}, activating fallback, trigger hyde generation...")
            postprocess_started = time.perf_counter()
            response = self._build_fallback_response(student_id=student_id, trigger_refresh=True)
            t_postprocess = time.perf_counter() - postprocess_started
            diagnostics = RecommendationDiagnostics(
                cache_hit=cache_hit,
                t_total=time.perf_counter() - started,
                t_cache_get=t_cache_get,
                t_vector_search=t_vector_search,
                t_postprocess=t_postprocess,
            )
            return response, diagnostics
        ### --------------------- return no embedding fallback response --------------------- ###

        try:
            # print(f"Embeddings retrieved for {student_id}, proceeding with vector search...")
            response, t_vector_search, t_postprocess = self._build_vector_response(
                student_id=student_id,
                embeddings=embeddings,
            )
            minimum_recommendation = self.settings.recommendation.minimum_recommendation
            if len(response.recommendations) < minimum_recommendation:
                postprocess_started = time.perf_counter()
                fallback_response = self._build_fallback_response(
                    student_id=student_id,
                    trigger_refresh=False,
                )
                existing_feed_ids = {rec.feed_id for rec in response.recommendations}
                topped_up_recommendations = list(response.recommendations)

                for fallback_recommendation in fallback_response.recommendations:
                    if fallback_recommendation.feed_id in existing_feed_ids:
                        continue
                    topped_up_recommendations.append(fallback_recommendation)
                    existing_feed_ids.add(fallback_recommendation.feed_id)
                    if len(topped_up_recommendations) >= minimum_recommendation:
                        break

                response = RecommendationResponse(
                    student_id=student_id,
                    source=f"{response.source}+{fallback_response.source}",
                    recommendations=topped_up_recommendations,
                )
                t_postprocess += time.perf_counter() - postprocess_started
            ### ----- return vector search, but less than minimum recommendations response ------ ###

            self.redis_cache.set_one(
                cache_key,
                response.model_dump(),
                ttl_seconds=self.settings.cache.ttl_seconds,
            )
            diagnostics = RecommendationDiagnostics(
                cache_hit=cache_hit,
                t_total=time.perf_counter() - started,
                t_cache_get=t_cache_get,
                t_vector_search=t_vector_search,
                t_postprocess=t_postprocess,
            )
            return response, diagnostics
            ### ------------------------- return vector search response ------------------------- ###

        except Exception as exc: 
            # print(f"vector search fallback activated for {student_id}: {exc}")
            postprocess_started = time.perf_counter()
            response = self._build_fallback_response(student_id=student_id, trigger_refresh=False)
            t_postprocess = time.perf_counter() - postprocess_started
            diagnostics = RecommendationDiagnostics(
                cache_hit=cache_hit,
                t_total=time.perf_counter() - started,
                t_cache_get=t_cache_get,
                t_vector_search=t_vector_search,
                t_postprocess=t_postprocess,
            )
            return response, diagnostics
            ### ------------------ return vector search fail; fallback response ------------------ ###


# ---------------------------------------------------------------------------------------------
# Helper methods for cache retrieval
# ---------------------------------------------------------------------------------------------
    def _get_cached_response(self, cache_key: str) -> RecommendationResponse | None:
        """Attempt to retrieve a cached recommendation response from Redis."""
        cached = self.redis_cache.get_one(cache_key)
        if not cached:
            return None
        payload = {**cached, "source": "redis_cache"}
        return RecommendationResponse(**payload)


# ---------------------------------------------------------------------------------------------
# Helper methods for vector search response building
# ---------------------------------------------------------------------------------------------
    def _build_vector_response(
        self,
        *,
        student_id: str,
        embeddings: list[list[float]],
    ) -> tuple[RecommendationResponse, float, float]:
        """Build a recommendation response using vector search results."""
        search_started = time.perf_counter()
        search_results = search_neighbors_async(embeddings, vector_search=self.vector_search)
        t_vector_search = time.perf_counter() - search_started

        postprocess_started = time.perf_counter()
        reranked = rerank_neighbors(
            student_id,
            search_results,
            embedding_store=self.embedding_store,
        )

        recommendations = format_recommendations(reranked, redis_cache=self.redis_cache)
        t_postprocess = time.perf_counter() - postprocess_started
        response = RecommendationResponse(
            student_id=student_id,
            source="vertex_vector_search",
            recommendations=recommendations,
        )
        return response, t_vector_search, t_postprocess


# ---------------------------------------------------------------------------------------------
# Helper methods for fallback response building
# ---------------------------------------------------------------------------------------------
    def _build_fallback_response(
        self,
        *,
        student_id: str,
        trigger_refresh: bool,
    ) -> RecommendationResponse:
        """Build a recommendation response using fallback data, optionally triggering a refresh of the HyDE generation."""
        if trigger_refresh:
            self.trigger_hyde_generation_service.trigger_hyde_generation(student_id=student_id)

        fallback_limit = self.settings.bigquery.fallback_limit
        fallback_source = "bigquery_fallback"
        feed_cache_keys = self.redis_cache.get_many_by_prefix("feeds")

        if len(feed_cache_keys) >= fallback_limit:
            selected_keys = feed_cache_keys[:fallback_limit]
            cached_payloads = self.redis_cache.get_many(selected_keys)
            fallback_items: list[tuple[str, FeedsMetadata | None]] = []
            for key in selected_keys:
                feed_id = key.split(":", 1)[1] if ":" in key else key
                payload = cached_payloads.get(key)
                metadata = FeedsMetadata(**payload) if isinstance(payload, dict) else None
                fallback_items.append((feed_id, metadata))
            fallback_source = "redis_fallback"
            # print(f"Using Redis fallback with {len(fallback_items)} cached feeds.")
        else:
            # print(f"Redis fallback cache is insufficient ({len(feed_cache_keys)}/{fallback_limit}); using BigQuery fallback.")
            fallback_items = fetch_fallback_recommendations(
                bigquery_client=self.bigquery_client,
                fallback_table=self.settings.bigquery.fallback_table,
                fallback_limit=fallback_limit,
            )

        metadata_by_feed_id = {feed_id: metadata for feed_id, metadata in fallback_items}
        feed_ids = [feed_id for feed_id, _ in fallback_items]

        reranked = rerank_with_subscore(
            student_id=student_id,
            feed_matrix=[feed_ids] if feed_ids else [],
            embedding_store=self.embedding_store,
        )

        if reranked:
            recommendations = format_recommendations(
                reranked,
                metadata_by_feed_id=metadata_by_feed_id,
            )
        else:
            recommendations = [
                FeedsRecommendation(
                    feed_id=feed_id,
                    score=0.0,
                    metadata=metadata,
                )
                for feed_id, metadata in fallback_items
            ]

        return RecommendationResponse(
            student_id=student_id,
            source=fallback_source,
            recommendations=recommendations,
        )
