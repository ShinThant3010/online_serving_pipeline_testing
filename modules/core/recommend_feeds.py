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
    t_fallback_prepare: float
    t_rerank: float
    t_metadata_fetch: float
    t_format_response: float
    t_top_up_merge: float


@dataclass
class PostprocessTimings:
    t_fallback_prepare: float = 0.0
    t_rerank: float = 0.0
    t_metadata_fetch: float = 0.0
    t_format_response: float = 0.0
    t_top_up_merge: float = 0.0

    def total(self) -> float:
        return (
            self.t_fallback_prepare
            + self.t_rerank
            + self.t_metadata_fetch
            + self.t_format_response
            + self.t_top_up_merge
        )

    def merged_with(self, other: "PostprocessTimings") -> "PostprocessTimings":
        return PostprocessTimings(
            t_fallback_prepare=self.t_fallback_prepare + other.t_fallback_prepare,
            t_rerank=self.t_rerank + other.t_rerank,
            t_metadata_fetch=self.t_metadata_fetch + other.t_metadata_fetch,
            t_format_response=self.t_format_response + other.t_format_response,
            t_top_up_merge=self.t_top_up_merge + other.t_top_up_merge,
        )


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
        postprocess_timings = PostprocessTimings()
        t_vector_search = 0.0
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
                t_postprocess=postprocess_timings.total(),
                t_fallback_prepare=postprocess_timings.t_fallback_prepare,
                t_rerank=postprocess_timings.t_rerank,
                t_metadata_fetch=postprocess_timings.t_metadata_fetch,
                t_format_response=postprocess_timings.t_format_response,
                t_top_up_merge=postprocess_timings.t_top_up_merge,
            )
            return cached_response, diagnostics
        ### --------------------------- return cached response --------------------------- ###

        # print(f"No cache found for {student_id}, retrieving embedding for vector search...")
        embeddings = self.embedding_store.load_embeddings(student_id)

        if not embeddings:
            # print(f"No embeddings found for {student_id}, activating fallback, trigger hyde generation...")
            response, postprocess_timings = self._build_fallback_response(
                student_id=student_id,
                trigger_refresh=True,
            )
            diagnostics = RecommendationDiagnostics(
                cache_hit=cache_hit,
                t_total=time.perf_counter() - started,
                t_cache_get=t_cache_get,
                t_vector_search=t_vector_search,
                t_postprocess=postprocess_timings.total(),
                t_fallback_prepare=postprocess_timings.t_fallback_prepare,
                t_rerank=postprocess_timings.t_rerank,
                t_metadata_fetch=postprocess_timings.t_metadata_fetch,
                t_format_response=postprocess_timings.t_format_response,
                t_top_up_merge=postprocess_timings.t_top_up_merge,
            )
            return response, diagnostics
        ### --------------------- return no embedding fallback response --------------------- ###

        try:
            # print(f"Embeddings retrieved for {student_id}, proceeding with vector search...")
            response, t_vector_search, postprocess_timings = self._build_vector_response(
                student_id=student_id,
                embeddings=embeddings,
            )
            minimum_recommendation = self.settings.recommendation.minimum_recommendation
            if len(response.recommendations) < minimum_recommendation:
                fallback_response, fallback_timings = self._build_fallback_response(
                    student_id=student_id,
                    trigger_refresh=False,
                )
                postprocess_timings = postprocess_timings.merged_with(fallback_timings)
                top_up_started = time.perf_counter()
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
                postprocess_timings.t_top_up_merge += time.perf_counter() - top_up_started
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
                t_postprocess=postprocess_timings.total(),
                t_fallback_prepare=postprocess_timings.t_fallback_prepare,
                t_rerank=postprocess_timings.t_rerank,
                t_metadata_fetch=postprocess_timings.t_metadata_fetch,
                t_format_response=postprocess_timings.t_format_response,
                t_top_up_merge=postprocess_timings.t_top_up_merge,
            )
            return response, diagnostics
            ### ------------------------- return vector search response ------------------------- ###

        except Exception as exc: 
            print(f"vector search fallback activated for {student_id}: {exc}")
            response, postprocess_timings = self._build_fallback_response(
                student_id=student_id,
                trigger_refresh=False,
            )
            diagnostics = RecommendationDiagnostics(
                cache_hit=cache_hit,
                t_total=time.perf_counter() - started,
                t_cache_get=t_cache_get,
                t_vector_search=t_vector_search,
                t_postprocess=postprocess_timings.total(),
                t_fallback_prepare=postprocess_timings.t_fallback_prepare,
                t_rerank=postprocess_timings.t_rerank,
                t_metadata_fetch=postprocess_timings.t_metadata_fetch,
                t_format_response=postprocess_timings.t_format_response,
                t_top_up_merge=postprocess_timings.t_top_up_merge,
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
    ) -> tuple[RecommendationResponse, float, PostprocessTimings]:
        """Build a recommendation response using vector search results."""

        ### ------------------------------------ async vector search ------------------------------------ ###
        search_started = time.perf_counter()
        search_results = search_neighbors_async(embeddings, vector_search=self.vector_search)
        t_vector_search = time.perf_counter() - search_started

        ### ---- Adjust vector search result format to subscore format & call subscore calc function ----- ###
        postprocess_timings = PostprocessTimings()
        rerank_started = time.perf_counter()
        reranked = rerank_neighbors(
            student_id,
            search_results,
            embedding_store=self.embedding_store,
        )
        postprocess_timings.t_rerank = time.perf_counter() - rerank_started

        ### -------------------------------------- prep metadata --------------------------------------- ###
        metadata_started = time.perf_counter()
        feed_ids = [
            str(item["feed_id"])
            for item in reranked
            if isinstance(item, dict) and item.get("feed_id")
        ]
        if feed_ids:
            redis_keys = [f"feeds:{feed_id}" for feed_id in feed_ids]
            cached_by_redis_key = self.redis_cache.get_many(redis_keys)
            resolved_metadata_by_feed_id = {
                feed_id: cached_by_redis_key.get(f"feeds:{feed_id}") for feed_id in feed_ids
            }
        else:
            resolved_metadata_by_feed_id = {}
        postprocess_timings.t_metadata_fetch = time.perf_counter() - metadata_started


        ### -------------- Convert reranked items to feed recommendations (final response) -------------- ###
        format_started = time.perf_counter()
        recommendations = format_recommendations(
            reranked,
            metadata_by_feed_id=resolved_metadata_by_feed_id,
        )
        postprocess_timings.t_format_response = time.perf_counter() - format_started
        response = RecommendationResponse(
            student_id=student_id,
            source="vertex_vector_search",
            recommendations=recommendations,
        )
        return response, t_vector_search, postprocess_timings


# ---------------------------------------------------------------------------------------------
# Helper methods for fallback response building
# ---------------------------------------------------------------------------------------------
    def _build_fallback_response(
        self,
        *,
        student_id: str,
        trigger_refresh: bool,
    ) -> tuple[RecommendationResponse, PostprocessTimings]:
        """Build a recommendation response using fallback data, optionally triggering a refresh of the HyDE generation."""
        postprocess_timings = PostprocessTimings()
        fallback_prepare_started = time.perf_counter()

        ### --------------------- trigger hyDE api call for no embeddings fallback --------------------- ###
        if trigger_refresh:
            self.trigger_hyde_generation_service.trigger_hyde_generation(student_id=student_id)

        fallback_limit = self.settings.bigquery.fallback_limit
        fallback_source = "bigquery_fallback"
        feed_cache_keys = self.redis_cache.get_many_by_prefix("feeds")

        ### ----------------------- if there is metadata in cache, cache fallback ----------------------- ###
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

        ### ------------------------ if no metadata in cache, bigquery fallback ------------------------ ###
        else:
            # print(f"Redis fallback cache is insufficient ({len(feed_cache_keys)}/{fallback_limit}); using BigQuery fallback.")
            fallback_items = fetch_fallback_recommendations(
                bigquery_client=self.bigquery_client,
                fallback_table=self.settings.bigquery.fallback_table,
                fallback_limit=fallback_limit,
            )

        metadata_by_feed_id = {feed_id: metadata for feed_id, metadata in fallback_items}
        feed_ids = [feed_id for feed_id, _ in fallback_items]
        postprocess_timings.t_fallback_prepare = time.perf_counter() - fallback_prepare_started

        ### ------------------- prep for subscore calc & call subscore calc function ------------------- ###
        rerank_started = time.perf_counter()
        reranked = rerank_with_subscore(
            student_id=student_id,
            feed_matrix=[feed_ids] if feed_ids else [],
            embedding_store=self.embedding_store,
        )
        postprocess_timings.t_rerank = time.perf_counter() - rerank_started

        ### -------------- Convert reranked items to feed recommendations (final response) -------------- ###
        format_started = time.perf_counter()
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
        postprocess_timings.t_format_response = time.perf_counter() - format_started

        return (
            RecommendationResponse(
                student_id=student_id,
                source=fallback_source,
                recommendations=recommendations,
            ),
            postprocess_timings,
        )
