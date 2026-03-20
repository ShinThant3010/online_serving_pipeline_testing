import time
import uuid
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request

from api.schema import RecommendationRequest, RecommendationResponse
from modules.core.recommend_feeds import RecommendationService
from modules.utils.load_config import load_settings
from modules.utils.performance_logging import (
    emit_recommendation_timing_log,
    extract_trace_id,
    request_id,
)

app = FastAPI(title="Feeds Recommendation API", version="0.1.0")

API_VERSION_HEADER = "X-API-Version"
CORRELATION_ID_HEADER = "X-Correlation-Id"


# ---------------------------------------------------------------------------------------------
# Lazily initializes and caches the RecommendationService instance.
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    """Lazily initializes and caches the RecommendationService instance."""
    settings = load_settings()
    return RecommendationService(settings=settings)


# ---------------------------------------------------------------------------------------------
# Response time header middleware to measure and log the time taken to process each request.
# ---------------------------------------------------------------------------------------------
@app.middleware("http")
async def add_response_time_header(request, call_next):
    """Middleware to add response time header to each response."""
    started = time.perf_counter()
    request.state.correlation_id = request.headers.get(CORRELATION_ID_HEADER) or f"corr_{uuid.uuid4()}"
    response = await call_next(request)
    response.headers["x-response-time-seconds"] = f"{time.perf_counter() - started:.6f}"
    response.headers[API_VERSION_HEADER] = app.version
    response.headers[CORRELATION_ID_HEADER] = request.state.correlation_id
    return response


# ---------------------------------------------------------------------------------------------
# API endpoint for health check
# ---------------------------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint to verify that the API is running."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------------------------
# API endpoint for getting recommendations
# ---------------------------------------------------------------------------------------------
@app.post("/recommendations", response_model=RecommendationResponse, response_model_by_alias=True)
def recommend(
    request: Request,
    payload: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    """Endpoint to get feed recommendations for a student."""
    started = time.perf_counter()
    req_id = request_id(request)
    trace_id = extract_trace_id(request)
    try:
        ### --- Call the recommendation service to get recommendations and diagnostics information. --- ###
        response, diagnostics = service.recommend(payload.student_id)

        write_started = time.perf_counter()
        response.model_dump_json()
        t_response_write = time.perf_counter() - write_started
        t_total = time.perf_counter() - started

        ### ------------------------------------- Emit timing log ------------------------------------- ###
        emit_recommendation_timing_log(
            request=request,
            sample_rate=service.settings.app.perf_log_sample_rate,
            student_id=payload.student_id,
            source=response.source,
            req_id=req_id,
            trace_id=trace_id,
            cache_hit=diagnostics.cache_hit,
            t_total=t_total,
            t_cache_get=diagnostics.t_cache_get,
            t_vector_search=diagnostics.t_vector_search,
            t_postprocess=diagnostics.t_postprocess,
            t_fallback_prepare=diagnostics.t_fallback_prepare,
            t_rerank=diagnostics.t_rerank,
            t_metadata_fetch=diagnostics.t_metadata_fetch,
            t_format_response=diagnostics.t_format_response,
            t_top_up_merge=diagnostics.t_top_up_merge,
            num_recommendations=diagnostics.num_recommendations,
            t_response_write=t_response_write,
        )
        return response
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
