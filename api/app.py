import time
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request

from api.schema import RecommendationRequest, RecommendationResponse
from modules.core.recommend_feeds import RecommendationService
from modules.utils.load_config import load_settings
from modules.utils.log_request import (
    emit_recommendation_timing_log,
    extract_trace_id,
    request_id,
)

app = FastAPI(title="Feeds Recommendation API", version="0.1.0")


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
    response = await call_next(request)
    response.headers["x-response-time-seconds"] = f"{time.perf_counter() - started:.6f}"
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
@app.post("/recommendations", response_model=RecommendationResponse)
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
            t_response_write=t_response_write,
        )
        return response
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
