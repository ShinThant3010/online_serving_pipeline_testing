import json
import os
import random
import time
import uuid
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request

from api.schema import RecommendationRequest, RecommendationResponse
from modules.core.recommend_feeds import RecommendationService
from modules.utils.load_config import load_settings

app = FastAPI(title="Feeds Recommendation API", version="0.1.0")


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    settings = load_settings()
    return RecommendationService(settings=settings)


@app.middleware("http")
async def add_response_time_header(request, call_next):
    """Middleware to add response time header to each response."""
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-response-time-seconds"] = f"{time.perf_counter() - started:.6f}"
    return response


def _extract_trace_id(request: Request) -> str | None:
    trace_context = request.headers.get("X-Cloud-Trace-Context", "").strip()
    if not trace_context:
        return None
    return trace_context.split("/", 1)[0] or None


def _instance_id() -> str:
    return (os.getenv("HOSTNAME") or "").strip()


def _request_id(request: Request) -> str:
    for header in ("X-Request-Id", "X-Correlation-Id"):
        value = (request.headers.get(header) or "").strip()
        if value:
            return value
    return uuid.uuid4().hex


def _locust_user_id(request: Request) -> str | None:
    value = (request.headers.get("X-Locust-User-Id") or "").strip()
    return value or None


def _should_log_request() -> bool:
    raw = (os.getenv("PERF_LOG_SAMPLE_RATE", "1.0") or "1.0").strip()
    try:
        sample_rate = float(raw)
    except ValueError:
        sample_rate = 1.0
    sample_rate = min(1.0, max(0.0, sample_rate))
    return random.random() <= sample_rate


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint to verify that the API is running."""
    return {"status": "ok"}


@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(
    request: Request,
    payload: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    """Endpoint to get feed recommendations for a student."""
    started = time.perf_counter()
    req_id = _request_id(request)
    locust_user_id = _locust_user_id(request)
    trace_id = _extract_trace_id(request)
    instance_id = _instance_id()
    try:
        # print(f"Received recommendation request for student_id: {payload.student_id}")
        response, diagnostics = service.recommend(
            payload.student_id,
            index_endpoint=payload.vertex.index_endpoint if payload.vertex else None,
            deployed_index_id=payload.vertex.deployed_index_id if payload.vertex else None,
        )

        write_started = time.perf_counter()
        # Force response serialization timing for observability.
        response.model_dump_json()
        t_response_write = time.perf_counter() - write_started
        t_total = time.perf_counter() - started

        if _should_log_request():
            print(
                json.dumps(
                    {
                        "event": "recommendation_timing",
                        "student_id": payload.student_id,
                        "source": response.source,
                        "request_id": req_id,
                        "locust_user_id": locust_user_id,
                        "trace_id": trace_id,
                        "x_cloud_trace_context": request.headers.get("X-Cloud-Trace-Context"),
                        "instance_id": instance_id,
                        "cache_hit": diagnostics.cache_hit,
                        "t_total": round(t_total, 6),
                        "t_cache_get": round(diagnostics.t_cache_get, 6),
                        "t_vector_search": round(diagnostics.t_vector_search, 6),
                        "t_postprocess": round(diagnostics.t_postprocess, 6),
                        "t_response_write": round(t_response_write, 6),
                    },
                    ensure_ascii=True,
                )
            )
        return response
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
