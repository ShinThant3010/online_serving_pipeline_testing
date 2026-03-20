import json
import random
import uuid

from fastapi import Request


### ----------------------- get "X-Request-Id"/"X-Correlation-Id" ----------------------- ###
def request_id(request: Request) -> str:
    for header in ("X-Request-Id", "X-Correlation-Id"):
        value = (request.headers.get(header) or "").strip()
        if value:
            return value
    return uuid.uuid4().hex

### ---------------------------- get "X-Cloud-Trace-Context" ---------------------------- ###
def extract_trace_id(request: Request) -> str | None:
    trace_context = request.headers.get("X-Cloud-Trace-Context", "").strip()
    if not trace_context:
        return None
    return trace_context.split("/", 1)[0] or None

### ------------------------------- control log/ not log ------------------------------- ###
def should_log_request(sample_rate: float) -> bool:
    sample_rate = min(1.0, max(0.0, sample_rate))
    return random.random() <= sample_rate


# ---------------------------------------------------------------------------------------------
# Main function: logging 
# ---------------------------------------------------------------------------------------------
def emit_recommendation_timing_log(
    request: Request,
    sample_rate: float,
    student_id: str,
    source: str,
    req_id: str,
    trace_id: str | None,
    cache_hit: bool,
    t_total: float,
    t_cache_get: float,
    t_vector_search: float,
    t_postprocess: float,
    t_fallback_prepare: float,
    t_rerank: float,
    t_metadata_fetch: float,
    t_format_response: float,
    t_top_up_merge: float,
    num_recommendations: list[int],
    t_response_write: float,
) -> None:
    if not should_log_request(sample_rate):
        return
    print(
        json.dumps(
            {
                "event": "recommendation_timing",
                "student_id": student_id,
                "source": source,
                "request_id": req_id,
                "trace_id": trace_id,
                "x_cloud_trace_context": request.headers.get("X-Cloud-Trace-Context"),
                "cache_hit": cache_hit,
                "t_total": round(t_total, 6),
                "t_cache_get": round(t_cache_get, 6),
                "t_vector_search": round(t_vector_search, 6),
                "t_postprocess": round(t_postprocess, 6),
                "t_fallback_prepare": round(t_fallback_prepare, 6),
                "t_rerank": round(t_rerank, 6),
                "t_metadata_fetch": round(t_metadata_fetch, 6),
                "t_format_response": round(t_format_response, 6),
                "t_top_up_merge": round(t_top_up_merge, 6),
                "num_recommendations": num_recommendations,
                "t_response_write": round(t_response_write, 6),
            },
            ensure_ascii=True,
        )
    )
