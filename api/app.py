import time

from fastapi import Depends, FastAPI, HTTPException

from api.schema import RecommendationRequest, RecommendationResponse
from modules.core.recommend_feeds import RecommendationService
from modules.utils.load_config import load_settings

app = FastAPI(title="Feeds Recommendation API", version="0.1.0")


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


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint to verify that the API is running."""
    return {"status": "ok"}


@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(
    payload: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    """Endpoint to get feed recommendations for a student."""
    try:
        print(f"Received recommendation request for student_id: {payload.student_id}")
        return service.recommend(
            payload.student_id,
            index_endpoint=payload.vertex.index_endpoint if payload.vertex else None,
            deployed_index_id=payload.vertex.deployed_index_id if payload.vertex else None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
