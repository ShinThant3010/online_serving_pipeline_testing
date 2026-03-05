from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------------------------
class RecommendationRequest(BaseModel):
    student_id: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------------------------
class FeedsMetadata(BaseModel):
    # Feed metadata shape may evolve; allow passthrough fields from Redis payloads.
    model_config = ConfigDict(extra="allow")


class FeedsRecommendation(BaseModel):
    feed_id: str
    score: float
    metadata: FeedsMetadata | None = None


class RecommendationResponse(BaseModel):
    student_id: str
    source: str
    recommendations: list[FeedsRecommendation]
