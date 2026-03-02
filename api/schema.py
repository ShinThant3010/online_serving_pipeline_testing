from pydantic import BaseModel, ConfigDict, Field


class VertexRequest(BaseModel):
    index_endpoint: str | None = None
    deployed_index_id: str | None = None


class RecommendationRequest(BaseModel):
    student_id: str = Field(..., min_length=1)
    vertex: VertexRequest | None = None


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
