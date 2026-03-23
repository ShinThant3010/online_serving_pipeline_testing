from pydantic import BaseModel, ConfigDict, Field


def to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

# ---------------------------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------------------------
class RecommendationRequest(APIModel):
    student_id: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------------------------
class FeedsMetadata(BaseModel):
    # Feed metadata shape may evolve; allow passthrough fields from Redis payloads.
    model_config = ConfigDict(extra="allow")


class FeedsRecommendation(APIModel):
    feed_id: str
    score: float
    metadata: FeedsMetadata | None = None


class RecommendationResponse(APIModel):
    student_id: str
    source: str
    num_recommendations: int
    recommendations: list[FeedsRecommendation]
