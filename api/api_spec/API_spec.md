# Feeds Recommendation API - Core Engine (REST API Spec)

Service: **online_serving_pipeline**

**1. Purpose**
This API returns ranked feed recommendations for a student using a cache-first online serving flow with Vertex AI vector retrieval and fallback recommendations.

**2. High-Level Flow**
1. Redis lookup for `recommendations:{student_id}`
2. HyDE embedding lookup from GCS on cache miss
3. Vertex AI Matching Engine retrieval and reranking when embeddings exist
4. Fallback recommendation build from Redis feed cache or BigQuery when embeddings are missing or vector search fails
5. Optional background HyDE-generation trigger when the student has no stored embeddings

---

## Base URLs

**Production / Cloud Run:**  
`https://online-serving-feeds-<project>-southeast1.run.app`

**Local (Uvicorn / FastAPI):**  
`http://127.0.0.1:8080`

**Swagger / OpenAPI:**  
`https://online-serving-feeds-api-810737581373.asia-southeast1.run.app/docs`

NOTE: MemoryStore (Redis) will not be able to connect for local run since it is deployed on CloudRun. 

---

## Guideline Alignment Notes

* ✅ **HTTP methods:** `GET` for health, `POST` for synchronous recommendation lookup
* ✅ **JSON request body:** `application/json`
* ✅ **Validation:** FastAPI / Pydantic validation for request payloads
* ✅ **Observed success headers:** `x-response-time-seconds`, `X-API-Version`, `X-Correlation-Id`
* ✅ **JSON naming:** requests accept camelCase and responses return camelCase for declared API fields

---

## Authentication & Authorization

No authentication or authorization is implemented at the application layer in the current codebase.

---

## Required Headers

* `Content-Type: application/json`
* `X-API-Version: 0.1.0` (current FastAPI application version)
* `X-Correlation-Id: <string>` (If a caller sends `X-Correlation-Id`, echoes; else generates `corr_<uuid>`)
* `x-response-time-seconds: <float>` (server-side processing time in seconds)

---

## Endpoints Summary

### Health
* `GET /health`

### Feed Recommendations
* `POST /recommendations`

---

## 1) Health Endpoint

### GET /health

Liveness check for the API service.

**Response:** `200 OK`

```json
{
  "status": "ok"
}
```

---

## 2) Get Feed Recommendations

### POST /recommendations

Returns ranked feed recommendations for a single student.

### Status Codes

* `200 OK` - recommendations returned successfully, including cache-served or fallback-served responses
* `422 Unprocessable Entity` - invalid request payload, such as missing or empty `student_id`
* `500 Internal Server Error` - unhandled runtime failure surfaced by the API layer

### Request Schema

#### Example

```json
{
  "studentId": "student_123"
}
```

#### Field Definitions

| Field | Type | Required | Notes | Examples
| ----- | ---- | -------: | ----- | -----
| studentId | string | ✅ | Non-empty student identifier | stu_p208, stu_p303, stu_p370, stu_p388, stu_p390, stu_p401, stu_p477, stu_p484, stu_p497, stu_p545, stu_p556, stu_p613, stu_p627, stu_p686, stu_p699

### NOTE
* The recommended feeds will be stored in cache for 1hr. If you want to reset feeds recommendation in cache, use following endpoint.
  * API: `https://hyde-cache-pipeline-api-810737581373.asia-southeast1.run.app/docs` 
  * endpoint: `DELETE /cache/delete-prefix/{cache_prefix}` with "recommendations" as prefix

### Successful Response

**200 OK**

Headers:
* `Content-Type: application/json`
* `x-response-time-seconds: <float>`
* `X-API-Version: 0.1.0`
* `X-Correlation-Id: <string>`

Body:

```json
{
  "studentId": "student_123",
  "source": "vertex_vector_search",
  "recommendations": [
    {
      "feedId": "TH_FEED_001",
      "score": 0.923451,
      "metadata": {
        "title": "Applied Algebra Basics",
        "language": "th"
      }
    },
    {
      "feedId": "TH_FEED_002",
      "score": 0.901234,
      "metadata": {
        "title": "Fractions Drill Set",
        "language": "th"
      }
    }
  ]
}
```

### Response Field Definitions

| Field | Type | Required | Notes |
| ----- | ---- | -------: | ----- |
| studentId | string | ✅ | Echo of the request input |
| source | string | ✅ | Source of recommendation generation |
| recommendations | array | ✅ | Ranked recommendation list |

### Recommendation Item

| Field | Type | Required | Notes |
| ----- | ---- | -------: | ----- |
| feedId | string | ✅ | Feed identifier |
| score | float | ✅ | Ranking score after vector search and/or subscore reranking |
| metadata | object or null | ❌ | Feed metadata payload, passthrough-friendly and schema-flexible |

### `source` Values

The current implementation can return one of the following source values:

* `redis_cache`
* `vertex_vector_search`
* `redis_fallback`
* `bigquery_fallback`
* Combined values such as `vertex_vector_search+redis_fallback`
* Combined values such as `vertex_vector_search+bigquery_fallback`

### Serving Behavior Notes

1. If a cached response exists in Redis, the API returns it immediately and labels the source as `redis_cache`.
2. If no HyDE embeddings exist for the student, the API triggers a background HyDE-generation request and returns fallback recommendations.
3. If vector search succeeds but returns fewer than the configured minimum recommendation count, fallback items are appended.
4. If vector search raises an exception, the API returns fallback recommendations instead of surfacing the error in most cases.
5. Because of that fallback behavior, many dependency issues still result in `200 OK` responses with a fallback `source`.

---

## Error Responses

### 422 Unprocessable Entity

FastAPI validation errors use the default validation payload format.

Example:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "student_id"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

### 500 Internal Server Error

Unhandled runtime exceptions are wrapped as `HTTPException`.

Example:

```json
{
  "detail": "Runtime error message"
}
```

---

## Implementation Notes

### Recommendation Data Sources

* Redis recommendation cache
* GCS HyDE embedding store
* Vertex AI Matching Engine
* Redis feed metadata cache
* BigQuery fallback table

### Background Refresh Trigger

When embeddings are missing, the service may emit a fire-and-forget POST request to the configured HyDE generation endpoint:

`https://hyderecomment-service-du7yhkyaqq-as.a.run.app/hyde/students/{student_id}`

This call is cooldown-controlled per student and does not block the main API response.

---

## OpenAPI Reference

When the service is running locally:

* Swagger UI: `http://127.0.0.1:8080/docs`
* OpenAPI JSON: `http://127.0.0.1:8080/openapi.json`
