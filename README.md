# Online Serving Pipeline (Feeds Recommendation API)

This repository contains the online recommendation serving layer for feeds. It provides a FastAPI endpoint that retrieves, reranks, and returns feed recommendations for each student with cache-first and fallback behavior for reliability.

## 1. What this service does

This service exposes `POST /recommendations` to return ranked feed recommendations for a `student_id`.

The recommendation strategy combines:
- Redis cache lookup (`recommendations:{student_id}`)
- HyDE embeddings loaded from GCS
- Vertex AI Matching Engine vector retrieval
- Subscore reranking
- Fallback recommendations from Redis feed cache or BigQuery

## 2. High-level flow

1. Read `recommendations:{student_id}` from Redis.
2. If cache miss, load student embeddings from GCS.
3. If embeddings exist, call Vertex AI and rerank candidates.
4. If embeddings are missing, trigger HyDE generation (cooldown-controlled) and use fallback.
5. If vector flow fails, use fallback.
6. If final result count is below configured minimum, top up from fallback.
7. Cache successful response back to Redis.

## 3. Tech Stack

- Python 3.11
- FastAPI + Pydantic
- Redis (caching + feed metadata fallback source)
- Google Cloud Storage (HyDE embeddings/query/metadata)
- Vertex AI Matching Engine (vector retrieval)
- BigQuery (fallback feed source)
- pytest (unit tests)
- Locust (load tests)

## 4. Prerequisites & Access

Required locally:
- Python `3.11+`
- Access to the project Redis on Google Cloud Memorystore (private network)
- GCP Application Default Credentials configured

Required GCP access:
- Network path to Memorystore from your runtime environment (for example Cloud Run via VPC connector)
- GCS read access to HyDE data bucket/prefixes
- Vertex AI Matching Engine query access for configured endpoint/index
- BigQuery read access to fallback table

## 5. Local setup

Install dependencies:

Option A (`uv`):
```bash
uv sync
```

Option B (`venv` + pip):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run API:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload
```

Quick checks:
```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/recommendations \
  -H "Content-Type: application/json" \
  -d '{"student_id":"student_123"}'
```

Note:
- In this project, Redis runs on Google Cloud Memorystore and is not expected to work from a purely local environment.
- Use local run mainly for API wiring/basic checks; run deployed tests (for example on Cloud Run with VPC connector) for end-to-end recommendation behavior.

## 6. Project Structure

```text
api/
  app.py                       # FastAPI app, middleware, endpoints
  schema.py                    # Request/response models
modules/
  core/recommend_feeds.py      # Main recommendation orchestration
  functions/
    hyde_embedding.py          # GCS embedding/query/metadata loader
    vector_search.py           # Vertex Matching Engine client
    bigquery_fallback.py       # BigQuery fallback retrieval
    trigger_hyde_generation.py # Background HyDE-generation trigger
  services/
    vector_recommendation.py   # Async vector search + prep
    recommend_with_subscore.py # Subscore reranking + response formatting
    calc_subscore.py           # Score aggregation logic
  utils/
    load_config.py             # YAML + env config loader
    redis.py                   # Redis wrapper
    gcs.py                     # GCS helper
    bigquery.py                # BigQuery helper
    performance_logging.py     # Timing log emission
  parameters/
    config.yaml
    retrieval_score_weights.yaml
locustfile.py
test_metrics/
  run_api_retrieval_metrics.py
  test_metrics_config.yaml
tests/
  conftest.py
  test_recommendation_service.py
  test_recommendation_helpers.py
cloudbuild.yaml
Dockerfile
requirements.txt
pyproject.toml
```

## 7. Data & Dependencies

Key data contracts:
- Request key: `student_id` (non-empty string).
- Redis recommendation cache key: `recommendations:{student_id}`.
- Redis feed metadata key pattern for fallback: `feeds:{feed_id}`.
- BigQuery fallback table must contain a `feed_id` column.

External dependencies:
- Redis: cache reads/writes and feed metadata fallback pool.
- GCS: student embeddings and supporting HyDE artifacts.
- Vertex AI: nearest-neighbor retrieval.
- BigQuery: fallback feeds when Redis feed cache is insufficient.

Dependency outage behavior:
- Redis unavailable: cache may miss/fail gracefully; service still attempts vector/fallback path.
- Missing embeddings: fallback path is used and HyDE generation trigger is attempted.
- Vertex search issues: fallback path is used.
- BigQuery schema/config errors: can raise runtime error.

## 8. API Contract

### 8.1 API Endpoints

`GET /health`
- Purpose: liveness check.
- Success response: `{"status":"ok"}`.

`POST /recommendations`
- Purpose: return ranked recommendations for one student.
- Response headers include `x-response-time-seconds`, `X-API-Version`, and `X-Correlation-Id`.
- Requests accept both `student_id` and `studentId`; declared response fields are emitted in camelCase.

### 8.2 Example - Request, Response

Request:
```json
{
  "studentId": "student_123"
}
```

Response example:
```json
{
  "studentId": "student_123",
  "source": "vertex_vector_search",
  "recommendations": [
    {
      "feedId": "TH_FEED_001",
      "score": 0.923451,
      "metadata": {
        "title": "...",
        "language": "th"
      }
    }
  ]
}
```

Response fields:
- `studentId`: echoed student id.
- `source`: serving source, e.g. `redis_cache`, `vertex_vector_search`, `redis_fallback`, `bigquery_fallback`, or combined source after top-up.
- `recommendations`: ranked items with `feedId`, `score`, and optional `metadata`.

### 8.3 Error Handling & Status Codes

- `200 OK`: health success or recommendation success (including fallback-served results).
- `422 Unprocessable Entity`: request validation error (e.g., missing/empty `student_id`).
- `500 Internal Server Error`: runtime exception raised to API layer (for example misconfigured fallback table).

Notes:
- Many vector-path failures are handled internally and return `200` with fallback source instead of returning `500`.
- FastAPI default validation/error payload format is used for non-200 responses.

## 9. Configuration (config.yaml, env vars)

Main config file: `modules/parameters/config.yaml`

Important sections:
- `app`: host, port, performance-log sampling rate.
- `cache`: Redis host/port, cache TTL, Redis timeout.
- `hyde_data`: GCS bucket and embedding/query/metadata prefixes.
- `vertex`: endpoint, deployed index id, neighbor count, restrict filters.
- `bigquery`: fallback table and fallback limit.
- `trigger_hyde_generation`: remote API URL/path, timeout, cooldown.
- `recommendation`: minimum recommendation count.

Supported environment variable overrides:
- `REDISHOST`
- `REDISPORT`
- `REDIS_TIMEOUT_SECONDS`
- `PORT` (container runtime port in Docker/Cloud Run command)

## 10. Testing

### 10.1 pytest

Run all tests:
```bash
python -m pytest -q
```

Run recommendation-focused tests:
```bash
python -m pytest -q tests/test_recommendation_service.py tests/test_recommendation_helpers.py
```

### 10.2 Load Testing (Locust)

Input CSV:
- `test_metrics/prep_stuid_locust/student_ids.csv`
- Required column: `student_id`

Run:
```bash
locust -f locustfile.py --host http://localhost:8080
```

UI:
- `http://localhost:8089`

### 10.3 Retrieval Quality Evaluation

Script:
- `test_metrics/run_api_retrieval_metrics.py`

Config:
- `test_metrics/test_metrics_config.yaml`

Metrics:
- `MRR@K`
- `Precision@K`
- `Hit@K`

Run:
```bash
python test_metrics/run_api_retrieval_metrics.py --config test_metrics/test_metrics_config.yaml
```

Optional limit:
```bash
python test_metrics/run_api_retrieval_metrics.py --config test_metrics/test_metrics_config.yaml --limit 100
```

## 11. Build & Deploy to Cloud Run (Cloud Build)

`cloudbuild.yaml` performs:
1. Docker build
2. Push to Artifact Registry
3. Deploy to Cloud Run

Default substitutions:
- `_REGION=asia-southeast1`
- `_REPO_NAME=online-serving-feeds-api`
- `_VPC_CONNECTOR_NAME=redis-test-connector`
- `_SA_NAME=test-result-data-api-sa`
- `_REDIS_HOST=10.86.221.99`
- `_REDIS_PORT=6379`

Deploy with defaults:
```bash
gcloud builds submit --config cloudbuild.yaml
```

Deploy with overrides:
```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_REGION=asia-southeast1,_REPO_NAME=<service-name>,_VPC_CONNECTOR_NAME=<vpc-connector>,_SA_NAME=<service-account-name>,_REDIS_HOST=<redis-host>,_REDIS_PORT=6379
```

Deployed endpoint (current):
- Base URL: `https://online-serving-feeds-api-810737581373.asia-southeast1.run.app`
- Swagger: `https://online-serving-feeds-api-810737581373.asia-southeast1.run.app/docs`

## 12. Observability

- Middleware emits `x-response-time-seconds` in every response.
- Recommendation timing logs are emitted from `modules/utils/performance_logging.py`.
- Sampling is controlled by `app.perf_log_sample_rate`.
- Core timing fields include:
  - `cache_hit`
  - `t_total`
  - `t_cache_get`
  - `t_vector_search`
  - `t_postprocess`
  - `t_fallback_prepare`
  - `t_rerank`
  - `t_metadata_fetch`
  - `t_format_response`
  - `t_top_up_merge`
  - `t_response_write`

## 13. Common Troubleshootings

Redis connection errors:
- Check `REDISHOST`, `REDISPORT`, VPC/network route, and Redis timeout settings.

No embeddings found in GCS:
- Verify bucket/prefix settings in `hyde_data`.
- Verify service account can read GCS objects.

Vertex retrieval failures:
- Validate `vertex.index_endpoint` and `vertex.deployed_index_id`.
- Confirm endpoint/index is deployed and accessible in the same region/project context.

BigQuery fallback errors:
- Ensure `bigquery.fallback_table` exists and includes `feed_id`.
- Confirm BigQuery permissions for the runtime service account.

`pytest` not found / stale entrypoint after folder rename:
- Use `python -m pytest -q`.
- Recreate virtualenv if entrypoint paths are stale.

## 14. Important implementation details

- Fallback uses Redis feed cache first (if enough `feeds:*` keys), then BigQuery.
- Missing embeddings can trigger a background HyDE-generation API call (cooldown protected per student).
- If vector results are below `recommendation.minimum_recommendation`, fallback results are used to top up.
- Feed-level metadata in response is best-effort and can be `null`.

## 15. Performance & SLOs

Current status:
- The service exposes timing fields (`x-response-time-seconds` and detailed timing logs), but no hard SLO is enforced in code.

Recommended operational tracking:
- P50/P95/P99 latency of `POST /recommendations`
- Cache-hit ratio (`cache_hit=true`)
- Fallback source ratio (`redis_fallback` / `bigquery_fallback`)
- Error rate by status code

## 16. Known Limitations / Assumptions

- Recommendation quality depends on availability and freshness of HyDE embeddings.
- Fallback ranking quality can be lower than vector-search ranking.
- BigQuery fallback assumes table schema includes `feed_id`.
- No built-in rate limiting or auth layer is implemented in this service.
- Per-student HyDE refresh cooldown is in-memory (not shared across instances).

## 17. Versioning / Change Log

Current API metadata version in app: `0.1.0`.

Change log:
- `2026-03-09`: README restructured with API contract, data dependencies, troubleshooting, performance, limitations, and verification flow sections.

## 18. Quick verification flow

1. Install dependencies (`uv sync` or `pip install -r requirements.txt`).
2. Build and deploy service to Cloud Run with VPC connector and Redis env vars.
3. Confirm runtime service account has access to Memorystore path, GCS, Vertex AI, and BigQuery.
4. Call deployed `GET /health` and expect `200`.
5. Call deployed `POST /recommendations` with a known `student_id`.
6. Validate response shape and `source`.
7. Re-call same `student_id` and confirm cache behavior (`source=redis_cache` expected after cache write).
