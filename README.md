# Online Serving Pipeline (Feeds Recommendation API)

This project provides a FastAPI service that returns feed recommendations for a `student_id`.

It combines:
- Redis cache lookup
- HyDE embeddings from GCS
- Vertex AI Matching Engine vector search
- Subscore-based reranking
- Fallback recommendations from Redis feed cache or BigQuery

## 1. What This Service Does

`POST /recommendations` returns a ranked list of feeds for a student.

High-level flow:
1. Try Redis cache (`recommendations:{student_id}`).
2. If cache miss, load student embeddings from GCS.
3. If embeddings exist, run Vertex vector search and rerank results with subscore logic.
4. If embeddings are missing or vector search fails, fallback to Redis feed cache or BigQuery.
5. If final result count is below minimum, top up with fallback results.
6. Cache successful recommendation response back to Redis.

## 2. API Endpoints

### `GET /health`
Returns service health.

Example response:
```json
{"status": "ok"}
```

### `POST /recommendations`
Request body:
```json
{
  "student_id": "student_123"
}
```

Response body:
```json
{
  "student_id": "student_123",
  "source": "vertex_vector_search",
  "recommendations": [
    {
      "feed_id": "TH_FEED_001",
      "score": 0.923451,
      "metadata": {
        "title": "...",
        "language": "th"
      }
    }
  ]
}
```

Response header includes:
- `x-response-time-seconds`: server-side request duration

## 3. Project Structure

```text
api/
  app.py                  # FastAPI app + middleware + endpoints
  schema.py               # Request/response models
modules/
  core/recommend_feeds.py # Main recommendation orchestration
  functions/
    hyde_embedding.py     # Load embeddings/query/metadata from GCS
    vector_search.py      # Vertex Matching Engine client
    bigquery_fallback.py  # Fallback feed retrieval from BigQuery
    trigger_hyde_generation.py
  services/
    vector_recommendation.py   # Async vector search + rerank prep
    recommend_with_subscore.py # Subscore reranking + formatting
    calc_subscore.py           # Score aggregation/subscore logic
  utils/
    load_config.py        # YAML/env config loader
    redis.py              # Redis wrapper
    gcs.py                # GCS read/write helpers
    bigquery.py           # BigQuery SQL helper
    performance_logging.py
  parameters/
    config.yaml
    retrieval_score_weights.yaml
locustfile.py             # Load test scenario
test_metrics/
  run_api_retrieval_metrics.py # Offline retrieval metric runner
  test_metrics_config.yaml
Dockerfile
requirements.txt
pyproject.toml
```

## 4. Requirements

- Python `3.11+`
- Redis instance reachable from your environment
- GCP authentication (Application Default Credentials) for:
  - GCS
  - Vertex AI
  - BigQuery

## 5. Setup (Local)

### Option A: `uv` (recommended if you use `uv.lock`)
```bash
uv sync
```

### Option B: `venv` + pip
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 6. Configuration

Main config file: `modules/parameters/config.yaml`

Important sections:
- `app`: API host/port and perf log sampling
- `cache`: Redis host/port/TTL/timeout
- `hyde_data`: GCS bucket and prefixes for embedding/query/metadata
- `vertex`: index endpoint, deployed index id, neighbor count, filters
- `bigquery`: fallback table and limit
- `trigger_hyde_generation`: remote endpoint + cooldown
- `recommendation`: minimum recommendation count

Environment variable overrides currently supported:
- `REDISHOST`
- `REDISPORT`
- `REDIS_TIMEOUT_SECONDS`
- `PORT` (used by Docker CMD / Cloud Run style execution)

## 7. Run the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload
```

Health check:
```bash
curl http://localhost:8080/health
```

Recommendation test:
```bash
curl -X POST http://localhost:8080/recommendations \
  -H "Content-Type: application/json" \
  -d '{"student_id":"student_123"}'
```

## 8. Run with Docker

Build image:
```bash
docker build -t online-serving-pipeline .
```

Run container:
```bash
docker run --rm -p 8080:8080 \
  -e REDISHOST=<redis-host> \
  -e REDISPORT=6379 \
  online-serving-pipeline
```

## 9. Load Testing (Locust)

`locustfile.py` reads student IDs from:
- `test_metrics/prep_stuid_locust/student_ids.csv`

Expected CSV column:
- `student_id`

Run Locust UI:
```bash
locust -f locustfile.py --host http://localhost:8080
```

Then open `http://localhost:8089`.

## 10. Retrieval Quality Evaluation

Script:
- `test_metrics/run_api_retrieval_metrics.py`

Config:
- `test_metrics/test_metrics_config.yaml`

Default metrics:
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

Output CSV path is configured in `output.csv` in the metrics config.

## 11. Observability

- Timing logs are emitted as JSON via `print` in `modules/utils/performance_logging.py`.
- Sampling controlled by `app.perf_log_sample_rate` in `config.yaml`.
- Key fields include:
  - `cache_hit`
  - `t_total`
  - `t_cache_get`
  - `t_vector_search`
  - `t_postprocess`
  - `t_response_write`

## 12. Common Failure Modes

- Redis unreachable:
  - Cache calls fail gracefully and warnings are printed.
- Missing GCS embeddings:
  - Service triggers HyDE generation request and falls back.
- Vertex endpoint/index misconfigured:
  - Runtime fallback to Redis/BigQuery recommendations.
- BigQuery fallback table missing or schema invalid:
  - Raises runtime error (must include `feed_id` column).

## 13. Notes for New Developers

- Existing `README.md` was empty; this file is the operational guide.
- There are currently no unit tests in this repository.
- Primary behavior lives in `modules/core/recommend_feeds.py`; start there when debugging recommendation logic.
