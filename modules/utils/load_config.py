import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------------------------
# Declare data classes
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    perf_log_sample_rate: float

@dataclass(frozen=True)
class CacheConfig:
    redis_host: str
    redis_port: int
    ttl_seconds: int
    redis_timeout_seconds: float

@dataclass(frozen=True)
class HydeDataConfig:
    bucket: str

@dataclass(frozen=True)
class VertexConfig:
    index_endpoint: str
    deployed_index_id: str
    neighbor_count: int
    return_full_datapoint: bool
    restricts_list: dict[str, Any]

@dataclass(frozen=True)
class BigQueryConfig:
    fallback_table: str
    fallback_limit: int

@dataclass(frozen=True)
class TriggerHydeGenerationConfig:
    http_timeout_seconds: float
    refresh_cooldown_seconds: float
    recommendation_api_base_url: str
    recommendation_path: str

@dataclass(frozen=True)
class RecommendationConfig:
    minimum_recommendation: int

@dataclass(frozen=True)
class Settings:
    app: AppConfig
    cache: CacheConfig
    hyde_data: HydeDataConfig
    vertex: VertexConfig
    bigquery: BigQueryConfig
    trigger_hyde_generation: TriggerHydeGenerationConfig
    recommendation: RecommendationConfig


# ---------------------------------------------------------------------------------------------
# Helper functions for validation
# ---------------------------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    section = payload.get(key, {})
    return section if isinstance(section, dict) else {}


# ---------------------------------------------------------------------------------------------
# Main function: load settings 
# ---------------------------------------------------------------------------------------------
def load_settings(config_path: str = "modules/parameters/config.yaml") -> Settings:
    load_dotenv()
    cfg = _read_yaml(Path(config_path))

    app_cfg = _section(cfg, "app")
    cache_cfg = _section(cfg, "cache")
    hyde_data_cfg = _section(cfg, "hyde_data")
    vertex_cfg = _section(cfg, "vertex")
    bq_cfg = _section(cfg, "bigquery")
    trigger_hyde_generation_cfg = _section(cfg, "trigger_hyde_generation")
    recommendation_cfg = _section(cfg, "recommendation")

    return Settings(
        app=AppConfig(
            host=str(app_cfg.get("host", "0.0.0.0")),
            port=_to_int(app_cfg.get("port"), 8080),
            perf_log_sample_rate=_to_float(app_cfg.get("perf_log_sample_rate"), 1.0),
        ),
        cache=CacheConfig(
            redis_host=str(os.getenv("REDISHOST", cache_cfg.get("redis_host", "localhost"))),
            redis_port=_to_int(os.getenv("REDISPORT"), _to_int(cache_cfg.get("redis_port"), 6379)),
            ttl_seconds=_to_int(cache_cfg.get("ttl_seconds"), 3600),
            redis_timeout_seconds=_to_float(
                os.getenv("REDIS_TIMEOUT_SECONDS", cache_cfg.get("redis_timeout_seconds")),
                1.0,
            ),
        ),
        hyde_data=HydeDataConfig(
            bucket=str(hyde_data_cfg.get("bucket", "hyde-datalake-feeds")),
        ),
        vertex=VertexConfig(
            index_endpoint=str(vertex_cfg.get("index_endpoint", "")),
            deployed_index_id=str(vertex_cfg.get("deployed_index_id", "")),
            neighbor_count=_to_int(vertex_cfg.get("neighbor_count"), 20),
            return_full_datapoint=_to_bool(vertex_cfg.get("return_full_datapoint"), False),
            restricts_list=(
                vertex_cfg.get("restricts_list")
                if isinstance(vertex_cfg.get("restricts_list"), dict)
                else {}
            ),
        ),
        bigquery=BigQueryConfig(
            fallback_table=str(bq_cfg.get("fallback_table", "")),
            fallback_limit=_to_int(bq_cfg.get("fallback_limit"), 20),
        ),
        trigger_hyde_generation=TriggerHydeGenerationConfig(
            http_timeout_seconds=_to_float(
                trigger_hyde_generation_cfg.get("http_timeout_seconds"),
                60.0,
            ),
            refresh_cooldown_seconds=_to_float(
                trigger_hyde_generation_cfg.get("refresh_cooldown_seconds"),
                300.0,
            ),
            recommendation_api_base_url=str(
                trigger_hyde_generation_cfg.get(
                    "recommendation_api_base_url",
                    "https://hyderecomment-service-du7yhkyaqq-as.a.run.app",
                )
            ),
            recommendation_path=str(
                trigger_hyde_generation_cfg.get(
                    "recommendation_path",
                    "/hyde/students/{student_id}",
                )
            ),
        ),
        recommendation=RecommendationConfig(
            minimum_recommendation=_to_int(
                recommendation_cfg.get("minimum_recommendation"),
                10,
            ),
        ),
    )
