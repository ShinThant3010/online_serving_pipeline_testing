import json
from typing import Any

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError


class RedisCache:

# ---------------------------------------------------------------------------------------------
# Initialization Redis client with connection parameters and timeouts
# ---------------------------------------------------------------------------------------------
    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_seconds: float = 1.0,
        decode_responses: bool = True,
    ) -> None:
        self.redis_client = Redis(
            host=host,
            port=port,
            decode_responses=decode_responses,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )


# ---------------------------------------------------------------------------------------------
# Public methods for getting and setting cache values, with error handling and logging
# ---------------------------------------------------------------------------------------------
    @staticmethod
    def _load_json(payload: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    ### ---------------------------------- get one item ---------------------------------- ###
    def get_one(self, key: str) -> dict[str, Any] | None:
        """
        Retrieves a value from Redis by key. Returns None if the key does not exist or if there is a connection error.
        """
        try:
            payload = self.redis_client.get(key)
            if not payload:
                return None
            return self._load_json(payload)
        except (RedisConnectionError, RedisTimeoutError) as exc:
            print(f"Warning: Redis get failed for key '{key}': {exc}")
            return None

    ### ---------------------------------- get many items ---------------------------------- ###
    def get_many(self, keys: list[str]) -> dict[str, dict[str, Any] | None]:
        """
        Retrieves multiple values from Redis by a list of keys. 
        Returns a dictionary mapping keys to their values (or None if a key does not exist or if there is a connection error).
        """
        if not keys:
            return {}

        try:
            payloads = self.redis_client.mget(keys)
            return {
                key: self._load_json(payload) if payload is not None else None
                for key, payload in zip(keys, payloads)
            }
        except (RedisConnectionError, RedisTimeoutError) as exc:
            print(f"Warning: Redis get_many failed: {exc}")
            return {}

    ### ---------------------------------- set one item ---------------------------------- ###
    def set_one(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        """
        Sets a value in Redis with a time-to-live (TTL). Logs a warning if there is a connection error.
        """
        try:
            self.redis_client.setex(key, ttl_seconds, json.dumps(value))
        except (RedisConnectionError, RedisTimeoutError) as exc:
            print(f"Warning: Redis set failed for key '{key}': {exc}")

    ### ---------------------------------- set many items ---------------------------------- ###
    def set_many(self, mapping: dict[str, dict[str, Any]], ttl_seconds: int) -> int:
        """
        Sets multiple key-value pairs in Redis with a time-to-live (TTL). 
        Returns the number of keys successfully set. Logs a warning if there is a connection error.
        """
        if not mapping:
            return 0

        try:
            pipeline = self.redis_client.pipeline()
            for key, value in mapping.items():
                pipeline.setex(key, ttl_seconds, json.dumps(value))
            results = pipeline.execute()
            return sum(1 for result in results if result)
        except (RedisConnectionError, RedisTimeoutError) as exc:
            print(f"Warning: Redis set_many failed: {exc}")
            return 0

    ### --------------------- get many items those have specific prefix --------------------- ###
    def get_many_by_prefix(self, cache_prefix: str) -> list[str]:
        """
        Retrieves and returns sorted cache keys that match the given prefix.
        """
        pattern = f"{cache_prefix}:*"
        try:
            return sorted(str(key) for key in self.redis_client.scan_iter(match=pattern))
        except (RedisConnectionError, RedisTimeoutError) as exc:
            print(f"Warning: Redis get_many_by_prefix failed for prefix '{cache_prefix}': {exc}")
            return []

# ---------------------------------------------------------------------------------------------
# There are more functions reusable in main cache-redis pipeline
# set-one 
# set-many
# set-many-bigquery
# get-one/{item-id}
# get-many 
# get-many-by-prefix
# delete-one/{item-id}
# delete-prefix
# clear-all
# memory-usage
# ---------------------------------------------------------------------------------------------
