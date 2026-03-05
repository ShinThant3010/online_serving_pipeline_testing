from typing import Any

from modules.utils.gcs import load_data_from_gcs_prefix


class HydeEmbeddingStore:
    def __init__(
        self,
        bucket: str,
        embedding_prefix: str,
        query_prefix: str,
        metadata_prefix: str,
    ) -> None:
        self.bucket = bucket.strip("/")
        self.embedding_prefix = embedding_prefix.strip("/")
        self.query_prefix = query_prefix.strip("/")
        self.metadata_prefix = metadata_prefix.strip("/")

    def _build_gcs_prefix(self, student_id: str, prefix: str) -> str:
        return "gs://" + "/".join(
            part for part in [self.bucket, student_id.strip("/"), prefix.strip("/")] if part
        )

    @staticmethod
    def _to_valid_vector(item: Any) -> list[float] | None:
        """Normalize a candidate embedding into a flat non-zero float vector."""
        if not isinstance(item, list) or not item:
            return None

        # Some .npy payloads are a 2D matrix in one object: [[...], [...]].
        if all(isinstance(row, list) for row in item):
            return None

        try:
            vector = [float(value) for value in item]
        except (TypeError, ValueError):
            return None

        return vector if any(value != 0.0 for value in vector) else None

    def _normalize_hyde_query_payload(self, items: list[Any]) -> dict[str, Any]:
        """
        Normalize query payload to {'hq': [query_dict, ...]} using basic structure checks.
        Supported input shapes:
        - first item is {'hq': [dict, ...]}
        - first item is [dict, ...]
        - items is [dict, ...]
        """
        if not items:
            return {}

        first = items[0]
        if isinstance(first, dict):
            if not isinstance(first.get("hq"), list):
                return {}
            hq = [q for q in first["hq"] if isinstance(q, dict)]
            return {**first, "hq": hq} if hq else {}

        source = first if isinstance(first, list) else items
        hq = [q for q in source if isinstance(q, dict)]
        return {"hq": hq} if hq else {}

    def _normalize_metadata_payload(self, items: list[Any]) -> dict[str, Any]:
        """
        Normalize loaded metadata payload to a dict.
        Returns {} for unsupported structures.
        """
        if not items:
            return {}

        first = items[0]
        if isinstance(first, dict):
            return first

        if isinstance(first, list):
            dict_rows = [row for row in first if isinstance(row, dict)]
            return {"interaction": dict_rows} if dict_rows else {}

        dict_items = [row for row in items if isinstance(row, dict)]
        if not dict_items:
            return {}

        # If multiple dict records look like interaction rows, expose them under "interaction".
        is_interaction_rows = all(
            any(key in row for key in ("student_id", "user_id", "feed_id", "event_type"))
            for row in dict_items
        )
        return {"interaction": dict_items} if is_interaction_rows else {"items": dict_items}

    def load_embeddings(self, student_id: str) -> list[list[float]]:
        """
        Load embeddings for a given student ID from GCS.
        """
        if not self.bucket or not student_id or not self.embedding_prefix:
            return []

        gcs_prefix = self._build_gcs_prefix(student_id, self.embedding_prefix)

        items = load_data_from_gcs_prefix(
            gcs_prefix,
            file_type="npy",
        )
        if not items:
            return []

        vectors: list[list[float]] = []
        for item in items:
            candidates = item if isinstance(item, list) and item and all(isinstance(row, list) for row in item) else [item]
            for candidate in candidates:
                if (vector := self._to_valid_vector(candidate)) is not None:
                    vectors.append(vector)

        # if not vectors:
            # print(f"No valid embeddings found for {student_id}.")
        return vectors

    def load_hyde_queries(self, student_id: str) -> dict[str, Any]:
        """
        Load and validate HyDE query payload for a given student ID from GCS.
        """
        if not self.bucket or not student_id or not self.query_prefix:
            return {}

        gcs_prefix = self._build_gcs_prefix(student_id, self.query_prefix)
        items = load_data_from_gcs_prefix(gcs_prefix, file_type="json")
        return self._normalize_hyde_query_payload(items)

    def load_metadata(self, student_id: str) -> dict[str, Any]:
        """
        Load and validate metadata payload for a given student ID from GCS.
        """
        if not self.bucket or not student_id or not self.metadata_prefix:
            return {}

        gcs_prefix = self._build_gcs_prefix(student_id, self.metadata_prefix)
        items = load_data_from_gcs_prefix(gcs_prefix, file_type="json")
        return self._normalize_metadata_payload(items)
