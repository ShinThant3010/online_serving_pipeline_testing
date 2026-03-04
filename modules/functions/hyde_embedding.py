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
        if item and all(isinstance(row, list) for row in item):
            return None

        try:
            vector = [float(value) for value in item]
        except (TypeError, ValueError):
            return None

        if not vector or not any(value != 0.0 for value in vector):
            return None
        return vector

    @staticmethod
    def _is_query_object(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        # Accept query objects with any common query field.
        return any(key in item for key in ("query_id", "query_text", "weight", "intent_label"))

    def _normalize_hyde_query_payload(self, items: list[Any], *, student_id: str) -> dict[str, Any]:
        """
        Normalize loaded query payload to {'hq': [query_dict, ...]} when possible.
        Returns {} for unsupported structures.
        """
        if not items:
            return {}

        first = items[0]
        if isinstance(first, dict):
            if isinstance(first.get("hq"), list):
                hq = [q for q in first["hq"] if isinstance(q, dict)]
                if hq:
                    payload = dict(first)
                    payload["hq"] = hq
                    return payload
                # print(f"Invalid HyDE query format for {student_id}: `hq` exists but has no valid objects.")
                return {}

            if self._is_query_object(first):
                return {"hq": [first]}

            if isinstance(first.get("items"), list):
                hq = [q for q in first["items"] if isinstance(q, dict)]
                if hq:
                    return {"hq": hq}

        if isinstance(first, list):
            hq = [q for q in first if isinstance(q, dict)]
            if hq:
                return {"hq": hq}

        hq = [q for q in items if isinstance(q, dict)]
        if hq:
            return {"hq": hq}

        # print(f"Invalid HyDE query payload for {student_id}: expected dict/list of query objects.")
        return {}

    def _normalize_metadata_payload(self, items: list[Any], *, student_id: str) -> dict[str, Any]:
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
            interactions = [row for row in first if isinstance(row, dict)]
            if interactions:
                return {"interaction": interactions}

        dict_items = [row for row in items if isinstance(row, dict)]
        if dict_items:
            # If multiple dict records look like interaction rows, expose them under "interaction".
            if all(any(k in row for k in ("student_id", "user_id", "feed_id", "event_type")) for row in dict_items):
                return {"interaction": dict_items}
            return {"items": dict_items}

        # print(f"Invalid metadata payload for {student_id}: expected dict or list of objects.")
        return {}

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
            if isinstance(item, list) and item and all(isinstance(row, list) for row in item):
                candidates = item
            else:
                candidates = [item]

            for candidate in candidates:
                vector = self._to_valid_vector(candidate)
                if vector is None:
                    continue
                vectors.append(vector)

        if not vectors:
            print(f"No valid embeddings found for {student_id}.")
        return vectors

    def load_hyde_queries(self, student_id: str) -> dict[str, Any]:
        """
        Load and validate HyDE query payload for a given student ID from GCS.
        """
        if not self.bucket or not student_id or not self.query_prefix:
            return {}

        gcs_prefix = self._build_gcs_prefix(student_id, self.query_prefix)
        items = load_data_from_gcs_prefix(gcs_prefix, file_type="json")
        return self._normalize_hyde_query_payload(items, student_id=student_id)

    def load_metadata(self, student_id: str) -> dict[str, Any]:
        """
        Load and validate metadata payload for a given student ID from GCS.
        """
        if not self.bucket or not student_id or not self.metadata_prefix:
            return {}

        gcs_prefix = self._build_gcs_prefix(student_id, self.metadata_prefix)
        items = load_data_from_gcs_prefix(gcs_prefix, file_type="json")
        return self._normalize_metadata_payload(items, student_id=student_id)
