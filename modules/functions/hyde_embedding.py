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
        """
        Standardize GCS URI prefix in the form:
        `gs://{bucket}/{student_id}/{prefix}`.

        Leading/trailing slashes are removed from each segment and empty segments are skipped.
        """
        return "gs://" + "/".join(
            part for part in [self.bucket, student_id.strip("/"), prefix.strip("/")] if part
        )

    ### --------------------------- Validate loaded embeddings --------------------------- ###
    @staticmethod
    def _to_valid_embeddings_payload(items: list[Any]) -> list[list[float]]:
        """Normalize embeddings payload into a list of flat non-zero float vectors."""
        if not items:
            return []

        vectors: list[list[float]] = []
        for item in items:
            candidates = (
                item
                if isinstance(item, list) and item and all(isinstance(row, list) for row in item)
                else [item]
            )
            for candidate in candidates:
                if not isinstance(candidate, list) or not candidate:
                    continue

                # 2D matrix nested deeper: [[[...], [...]]]
                if all(isinstance(row, list) for row in candidate):
                    continue

                try:
                    vector = [float(value) for value in candidate]
                except (TypeError, ValueError):
                    continue

                if any(value != 0.0 for value in vector):
                    vectors.append(vector)

        return vectors

    ### --------------------------- Validate loaded hyde query --------------------------- ###
    @staticmethod
    def _to_valid_hyde_query_payload(items: list[Any]) -> list[dict[str, Any]]:
        """
        Normalize query payload to a list of query dicts using basic structure checks.
        Supported input shapes:
        - first item is {'hq': [dict, ...]}
        - first item is [dict, ...]
        - items is [dict, ...]
        """
        if not items:
            return []

        first = items[0]
        if isinstance(first, dict):
            hq = first.get("hq")
            if not isinstance(hq, list):
                return []
            return [q for q in hq if isinstance(q, dict)]

        source = first if isinstance(first, list) else items
        return [q for q in source if isinstance(q, dict)]

    ### ---------------------------- Validate loaded metadata ---------------------------- ###
    @staticmethod
    def _to_valid_metadata_payload(items: list[Any]) -> dict[str, Any]:
        """
        Validate metadata payload as a single student-profile dict.
        Returns {} for unsupported structures.
        """
        if not items or not isinstance(items[0], dict):
            return {}

        payload = items[0]

        student_id = payload.get("student_id")
        if not isinstance(student_id, str) or not student_id.strip():
            return {}

        interaction = payload.get("interaction")
        if interaction is not None:
            if not isinstance(interaction, list):
                return {}
            payload["interaction"] = [row for row in interaction if isinstance(row, dict)]

        return payload

    
# ---------------------------------------------------------------------------------------------
# Load embeddings from hyde-data-lake
# ---------------------------------------------------------------------------------------------
    def load_embeddings(self, student_id: str) -> list[list[float]]:
        """
        Load embeddings for a given student ID from GCS.
        """
        if not self.bucket or not student_id or not self.embedding_prefix:
            return []

        gcs_prefix = self._build_gcs_prefix(student_id, self.embedding_prefix)
        items = load_data_from_gcs_prefix(gcs_prefix,file_type="npy")
        return self._to_valid_embeddings_payload(items)


# ---------------------------------------------------------------------------------------------
# Load hyDE query from hyde-data-lake
# ---------------------------------------------------------------------------------------------
    def load_hyde_queries(self, student_id: str) -> list[dict[str, Any]]:
        """
        Load and validate HyDE query rows for a given student ID from GCS.
        """
        if not self.bucket or not student_id or not self.query_prefix:
            return []

        gcs_prefix = self._build_gcs_prefix(student_id, self.query_prefix)
        items = load_data_from_gcs_prefix(gcs_prefix, file_type="json")
        return self._to_valid_hyde_query_payload(items)


# ---------------------------------------------------------------------------------------------
# Load metadata from hyde-data-lake
# ---------------------------------------------------------------------------------------------
    def load_metadata(self, student_id: str) -> dict[str, Any]:
        """
        Load and validate metadata payload for a given student ID from GCS.
        """
        if not self.bucket or not student_id or not self.metadata_prefix:
            return {}

        gcs_prefix = self._build_gcs_prefix(student_id, self.metadata_prefix)
        items = load_data_from_gcs_prefix(gcs_prefix, file_type="json")
        return self._to_valid_metadata_payload(items)
