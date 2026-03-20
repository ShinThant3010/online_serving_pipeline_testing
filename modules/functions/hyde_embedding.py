from typing import Any

from modules.utils.gcs import load_json_from_gcs_uri


class HydeEmbeddingStore:
    HYDE_BUNDLE_BUCKET = "hyde-datalake-feeds"
    HYDE_BUNDLE_FILENAME = "hyde_bundle.json"

    def __init__(
        self,
        bucket: str | None = None,
    ) -> None:
        self.bucket = (bucket or self.HYDE_BUNDLE_BUCKET).strip()

    def _build_bundle_gcs_uri(self, student_id: str) -> str:
        """
        Standardize GCS object URI in the form:
        `gs://hyde-datalake-feeds/{student_id}/hyde_bundle.json`.
        """
        clean_student_id = student_id.strip("/")
        return f"gs://{self.bucket}/{clean_student_id}/{self.HYDE_BUNDLE_FILENAME}"

    def _load_bundle(self, student_id: str) -> dict[str, Any]:
        if not self.bucket or not student_id:
            return {}

        gcs_uri = self._build_bundle_gcs_uri(student_id)
        payload = load_json_from_gcs_uri(gcs_uri)
        return payload if isinstance(payload, dict) else {}

    ### --------------------------- Validate loaded embeddings --------------------------- ###
    @staticmethod
    def _to_valid_embeddings_payload(bundle: dict[str, Any]) -> list[list[float]]:
        """Normalize bundle embeddings payload into a list of flat non-zero float vectors."""
        raw_embeddings = bundle.get("embeddings")
        if not isinstance(raw_embeddings, dict):
            return []

        vectors: list[list[float]] = []
        for key in sorted(raw_embeddings):
            candidate = raw_embeddings.get(key)
            if not isinstance(candidate, list) or not candidate:
                continue

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
    def _to_valid_hyde_query_payload(bundle: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Normalize query payload to a list of query dicts using basic structure checks.
        """
        hyde_queries = bundle.get("hyde_queries")
        if not isinstance(hyde_queries, list):
            return []
        return [q for q in hyde_queries if isinstance(q, dict)]

    ### ---------------------------- Validate loaded metadata ---------------------------- ###
    @staticmethod
    def _to_valid_metadata_payload(bundle: dict[str, Any]) -> dict[str, Any]:
        """
        Validate metadata payload as a single student-profile dict.
        Returns {} for unsupported structures.
        """
        payload = bundle.get("metadata")
        if not isinstance(payload, dict):
            return {}

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
        return self._to_valid_embeddings_payload(self._load_bundle(student_id))


# ---------------------------------------------------------------------------------------------
# Load hyDE query from hyde-data-lake
# ---------------------------------------------------------------------------------------------
    def load_hyde_queries(self, student_id: str) -> list[dict[str, Any]]:
        """
        Load and validate HyDE query rows for a given student ID from GCS.
        """
        return self._to_valid_hyde_query_payload(self._load_bundle(student_id))


# ---------------------------------------------------------------------------------------------
# Load metadata from hyde-data-lake
# ---------------------------------------------------------------------------------------------
    def load_metadata(self, student_id: str) -> dict[str, Any]:
        """
        Load and validate metadata payload for a given student ID from GCS.
        """
        return self._to_valid_metadata_payload(self._load_bundle(student_id))
