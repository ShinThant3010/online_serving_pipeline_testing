from typing import Any

from google.api_core.exceptions import NotFound
from google.cloud import aiplatform


def _parse_project_and_region(index_endpoint: str) -> tuple[str, str]:
    """
    Parses the project ID and region from a Vertex AI index endpoint resource name.
    """
    parts = index_endpoint.strip("/").split("/")
    if len(parts) < 6:
        raise RuntimeError(
            "VERTEX_INDEX_ENDPOINT must be a full resource path like "
            "'projects/<project>/locations/<region>/indexEndpoints/<id>'."
        )
    if parts[0] != "projects" or parts[2] != "locations":
        raise RuntimeError(
            "VERTEX_INDEX_ENDPOINT must include project and region "
            "in the format 'projects/<project>/locations/<region>/indexEndpoints/<id>'."
        )
    return parts[1], parts[3]


class VectorSearchClient:

    ### ------------------------------- initialize settings ------------------------------- ###
    def __init__(
        self,
        index_endpoint: str,
        deployed_index_id: str,
        neighbor_count: int,
        return_full_datapoint: bool = False,
        restricts_list: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize Vertex Matching Engine client settings and endpoint connection.
        Also prebuild default namespace restrict filters from config input.
        """
        self.index_endpoint = index_endpoint
        self.deployed_index_id = deployed_index_id
        self.neighbor_count = neighbor_count
        self.return_full_datapoint = return_full_datapoint
        self.default_restricts = self._build_restricts(restricts_list)
        self.project_id, self.region = _parse_project_and_region(index_endpoint)
        aiplatform.init(project=self.project_id, location=self.region)
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=self.index_endpoint
        )


    ### ------------------------- helper: format restricts for filtering ------------------------- ###
    @staticmethod
    def _build_restricts(restricts_list: dict[str, Any] | None) -> list[Any] | None:
        """
        Convert a namespace->tokens mapping into Vertex AI Namespace filters.
        Returns None when input is empty/invalid or no usable tokens exist.
        """
        if not isinstance(restricts_list, dict) or not restricts_list:
            return None

        restricts: list[Any] = []
        namespace_cls = aiplatform.matching_engine.matching_engine_index_endpoint.Namespace
        for namespace, raw_tokens in restricts_list.items():
            if raw_tokens is None:
                continue
            if isinstance(raw_tokens, list):
                tokens = [str(token) for token in raw_tokens if token is not None and str(token)]
            else:
                token = str(raw_tokens)
                tokens = [token] if token else []
            if not tokens:
                continue
            restricts.append(namespace_cls(name=str(namespace), allow_tokens=tokens))

        return restricts or None


# ---------------------------------------------------------------------------------------------
# main function: trigger hyde generation of student
# ---------------------------------------------------------------------------------------------
    def search(self, embeddings: list[list[float]], restricts: list[Any] | None = None) -> list[dict[str, Any]]:
        """
        Performs a vector search using the Vertex AI Matching Engine.
        """
        if not self.index_endpoint or not self.deployed_index_id:
            raise RuntimeError(
                "Vertex AI Vector Search is not configured. "
                "Set VERTEX_INDEX_ENDPOINT and VERTEX_DEPLOYED_INDEX_ID."
            )
        try:
            effective_restricts = restricts if restricts is not None else self.default_restricts
            neighbors = self.endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=embeddings,
                num_neighbors=self.neighbor_count,
                return_full_datapoint=self.return_full_datapoint,
                filter=effective_restricts or None,
            )
        except NotFound as exc:
            raise RuntimeError(f"Vertex index endpoint not found: {exc}") from exc

        # The neighbors object is a list of lists of Neighbor objects. Each Neighbor has attributes like id and distance.
        result = [{"feed_id": n.id, "score": float(n.distance)} for group in neighbors for n in group if n.id and n.distance is not None]
            
        return result
