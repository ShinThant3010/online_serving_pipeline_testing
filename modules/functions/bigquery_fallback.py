import json

from google.cloud import bigquery

from api.schema import FeedsMetadata
from modules.utils.bigquery import query_sql


# ---------------------------------------------------------------------------------------------
# bigquery - fallback
# ---------------------------------------------------------------------------------------------
def fetch_fallback_recommendations(
    bigquery_client: bigquery.Client,
    fallback_table: str,
    fallback_limit: int,
) -> list[tuple[str, FeedsMetadata | None]]:
    """
    Fetches fallback recommendations from a BigQuery table.
    """
    if not fallback_table:
        raise RuntimeError("BQ_FALLBACK_TABLE is not configured.")

    table = bigquery_client.get_table(fallback_table)
    column_names = {field.name.lower() for field in table.schema}
    if "feed_id" not in column_names:
        raise RuntimeError(f"`{fallback_table}` must contain a `feed_id` column.")

    query = f"""
        SELECT
          CAST(feed_id AS STRING) AS feed_id,
          TO_JSON_STRING(t) AS metadata
        FROM `{fallback_table}`
        AS t
        LIMIT @limit
    """

    rows = query_sql(
        query,
        query_parameters=[bigquery.ScalarQueryParameter("limit", "INT64", fallback_limit)],
        client=bigquery_client,
    )

    items: list[tuple[str, FeedsMetadata | None]] = []

    for row in rows:
        feed_id = str(row["feed_id"])

        metadata_payload = row.get("metadata")
        metadata_dict = {}

        if isinstance(metadata_payload, str):
            try:
                parsed = json.loads(metadata_payload)
                if isinstance(parsed, dict):
                    metadata_dict = parsed
            except json.JSONDecodeError:
                metadata_dict = {}

        metadata = FeedsMetadata(**metadata_dict) if metadata_dict else None
        items.append((feed_id, metadata))

    # print(f"Fetched {len(items)} fallback recommendations from `{fallback_table}`.")
    return items
