from google.api_core.exceptions import BadRequest
from google.cloud import bigquery
from typing import Any


# ---------------------------------------------------------------------------------------------
# load data of the given sql from bigQuery 
# ---------------------------------------------------------------------------------------------
def query_sql(
    query: str,
    *,
    query_parameters: list[bigquery.ScalarQueryParameter] | None = None,
    client: bigquery.Client | None = None,
) -> list[dict[str, Any]]:
    """
    Execute SQL in BigQuery and return rows as dictionaries.
    """
    try:
        bq_client = client or bigquery.Client()
        job_config = None
        if query_parameters:
            job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
        result = bq_client.query(query=query, job_config=job_config).result()
        return [dict(row.items()) for row in result]
    except BadRequest as exc:
        raise ValueError(str(exc)) from exc
