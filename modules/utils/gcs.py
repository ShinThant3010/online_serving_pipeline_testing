import json
from io import BytesIO
from datetime import date, datetime, time
from typing import Any

import numpy as np
from google.cloud import storage


# ---------------------------------------------------------------------------------------------
# Helper function - validate gcs path
# ---------------------------------------------------------------------------------------------
def parse_gcs_prefix(prefix: str) -> tuple[str, str]:
    """
    Parse a GCS prefix into bucket and path.
    """
    if not prefix.startswith("gs://"):
        raise ValueError("GCS prefix must start with gs://")
    remainder = prefix[5:]
    bucket, _, path = remainder.partition("/")
    if not bucket or not path:
        raise ValueError("GCS prefix must include bucket and folder path")
    return bucket, path


# ---------------------------------------------------------------------------------------------
# write data to gcs
# ---------------------------------------------------------------------------------------------
def write_to_gcs(
    gcs_prefix: str,
    items: list[dict[str, Any]],
    *,
    filename: str = "part-00000",
    file_type: str = "json",
) -> str:
    """
    Write items to GCS as a single file.
    """

    ### ------------------------- format datatime data to str ------------------------- ###
    def _json_default(value: Any) -> Any:
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        return str(value)

    ### ------------------------ prep target folder path & data ------------------------ ###
    bucket_name, path = parse_gcs_prefix(gcs_prefix)

    clean_filename = filename.strip() or "part-00000"
    clean_file_type = file_type.strip().lstrip(".") or "json"
    blob_name = f"{path.rstrip('/')}/{clean_filename}.{clean_file_type}"
    payload = (
        "\n".join(json.dumps(item, ensure_ascii=True, default=_json_default) for item in items)
        + "\n"
    )

    ### -------------------------------- load json data -------------------------------- ###
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_string(payload, content_type="application/json")

    return f"gs://{bucket_name}/{blob_name}"


# ---------------------------------------------------------------------------------------------
# General function to load data from gcs; used in modules/functions/hyde_embedding.py
# ---------------------------------------------------------------------------------------------
def load_data_from_gcs_prefix(
    gcs_prefix: str,
    file_type: str = "json",
) -> list[Any]:
    """
    Load data items from GCS prefix. 
    Supports json, txt, and npy file types. 
    """
    supported = {"json", "txt", "npy"}
    target_type = file_type.strip().lstrip(".").lower() or "json"

    # Validate file type
    if target_type not in supported:
        raise ValueError(f"Unsupported file_type `{file_type}`. Supported: json, txt, npy")

    # Parse GCS prefix
    bucket_name, prefix = parse_gcs_prefix(gcs_prefix)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Load items
    items: list[Any] = []
    for blob in bucket.list_blobs(prefix=prefix.rstrip("/") + "/"):
        if blob.name.endswith("/"):
            continue
        ext = blob.name.rsplit(".", 1)[-1].lower() if "." in blob.name else ""
        if ext != target_type:
            continue

        ### -------------------------------- load json data -------------------------------- ###
        if target_type == "json":
            content = blob.download_as_text()
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    # Fall back to whole-file JSON parse if not jsonl.
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        items.extend(parsed)
                    else:
                        items.append(parsed)
                    break
            continue

        ### --------------------------------- load txt data --------------------------------- ###
        if target_type == "txt":
            content = blob.download_as_text()
            for line in content.splitlines():
                line = line.strip()
                if line:
                    items.append(line)
            continue

        ### --------------------------------- load npy data --------------------------------- ###
        if target_type == "npy":
            raw = blob.download_as_bytes()
            array = np.load(BytesIO(raw), allow_pickle=False)
            loaded = array.tolist()

            # Handle both list of vectors and single vector cases.
            if isinstance(loaded, list):
                items.append([float(v) for v in loaded])
                continue

            items.append([float(loaded)])
            continue

    return items


def load_json_from_gcs_uri(gcs_uri: str) -> Any | None:
    """
    Load a single JSON document from an exact GCS object URI.
    """
    bucket_name, blob_name = parse_gcs_prefix(gcs_uri)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return None

    return json.loads(blob.download_as_text())
