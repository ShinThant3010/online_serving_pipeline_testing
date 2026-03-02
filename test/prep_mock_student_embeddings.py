#!/usr/bin/env python3
"""Generate per-query embedding .npy files for each mock student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

MAX_EMBEDDING_INSTANCES_PER_REQUEST = 250


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.ndim != 2:
        raise ValueError("l2_normalize expects a 2D array")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def embed_texts(
    *,
    project_id: str,
    region: str,
    embedding_model: str,
    output_dimensionality: int,
    texts: list[str],
    task_type: str,
) -> np.ndarray:
    if not texts:
        raise ValueError("texts must not be empty")

    vertexai.init(project=project_id, location=region)
    model = TextEmbeddingModel.from_pretrained(embedding_model)

    embeddings = []
    for offset in range(0, len(texts), MAX_EMBEDDING_INSTANCES_PER_REQUEST):
        batch = texts[offset : offset + MAX_EMBEDDING_INSTANCES_PER_REQUEST]
        inputs = [TextEmbeddingInput(text=text, task_type=task_type) for text in batch]
        embeddings.extend(
            model.get_embeddings(inputs, output_dimensionality=output_dimensionality)
        )

    return l2_normalize(
        np.asarray([embedding.values for embedding in embeddings], dtype=np.float32)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed each student's 5 hyde query_text values and save .npy files."
    )
    parser.add_argument("--project-id", required=True, default="poc-piloturl-nonprod")
    parser.add_argument("--region", required=True, default="asia-southeast1")
    parser.add_argument(
        "--embedding-model",
        default="gemini-embedding-001",
    )
    parser.add_argument(
        "--output-dimensionality",
        type=int,
        default=768,
        help="Expected output embedding dimensionality",
    )
    parser.add_argument(
        "--task-type",
        default="RETRIEVAL_QUERY",
        help="Vertex embedding task type",
    )
    parser.add_argument(
        "--student-data-dir",
        type=Path,
        default=Path("test/student_data"),
        help="Directory containing per-student folders",
    )
    parser.add_argument(
        "--limit-students",
        type=int,
        default=0,
        help="If > 0, process only first N students (sorted by folder name)",
    )
    return parser.parse_args()


def load_student_query_texts(student_data_dir: Path) -> tuple[list[tuple[str, int]], list[str]]:
    if not student_data_dir.exists():
        raise FileNotFoundError(f"Student data directory not found: {student_data_dir}")

    index_to_student_query: list[tuple[str, int]] = []
    texts: list[str] = []

    for student_dir in sorted(student_data_dir.iterdir()):
        print(f"Processing student directory: {student_dir.name}...")
        if not student_dir.is_dir():
            continue

        hyde_path = student_dir / "hyde" / "hyde.json"
        if not hyde_path.exists():
            continue

        with hyde_path.open("r", encoding="utf-8") as f:
            hyde_payload = json.load(f)

        hq = hyde_payload.get("hq", [])
        if not isinstance(hq, list):
            raise ValueError(f"Invalid hq format in {hyde_path}")
        if len(hq) != 5:
            raise ValueError(
                f"Expected exactly 5 hq queries in {hyde_path}, found {len(hq)}"
            )

        for idx, item in enumerate(hq, start=1):
            query_text = item.get("query_text")
            if not isinstance(query_text, str) or not query_text.strip():
                raise ValueError(
                    f"Missing or empty query_text in {hyde_path} at hq index {idx}"
                )
            index_to_student_query.append((student_dir.name, idx))
            texts.append(query_text)

    if not texts:
        raise ValueError("No query_text values found under student_data.")

    print(f"Loaded {len(texts)} query_text values from {len(index_to_student_query)} student-query pairs.")
    return index_to_student_query, texts


def main() -> None:
    args = parse_args()

    student_data_dir = args.student_data_dir
    student_dirs = sorted([d for d in student_data_dir.iterdir() if d.is_dir()])
    if args.limit_students > 0:
        limited_set = {d.name for d in student_dirs[: args.limit_students]}
    else:
        limited_set = None

    print(f"Loading query_text values from {student_data_dir}...")
    index_to_student_query, texts = load_student_query_texts(student_data_dir)
    if limited_set is not None:
        filtered_pairs: list[tuple[str, int]] = []
        filtered_texts: list[str] = []
        for pair, text in zip(index_to_student_query, texts):
            if pair[0] in limited_set:
                filtered_pairs.append(pair)
                filtered_texts.append(text)
        index_to_student_query, texts = filtered_pairs, filtered_texts

    print(f"Embedding {len(texts)} query_text values using model {args.embedding_model}...")
    embeddings = embed_texts(
        project_id=args.project_id,
        region=args.region,
        embedding_model=args.embedding_model,
        output_dimensionality=args.output_dimensionality,
        texts=texts,
        task_type=args.task_type,
    )

    if embeddings.shape[1] != args.output_dimensionality:
        raise ValueError(
            f"Expected embedding dimensionality {args.output_dimensionality}, "
            f"got {embeddings.shape[1]}"
        )

    print(f"Saving embeddings to {student_data_dir}/<student_id>/embedding/...")
    for emb, (student_id, query_idx) in zip(embeddings, index_to_student_query):
        embedding_dir = student_data_dir / student_id / "embedding"
        embedding_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"embedding{query_idx:02d}.npy"
        np.save(embedding_dir / output_name, emb)

    unique_students = {sid for sid, _ in index_to_student_query}
    print(
        f"Done. Saved {len(index_to_student_query)} embeddings for "
        f"{len(unique_students)} student(s) to {student_data_dir}/<student_id>/embedding/"
    )


if __name__ == "__main__":
    main()
