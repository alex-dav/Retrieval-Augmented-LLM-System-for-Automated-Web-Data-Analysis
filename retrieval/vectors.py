"""Load the document-vector matrix and its aligned IDs for retrieval."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from common.config import config_path


def load_document_embeddings(
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Return a memory-mapped matrix and its document IDs in the same row order.

    config["paths"]["document_embeddings"] names the .npy matrix.
    config["paths"]["document_ids"] names the JSON ID list.
    The matrix has shape (number_of_documents, embedding_dimension).
    Row i represents ids[i], with row indices starting at zero.
    Keep this ID order when assigning scores to documents. The full supplied matrix is float16.

    Unpack the result as document_vectors, document_ids = load_document_embeddings(config).
    Week 08's load_embedding_store returns IDs first. This helper returns the matrix first.
    """
    # mmap_mode="r" reads matrix slices from disk without loading the full array at once.
    matrix = np.load(config_path(config, "document_embeddings"), mmap_mode="r")
    # The ID list is loaded in full so each matrix row can be matched to its document.
    with config_path(config, "document_ids").open("r", encoding="utf-8") as stream:
        ids = json.load(stream)
    # A row-count mismatch would attach vector scores to the wrong set of document IDs.
    if not isinstance(ids, list) or matrix.ndim != 2 or len(ids) != matrix.shape[0]:
        raise ValueError("The embedding matrix and document IDs are not aligned")
    return matrix, [str(doc_id) for doc_id in ids]


def get_document_batch(
    document_vectors: np.ndarray,
    document_ids: list[str],
    start: int,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Return one aligned float32 batch beginning at start.

    document_vectors has shape (number_of_documents, embedding_dimension).
    document_vectors[i] belongs to document_ids[i].
    start is the first row to read, and batch_size is the maximum number of rows.
    Each vector_batch has shape (batch_rows, embedding_dimension).
    vector_batch[i] represents id_batch[i]. The final batch may have fewer rows.

    Only the requested slice is converted to float32.
    The source matrix remains memory-mapped. Converting values does not change their scale.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("retrieval.batch_size must be a positive integer")
    if type(start) is not int or start < 0:
        raise ValueError("start must be a non-negative integer")
    if document_vectors.ndim != 2 or len(document_ids) != document_vectors.shape[0]:
        raise ValueError("The embedding matrix and document IDs are not aligned")

    end = min(start + batch_size, len(document_ids))
    vector_batch = np.asarray(document_vectors[start:end], dtype=np.float32)
    id_batch = document_ids[start:end]
    return vector_batch, id_batch


def cosine_scores(
    query_vector: np.ndarray,
    vector_batch: np.ndarray,
) -> np.ndarray:
    """Return one cosine-similarity score per document row.

    query_vector has shape (embedding_dimension,).
    vector_batch has shape (batch_rows, embedding_dimension).
    An all-zero query or document vector receives a score of 0.0.
    """
    query = np.asarray(query_vector, dtype=np.float32)
    documents = np.asarray(vector_batch, dtype=np.float32)
    if (
        query.ndim != 1
        or documents.ndim != 2
        or documents.shape[1] != query.shape[0]
    ):
        raise ValueError("Query and document vectors must have matching dimensions")

    query_norm = float(np.linalg.norm(query))
    document_norms = np.linalg.norm(documents, axis=1)
    denominators = document_norms * query_norm
    scores = np.zeros(len(documents), dtype=np.float32)
    np.divide(documents @ query, denominators, out=scores, where=denominators != 0)
    return np.clip(scores, -1.0, 1.0)
