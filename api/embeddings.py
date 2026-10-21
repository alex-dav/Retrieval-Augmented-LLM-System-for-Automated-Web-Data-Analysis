"""Encode text through the course embedding API."""

from __future__ import annotations

import numpy as np
from openai import OpenAI


def embed_texts(
    texts: list[str],
    *,
    api_key: str | None,
    base_url: str,
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    """Return a float32 matrix with one embedding row per input string.

    Args:
        texts: Strings to encode, such as query texts.
        api_key: Course key supplied with --api_key.
        base_url: Course gateway URL from config["api"]["base_url"].
        model: Use the document embedding model for comparable query vectors.

    Returns:
        An array of shape (len(texts), embedding_dimension), in input order.

    Each call sends all supplied texts in one request and uses the API budget.
    Query texts are encoded during retrieval.
    """
    if not api_key:
        raise ValueError("Dense encoding requires --api_key")
    response = OpenAI(api_key=api_key, base_url=base_url).embeddings.create(
        model=model,
        input=texts,
    )
    # The API index identifies the input string. Restore that order before making the matrix.
    if any(type(item.index) is not int for item in response.data):
        raise ValueError("Embedding response indices must be integers")
    ordered_responses = sorted(response.data, key=lambda item: item.index)
    indices = [item.index for item in ordered_responses]
    # Every input needs exactly one row before vector positions can be used as document positions.
    if indices != list(range(len(texts))):
        raise ValueError("Embedding response indices must cover every input text exactly once")
    vectors = [item.embedding for item in ordered_responses]
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Embedding response must contain numeric vectors of equal length") from error
    # A matrix row must contain one non-empty vector with the same dimension as the other rows.
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Embedding response must form a two-dimensional matrix with non-empty vectors")
    return matrix
