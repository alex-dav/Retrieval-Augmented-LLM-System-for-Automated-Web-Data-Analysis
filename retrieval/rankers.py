"""Suggested starting points for retrieval and ranking methods."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from whoosh.qparser import SimpleParser
from whoosh.scoring import BM25F

from common.config import SUPPORTS_K, config_path
from index.access import LexicalIndex
from index.text_processing import tokenize
from retrieval.baseline import b0_retrieve


def retrieve_and_rank(
    question: str,
    options: list[str],
    index: LexicalIndex,
    config: dict[str, Any],
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return the final ranked supports for one survey question.

    Args:
        question: Exact question text from the configured question file.
        options: Exact answer-option strings in their original order.
        index: Already-open LexicalIndex supplied by the runner.
            It wraps the Whoosh index created by python -m index.build.
        config: Parsed settings from the INI file used for this run.
        api_key: Course API key, or ``None`` when the retrieval method does not need it.

    Returns:
        Exactly ``SUPPORTS_K`` dictionaries.
        Each dictionary must contain a unique corpus ``doc_id`` and a finite numeric ``score``.
        Scores must be in descending order.
        A method may search a larger internal pool before selecting these final supports.

        Example records from a full list of 100:
        [{"doc_id": "22E5PQJSLD", "score": 1.0}, ...]

    The runner checks the count, document IDs, uniqueness, scores, and order.
    You may replace this function or reorganise the internal pipeline.
    """
    # Keep B0 available for experiments. The INI setting selects which branch runs.
    method = str(config["retrieval"]["method"]).strip().lower()

    if method == "b0":
        return b0_retrieve(question, options, index, SUPPORTS_K)

    if method == "bm25":
        return bm25_retrieve(question, options, index, config)

    if method == "dense":
        return dense_retrieve(question, options, index, config, api_key)

    if method == "fusion":
        return fusion_retrieve(question, options, index, config, api_key)

    # If you add a method name in the config, implement its branch above too.
    raise ValueError(f"Unknown retrieval method: {method}")


def bm25_retrieve(
    question: str,
    options: list[str],
    index: LexicalIndex,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the highest BM25F-scoring supports for one survey question.

    question and options supply the query text, index is an open LexicalIndex,
    and config supplies retrieval.k1 and retrieval.b for the BM25F weighting.
    Documents are ranked over the searchable "text" field, which holds title and content.
    """
    records = bm25_rank(question, options, index, config, SUPPORTS_K)
    # A query may match fewer than SUPPORTS_K documents, so unmatched IDs fill the rest.
    return _fill_supports(records, index, SUPPORTS_K)


def bm25_rank(
    question: str,
    options: list[str],
    index: LexicalIndex,
    config: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` unpadded BM25F records in descending score order.

    This is the raw ranking used by bm25_retrieve and by the fusion method.
    No filling is applied, so the list may be shorter than the final support count.
    """
    settings = config["retrieval"]
    k1 = _positive_number(settings.get("k1", 1.2), "retrieval.k1")
    b = _bounded_number(settings.get("b", 0.75), "retrieval.b", 0.0, 1.0)

    records: list[dict[str, Any]] = []
    terms = _query_terms(question, options)
    if terms:
        # SimpleParser treats the space-separated terms as an OR-style query.
        parser = SimpleParser("text", schema=index.raw_index.schema)
        query = parser.parse(" ".join(terms))
        with index.searcher(weighting=BM25F(B=b, K1=k1)) as searcher:
            # Whoosh returns hits in descending score order. Keep that order for ties.
            results = searcher.search(query, limit=limit)
            records = [
                {"doc_id": str(hit["doc_id"]), "score": float(hit.score)}
                for hit in results
            ]
        records.sort(key=lambda row: row["score"], reverse=True)
    return records[:limit]


def dense_retrieve(
    question: str,
    options: list[str],
    index: LexicalIndex,
    config: dict[str, Any],
    api_key: str | None,
) -> list[dict[str, Any]]:
    """Return the highest cosine-scoring supports for one survey question.

    The question and options are encoded as one query text with the configured
    embedding model, then compared against the supplied document vectors in batches.
    config supplies retrieval.batch_size and the optional retrieval.dense_pool,
    the number of candidate documents kept while scanning the matrix.
    """
    records = dense_rank(question, options, config, api_key, SUPPORTS_K)
    return _fill_supports(records, index, SUPPORTS_K)


def dense_rank(
    question: str,
    options: list[str],
    config: dict[str, Any],
    api_key: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` unpadded cosine records in descending score order.

    This is the raw ranking used by dense_retrieve and by the fusion method.
    The scan keeps at least ``limit`` candidates, so no candidate above the
    requested depth is dropped while the batches are merged.
    """
    # Dense-only helpers stay local so B0 and BM25 do not load them.
    from api.embeddings import embed_texts
    from retrieval.vectors import cosine_scores, get_document_batch

    settings = config["retrieval"]
    batch_size = int(settings.get("batch_size", 512))
    pool = max(limit, int(settings.get("dense_pool", 1000)))

    query_text = " ".join([question, *options])
    query_vector = embed_texts(
        [query_text],
        api_key=api_key,
        base_url=str(config["api"]["base_url"]),
    )[0]

    document_vectors, document_ids = _embedding_store(config)
    candidate_scores = np.zeros(0, dtype=np.float32)
    candidate_ids: list[str] = []
    for start in range(0, len(document_ids), batch_size):
        vector_batch, id_batch = get_document_batch(
            document_vectors, document_ids, start, batch_size
        )
        batch_scores = cosine_scores(query_vector, vector_batch)
        candidate_scores, candidate_ids = _merge_top_k(
            candidate_scores, candidate_ids, batch_scores, id_batch, pool
        )

    order = np.argsort(-candidate_scores, kind="stable")
    return [
        {"doc_id": candidate_ids[int(position)], "score": float(candidate_scores[int(position)])}
        for position in order[:limit]
    ]


def fusion_retrieve(
    question: str,
    options: list[str],
    index: LexicalIndex,
    config: dict[str, Any],
    api_key: str | None,
) -> list[dict[str, Any]]:
    """Return supports ranked by reciprocal rank fusion of BM25F and cosine.

    Both systems rank their own top ``retrieval.fusion_pool`` documents with the
    same query text. Each document's fused score is the weighted sum of
    ``1 / (retrieval.rrf_k + rank)`` over the systems that returned it, so the
    two score scales never have to be compared directly.
    """
    settings = config["retrieval"]
    rrf_k = _positive_number(settings.get("rrf_k", 60), "retrieval.rrf_k")
    pool = max(
        SUPPORTS_K,
        _positive_int(settings.get("fusion_pool", 1000), "retrieval.fusion_pool"),
    )
    weights = {
        "bm25": _non_negative_number(
            settings.get("bm25_weight", 1.0), "retrieval.bm25_weight"
        ),
        "dense": _non_negative_number(
            settings.get("dense_weight", 1.0), "retrieval.dense_weight"
        ),
    }
    if not any(weights.values()):
        raise ValueError("retrieval.bm25_weight and retrieval.dense_weight cannot both be zero")

    ranked_lists = {
        "bm25": bm25_rank(question, options, index, config, pool),
        "dense": dense_rank(question, options, config, api_key, pool),
    }
    records = _rrf_fuse(ranked_lists, rrf_k, weights)
    return _fill_supports(records, index, SUPPORTS_K)


def _rrf_fuse(
    ranked_lists: dict[str, list[dict[str, Any]]],
    rrf_k: float,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    """Return one record per distinct document, sorted by fused RRF score.

    Ranks start at one. A document missing from a system receives nothing from
    that system. Ties fall back to the total source rank and then the document
    ID, so the same inputs always produce the same ordering.
    """
    fused: dict[str, float] = {}
    total_rank: dict[str, float] = {}
    for name, records in ranked_lists.items():
        weight = weights.get(name, 0.0)
        if weight == 0.0:
            continue
        for rank, record in enumerate(records, start=1):
            doc_id = str(record["doc_id"])
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (rrf_k + rank)
            total_rank[doc_id] = total_rank.get(doc_id, 0.0) + rank

    ordered = sorted(
        fused,
        key=lambda doc_id: (-fused[doc_id], total_rank[doc_id], doc_id),
    )
    return [
        {"doc_id": doc_id, "score": float(fused[doc_id])}
        for doc_id in ordered
    ]


def _query_terms(question: str, options: list[str]) -> list[str]:
    """Return sorted unique tokens for the question and its answer options.

    The index's own token rules are used, and duplicates are removed so a term
    repeated across the question and options is not weighted several times.
    """
    return sorted(set(tokenize(" ".join([question, *options]))))


def _fill_supports(
    records: list[dict[str, Any]],
    index: LexicalIndex,
    target: int,
) -> list[dict[str, Any]]:
    """Return exactly target records in descending order, padding with unseen IDs.

    Padding scores are never above the last real score, so the descending order
    required by the runner still holds. Corpus IDs already selected are skipped.
    """
    selected = records[:target]
    padding_score = min(0.0, selected[-1]["score"]) if selected else 0.0
    if len(selected) < target:
        seen = {row["doc_id"] for row in selected}
        selected.extend(
            {"doc_id": doc_id, "score": padding_score}
            for doc_id in index.all_document_ids()
            if doc_id not in seen
        )
        selected = selected[:target]
    if len(selected) < target:
        raise ValueError(
            f"Retriever found only {len(selected)} of the required {target} documents"
        )
    return selected


def _merge_top_k(
    candidate_scores: np.ndarray,
    candidate_ids: list[str],
    batch_scores: np.ndarray,
    batch_ids: list[str],
    pool: int,
) -> tuple[np.ndarray, list[str]]:
    """Return the best pool candidates after merging one batch of scores.

    The merged list keeps the highest scores so far. Ties keep the earlier
    candidate, which makes the selected documents stable across runs.
    """
    merged_scores = np.concatenate(
        [candidate_scores, np.asarray(batch_scores, dtype=np.float32)]
    )
    merged_ids = [*candidate_ids, *batch_ids]
    if len(merged_ids) > pool:
        keep = np.argsort(-merged_scores, kind="stable")[:pool]
        merged_scores = merged_scores[keep]
        merged_ids = [merged_ids[int(position)] for position in keep]
    return merged_scores, merged_ids


_EMBEDDING_STORES: dict[str, tuple[np.ndarray, list[str]]] = {}


def _embedding_store(
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Return the document-vector matrix and IDs for this config, loaded once.

    Every question of a run reads the same files, so the memory-mapped matrix
    and its aligned ID list are cached by embedding-file path.
    """
    from retrieval.vectors import load_document_embeddings

    key = str(config_path(config, "document_embeddings"))
    store = _EMBEDDING_STORES.get(key)
    if store is None:
        store = load_document_embeddings(config)
        _EMBEDDING_STORES[key] = store
    return store


def _positive_number(value: Any, label: str) -> float:
    """Return a finite number greater than zero, or raise a labelled error."""
    number = _as_number(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be greater than zero")
    return number


def _non_negative_number(value: Any, label: str) -> float:
    """Return a finite number of zero or more, or raise a labelled error."""
    number = _as_number(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be zero or greater")
    return number


def _positive_int(value: Any, label: str) -> int:
    """Return a config value as an integer above zero, or raise a labelled error."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive integer") from error
    if number < 1 or number != float(value):
        raise ValueError(f"{label} must be a positive integer")
    return number


def _bounded_number(value: Any, label: str, low: float, high: float) -> float:
    """Return a finite number inside an inclusive range, or raise a labelled error."""
    number = _as_number(value, label)
    if not low <= number <= high:
        raise ValueError(f"{label} must be a number from {low} to {high}")
    return number


def _as_number(value: Any, label: str) -> float:
    """Convert a config value to a finite float, rejecting other shapes."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number
