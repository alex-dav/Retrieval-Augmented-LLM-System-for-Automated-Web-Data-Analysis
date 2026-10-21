"""B0 retrieves Boolean OR matches and gives every returned document score 1.0."""

from __future__ import annotations

from typing import Any

from whoosh.qparser import SimpleParser

from index.access import LexicalIndex
from index.text_processing import tokenize


def b0_retrieve(
    question: str,
    options: list[str],
    index: LexicalIndex,
    supports_k: int,
) -> list[dict[str, Any]]:
    """Return question/option matches in document-ID order with equal scores.

    question and options supply query text. index is an open LexicalIndex.
    supports_k is the number of unique {doc_id, score} records to return.
    """
    # Use the index's token rules and remove duplicate query terms.
    query_text = " ".join([question, *options])
    query_tokens = tokenize(query_text)
    terms = sorted(set(query_tokens))
    matched: list[str] = []
    if terms:
        # SimpleParser treats the space-separated terms as an OR-style query.
        parser = SimpleParser("text", schema=index.raw_index.schema)
        query = parser.parse(" ".join(terms))
        with index.searcher() as searcher:
            # An OR match is enough for B0. Document-ID order is not relevance ranking.
            results = searcher.search(query, limit=None, sortedby="doc_id")
            matched = [str(hit["doc_id"]) for hit in results]

    # B0 orders matches by ID. Unmatched IDs fill any remaining places.
    target = min(int(supports_k), len(index))
    selected = matched[:target]
    if len(selected) < target:
        seen = set(selected)
        selected.extend(
            doc_id for doc_id in index.all_document_ids() if doc_id not in seen
        )
        selected = selected[:target]
    # The runner keeps this list order. Every B0 record has the same score.
    return [{"doc_id": doc_id, "score": 1.0} for doc_id in selected]
