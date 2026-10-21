"""Check method return values and final submission fields.

These helpers check formats, not retrieval relevance or judgment accuracy.
An error stops the run before malformed results are saved.
The ranked-document and judgment checks return new records with standard numeric and reason fields.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from common.config import NON_RELEVANT, SUPPORTS_K


def validate_distribution(
    distribution: Any,
    options: list[str],
    *,
    label: str,
) -> dict[str, float]:
    """Check exact option keys and probabilities, then return float values."""
    if not isinstance(distribution, dict):
        raise ValueError(f"{label}: distribution must be a JSON object")
    if set(distribution) != set(options):
        raise ValueError(f"{label}: distribution keys must exactly match the options")

    # Check values before summing, so a missing or invalid probability is identified.
    result: dict[str, float] = {}
    for option in options:
        value = distribution[option]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}: probability for '{option}' must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(
                f"{label}: probability for '{option}' must be finite and non-negative"
            )
        result[option] = number

    # Allow the published rounding tolerance while keeping your probability values unchanged.
    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"{label}: probabilities must sum to 1.0 (got {total:.8f})"
        )
    return result


def validate_supports(
    supports: Any,
    *,
    expected_count: int,
    label: str,
    known_ids: set[str] | None = None,
) -> list[str]:
    """Check the number, uniqueness and corpus membership of document ID strings."""
    if not isinstance(supports, list):
        raise ValueError(f"{label}: supports must be a JSON list")
    if any(not isinstance(value, str) or not value for value in supports):
        raise ValueError(f"{label}: support IDs must be non-empty strings")
    values = supports
    if len(values) != len(set(values)):
        raise ValueError(f"{label}: support IDs must be unique")
    if len(values) != expected_count:
        raise ValueError(
            f"{label}: supports must contain exactly {expected_count} IDs"
        )
    # For retrieval, known_ids is the corpus. For judging, it is the retrieved set.
    if known_ids is not None:
        unknown = [value for value in values if value not in known_ids]
        if unknown:
            raise ValueError(f"{label}: unknown support ID: {unknown[0]}")
    return values


def validate_ranked_documents(
    ranked_documents: Any, known_ids: set[str]
) -> list[dict[str, Any]]:
    """Check a ranker's 100 {doc_id, score} records and convert scores to float.

    Return new records containing only doc_id and score, in the supplied order.
    The retrieval runner attaches document fields to these records for judging.
    Converting to float preserves the score scale.
    Extra input fields are omitted.
    """
    if not isinstance(ranked_documents, list) or not all(
        isinstance(row, dict) for row in ranked_documents
    ):
        raise ValueError("Retriever must return a list of document dictionaries")
    # Check IDs together so duplicates and a short result list are caught as well.
    validate_supports(
        [row.get("doc_id") for row in ranked_documents],
        expected_count=SUPPORTS_K,
        label="Retriever",
        known_ids=known_ids,
    )
    validated = []
    previous = math.inf
    for row in ranked_documents:
        score = row.get("score")
        # bool is numeric in Python, but True/False are not retrieval scores.
        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(score)
        ):
            raise ValueError(f"Retriever score must be finite and numeric: {row['doc_id']}")
        score = float(score)
        # Equal scores are allowed. The small tolerance ignores floating-point noise.
        if score > previous + 1e-12:
            raise ValueError("Retriever scores must be in descending order")
        validated.append({"doc_id": row["doc_id"], "score": score})
        previous = score
    return validated


def validate_judgments(
    judgments: Any,
    ranked_documents: list[dict[str, Any]],
    options: list[str],
    expected_count: int,
) -> list[dict[str, Any]]:
    """Check selected IDs and exact option strings, adding an empty reason if omitted.

    Return new records containing doc_id, option and reason, in the supplied order.
    The judging runner saves these records and passes them to aggregation.
    Extra input fields are omitted.
    """
    if not isinstance(judgments, list) or not all(
        isinstance(row, dict) for row in judgments
    ):
        raise ValueError("Judge must return a list of judgment dictionaries")
    # Each selected document must come from the ranked supports and appear once.
    validate_supports(
        [row.get("doc_id") for row in judgments],
        expected_count=expected_count,
        label="Judge",
        known_ids={row["doc_id"] for row in ranked_documents},
    )
    validated = []
    for row in judgments:
        option = row.get("option")
        if option != NON_RELEVANT and option not in options:
            raise ValueError(f"Judgment option must be an exact option or {NON_RELEVANT!r}: {option}")
        # Reasons are optional. An empty string keeps the saved record format consistent.
        validated.append({
            "doc_id": row["doc_id"],
            "option": option,
            "reason": str(row.get("reason", "")),
        })
    return validated
