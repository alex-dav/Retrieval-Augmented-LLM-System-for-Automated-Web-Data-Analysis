"""B0 assigns options by token overlap, with one judgment per document."""

from __future__ import annotations

from typing import Any

from common.config import NON_RELEVANT
from index.text_processing import tokenize


def b0_judge(
    options: list[str],
    judged_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return {doc_id, option, reason} records for the selected documents.

    options contains the exact answer strings.
    Each judged_documents record contains doc_id and text.
    Positive ties choose the first option.
    Documents with no overlapping option tokens receive non_relevant.
    """
    # Sets count shared words once, even when a word repeats in the document.
    option_terms = [set(tokenize(option)) for option in options]
    judgments: list[dict[str, Any]] = []
    for document in judged_documents:
        document_terms = set(tokenize(str(document["text"])))
        overlaps = [len(terms & document_terms) for terms in option_terms]
        best = max(overlaps, default=0)
        if best > 0:
            # index() selects the first tied option in the original option order.
            option = options[overlaps.index(best)]
        else:
            option = NON_RELEVANT
        judgments.append(
            {
                "doc_id": str(document["doc_id"]),
                "option": option,
                "reason": (
                    "B0 token overlap" if option != NON_RELEVANT else "No overlap"
                ),
            }
        )
    return judgments
