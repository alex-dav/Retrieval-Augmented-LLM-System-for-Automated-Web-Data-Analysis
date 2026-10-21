"""Supplied aggregation from document judgments to option probabilities."""

from __future__ import annotations

from typing import Any

from common.config import NON_RELEVANT


def aggregate_distribution(
    options: list[str],
    judgments: list[dict[str, Any]],
) -> dict[str, float]:
    """Convert document judgments into probabilities in the original option order.

    options contains the exact answer strings.
    Each judgment has an option field containing one of those strings or non_relevant.
    Count one vote per relevant judgment and divide by the vote total.
    If every judgment is non_relevant, return a uniform distribution.
    The returned dict maps each option to a float.
    """
    # Initialise every option, including ones that may receive no votes.
    counts = {option: 0 for option in options}
    for judgment in judgments:
        option = judgment["option"]
        if option == NON_RELEVANT:
            # Skip non_relevant judgments when counting votes and the vote total.
            continue
        counts[option] += 1

    total = sum(counts.values())
    if total == 0:
        # With no option votes, use equal probabilities and avoid division by zero.
        uniform = 1.0 / len(options)
        return {option: uniform for option in options}
    # One judgment, one vote. For counts 3 and 1, probabilities are 0.75 and 0.25.
    return {option: counts[option] / total for option in options}
