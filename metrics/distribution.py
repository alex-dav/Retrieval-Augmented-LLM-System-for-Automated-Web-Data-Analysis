"""Distribution metric used for Assignment 2 evaluation.

See Appendix Section 4 for the detailed metric formulas.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


EPS = 1e-6  # Added to probabilities before logarithms.
GOLD_TIE_TOLERANCE = 0.02  # Two percentage points in the reference probabilities.
FLOAT_TOLERANCE = 1e-12
PREDICTED_TIE_TOLERANCE = 1e-15
DISTRIBUTION_WEIGHT = 2.0
SCORE_SMOOTHING = 0.1  # Used in the final score formula, after the metric components are calculated.


def _smooth(values: np.ndarray) -> np.ndarray:
    """Add EPS to a one-dimensional probability array and return values summing to one."""
    smoothed = values + EPS
    return smoothed / smoothed.sum()


def js_divergence(gold: np.ndarray, predicted: np.ndarray) -> float:
    """Return base-2 Jensen-Shannon divergence for two positive probability arrays.

    Both arrays must use the same option order and sum to one.
    """
    midpoint = 0.5 * (gold + predicted)
    gold_kl = np.sum(gold * np.log2(gold / midpoint))
    predicted_kl = np.sum(predicted * np.log2(predicted / midpoint))
    # Roundoff can make the divergence slightly negative for almost equal inputs.
    return max(0.0, float(0.5 * (gold_kl + predicted_kl)))


def soft_rank_recovery(
    gold: np.ndarray,
    predicted: np.ndarray,
) -> float | None:
    """Measure recovery of reference option gaps, including direction and magnitude.

    Both arrays have shape (number_of_options,) and use the same option order.
    Return a value from -1 to 1, or None when all reference gaps are within the tie tolerance.
    question_score clips negative recovery to zero before combining it with distribution similarity.
    """
    recovered = 0.0
    available = 0.0
    # Visit each pair of answer options once. left and right index the same options in both arrays.
    for left in range(len(gold)):
        for right in range(left + 1, len(gold)):
            gold_gap = float(gold[left] - gold[right])
            gold_magnitude = abs(gold_gap)
            # Only reference gaps above the tolerance contribute to either total.
            if gold_magnitude <= GOLD_TIE_TOLERANCE + FLOAT_TOLERANCE:
                continue

            predicted_gap = float(predicted[left] - predicted[right])
            predicted_magnitude = abs(predicted_gap)
            available += gold_magnitude
            # A predicted tie earns no recovery, while the reference gap remains in the denominator.
            if predicted_magnitude <= PREDICTED_TIE_TOLERANCE:
                continue
            direction = 1.0 if gold_gap * predicted_gap > 0.0 else -1.0
            # Matching directions add credit. Reversed directions subtract it.
            # The contribution is capped at the reference gap.
            recovered += direction * min(gold_magnitude, predicted_magnitude)

    if available <= 0.0:
        return None
    return float(np.clip(recovered / available, -1.0, 1.0))


def question_score(
    gold_values: Sequence[float],
    predicted_values: Sequence[float],
) -> dict[str, float | None]:
    """Score two non-negative sequences whose values use the same answer-option order.

    Each sequence is normalised to sum to one before scoring.
    Return distribution_similarity, signed_rank_recovery, rank_skill, and score.
    The two rank values are None when no reference option gap exceeds the tie tolerance.
    """
    gold = np.asarray(gold_values, dtype=np.float64)
    predicted = np.asarray(predicted_values, dtype=np.float64)
    if gold.ndim != 1 or predicted.ndim != 1 or gold.shape != predicted.shape:
        raise ValueError("Gold and predicted distributions must have equal lengths")
    if len(gold) < 2 or not np.all(np.isfinite(gold)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Distributions must contain finite values for at least two options")
    if np.any(gold < 0.0) or np.any(predicted < 0.0):
        raise ValueError("Distribution values must be non-negative")
    if gold.sum() <= 0.0 or predicted.sum() <= 0.0:
        raise ValueError("Distributions must have positive mass")

    # Normalise both distributions and smooth them before taking logarithms.
    gold = gold / gold.sum()
    predicted = predicted / predicted.sum()
    divergence = js_divergence(_smooth(gold), _smooth(predicted))
    distribution_similarity = 1.0 - math.sqrt(divergence)
    signed_recovery = soft_rank_recovery(gold, predicted)
    # Use distribution similarity alone when the reference has no informative option gaps.
    if signed_recovery is None:
        rank_skill = None
        score = distribution_similarity
    else:
        rank_skill = max(0.0, signed_recovery)
        # Smoothing preserves a small similarity credit when rank recovery is zero.
        denominator = (
            DISTRIBUTION_WEIGHT * rank_skill + distribution_similarity + SCORE_SMOOTHING
        )
        score = (
            distribution_similarity
            * ((DISTRIBUTION_WEIGHT + 1.0) * rank_skill + SCORE_SMOOTHING)
            / denominator
        )
    return {
        "distribution_similarity": float(np.clip(distribution_similarity, 0.0, 1.0)),
        "signed_rank_recovery": signed_recovery,
        "rank_skill": rank_skill,
        "score": float(np.clip(score, 0.0, 1.0)),
    }
