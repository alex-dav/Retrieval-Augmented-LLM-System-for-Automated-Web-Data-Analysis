"""Evaluate a mini submission against the supplied mini ground truth.

See Appendix Section 4 for the detailed metric formulas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.config import SUPPORTS_K
from common.io import print_status, read_jsonl, read_submission
from common.validation import validate_distribution, validate_supports
from metrics.distribution import question_score


def average_precision_at_k(
    predicted: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Score one ranked ID list against its relevant-ID set at cutoff k.

    predicted contains unique document IDs in retrieval order.
    relevant contains all labelled relevant IDs.
    k is a document count. evaluate averages this per-question AP to obtain MAP.
    """
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(predicted[:k], start=1):
        if doc_id in relevant:
            hits += 1
            # Add precision at this relevant document's rank, which starts at one.
            precision_sum += hits / rank
    # Relevant documents outside the returned list still contribute to the denominator, up to k.
    return precision_sum / min(len(relevant), k)


def evaluate(
    submission_path: Path,
    groundtruth_path: Path,
    *,
    documents_path: Path | None = None,
) -> dict[str, Any]:
    """Return mean scores and per-question scores for a saved CSV.

    groundtruth_path supplies option probabilities and relevant document IDs.
    documents_path optionally restricts supports to the supplied corpus IDs.
    """
    known_ids = (
        {str(document["id"]) for document in read_jsonl(documents_path)}
        if documents_path is not None else None
    )
    groundtruth = json.loads(groundtruth_path.read_text(encoding="utf-8-sig"))
    if isinstance(groundtruth, dict) and set(groundtruth).issubset({"dev", "test"}):
        questions = {}
        for split in ("dev", "test"):
            for question, payload in groundtruth.get(split, {}).items():
                questions[question] = payload
        groundtruth = questions
    if not isinstance(groundtruth, dict) or not groundtruth:
        raise ValueError("Ground truth must contain at least one question")

    submission_rows = read_submission(submission_path)
    submission = {row["question"]: row for row in submission_rows}
    if len(submission) != len(submission_rows):
        raise ValueError("Submission contains duplicate question rows")
    if set(submission) != set(groundtruth):
        raise ValueError("Submission questions must exactly match the mini ground truth")

    distribution_scores: list[float] = []
    average_precisions: list[float] = []
    per_question: list[dict[str, Any]] = []
    for question, gold in groundtruth.items():
        row = submission[question]
        options = [str(option) for option in gold["distribution"]]
        predicted_distribution = validate_distribution(
            row["distribution"],
            options,
            label=question,
        )
        supports = validate_supports(
            row["supports"],
            expected_count=SUPPORTS_K,
            label=question,
            known_ids=known_ids,
        )
        gold_values = [float(gold["distribution"][option]) for option in options]
        predicted_values = [predicted_distribution[option] for option in options]
        distribution = question_score(gold_values, predicted_values)
        ap = average_precision_at_k(
            supports,
            {str(doc_id) for doc_id in gold.get("supports", [])},
            SUPPORTS_K,
        )
        distribution_scores.append(float(distribution["score"]))
        average_precisions.append(ap)
        per_question.append(
            {
                "question": question,
                "distribution_score": float(distribution["score"]),
                "map_support_ap": ap,
            }
        )

    return {
        "distribution_score": float(np.mean(distribution_scores)),
        "map_support": float(np.mean(average_precisions)),
        "questions": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument(
        "--groundtruth",
        default="data/mini/mini_groundtruth.json",
    )
    parser.add_argument("--documents", default="data/mini/mini_documents.jsonl")
    arguments = parser.parse_args()
    result = evaluate(
        Path(arguments.submission),
        Path(arguments.groundtruth),
        documents_path=Path(arguments.documents),
    )
    print_status(f"Distribution score: {result['distribution_score']:.6f}")
    print_status(f"MAP@{SUPPORTS_K}: {result['map_support']:.6f}")
    for row in result["questions"]:
        print_status(
            f"- {row['question']}: distribution={row['distribution_score']:.6f}, "
            f"AP={row['map_support_ap']:.6f}"
        )


if __name__ == "__main__":
    main()
