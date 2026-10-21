"""Stage 3: judge retrieved evidence, aggregate votes, and write submission.csv."""

from __future__ import annotations

import argparse
from typing import Any

from common.config import NON_RELEVANT, config_path, load_config, judging_count, SUPPORTS_K
from common.io import print_status, read_jsonl, write_jsonl, write_submission
from common.validation import validate_judgments, validate_supports
from judging.aggregate import aggregate_distribution
from judging.judges import judge_evidence


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _reorder_enabled(config: dict[str, Any]) -> bool:
    """Return True when judging.reorder_supports asks for relevance-first supports."""
    return (
        str(config["judging"].get("reorder_supports", "false")).strip().lower()
        in _TRUE_VALUES
    )


def _reorder_by_relevance(
    supports: list[str],
    judgments: list[dict[str, Any]],
) -> list[str]:
    """Return supports with judged-relevant documents first, preserving order.

    Documents the judge labelled non_relevant, plus any support without a
    judgment, keep their relative order but follow every judged-relevant one.
    With judge_k = 100 every support has a judgment, so this is a stable
    partition of the full list.
    """
    relevant = {
        str(record["doc_id"])
        for record in judgments
        if str(record["option"]) != NON_RELEVANT
    }
    return [d for d in supports if d in relevant] + [
        d for d in supports if d not in relevant
    ]


def run_judging(config: dict[str, Any], api_key: str | None) -> list[dict[str, Any]]:
    """Read retrieval rows, call the judge, and return the final CSV rows.

    config contains parsed INI settings. api_key is passed to the judge.
    Each final row contains question, distribution and supports.
    """
    # Read the previous stage's saved records. Retrieval is not repeated here.
    run_dir = config_path(config, "run_dir")
    retrieved_rows = read_jsonl(run_dir / "retrieved.jsonl")
    judge_k = judging_count(config)
    judgment_rows: list[dict[str, Any]] = []
    submission_rows: list[dict[str, Any]] = []

    for position, row in enumerate(retrieved_rows, start=1):
        question = "[missing question]"
        try:
            question = str(row["question"])
            options = [str(option) for option in row["options"]]
            ranked_documents = row["ranked_documents"]
            # A separately run stage may be reading an edited or incomplete file.
            if not isinstance(ranked_documents, list) or not all(
                isinstance(document, dict) for document in ranked_documents
            ):
                raise ValueError("retrieved.jsonl ranked_documents must be a list of document dictionaries")
            ranked_documents = list(ranked_documents)
            validate_supports(
                [document["doc_id"] for document in ranked_documents],
                expected_count=SUPPORTS_K,
                label="retrieved.jsonl",
            )
            # 1. Let the selected method choose documents and judge their options.
            raw_judgments = judge_evidence(
                question,
                options,
                ranked_documents,
                config,
                api_key,
            )
            # 2. Check the returned format, selected IDs and exact option strings.
            judgments = validate_judgments(
                raw_judgments,
                ranked_documents,
                options,
                judge_k,
            )
            # 3. Turn selected judgments into probabilities. Keep all 100 IDs as supports.
            distribution = aggregate_distribution(options, judgments)
        except Exception:
            # Add the question to the error output, then stop with the original exception.
            print_status(f"[judging error] Question {position}/{len(retrieved_rows)}: {question}", flush=True)
            print_status(
                f"[judging error] No new judgments.jsonl or submission.csv was written to {run_dir}. "
                "Any existing files are from an earlier run.",
                flush=True,
            )
            raise
        supports = [str(document["doc_id"]) for document in ranked_documents]
        if _reorder_enabled(config):
            # Evidence-ranking lever: judged-relevant supports keep their
            # retrieval order first; judged non_relevant (and any unjudged)
            # supports follow in their original order. The distribution is
            # unaffected because aggregation uses the judgments, not this list.
            supports = _reorder_by_relevance(supports, judgments)
        judgment_rows.append(
            {
                "question": question,
                "options": options,
                "judgments": judgments,
            }
        )
        submission_rows.append(
            {
                "question": question,
                "distribution": distribution,
                "supports": supports,
            }
        )
        print_status(f"Judged question {position}/{len(retrieved_rows)}")

    # 4. Keep readable judgments for debugging and write the Kaggle CSV.
    # write_submission converts the distribution dict and supports list to JSON cells.
    try:
        write_jsonl(run_dir / "judgments.jsonl", judgment_rows)
        write_submission(run_dir / "submission.csv", submission_rows)
    except OSError:
        print_status(f"[output error] Could not finish saving results in {run_dir}. Output files may be incomplete.", flush=True)
        raise
    print_status(f"Wrote {run_dir / 'judgments.jsonl'}")
    print_status(f"Wrote {run_dir / 'submission.csv'}")
    return submission_rows


def main() -> None:
    """Read command-line arguments and start the judging stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--api_key")
    args = parser.parse_args()
    run_judging(load_config(args.config), args.api_key)


if __name__ == "__main__":
    main()
