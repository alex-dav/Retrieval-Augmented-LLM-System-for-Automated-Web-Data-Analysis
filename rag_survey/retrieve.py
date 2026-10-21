"""Stage 2: retrieve supports, attach document text, and save retrieved.jsonl."""

from __future__ import annotations

import argparse
from typing import Any

from common.config import config_path, load_config
from common.io import load_questions, print_status, write_jsonl
from common.validation import validate_ranked_documents
from index.access import LexicalIndex
from retrieval.rankers import retrieve_and_rank


def run_retrieval(config: dict[str, Any], api_key: str | None) -> list[dict[str, Any]]:
    """Run the configured ranker for every question and return the saved rows.

    config contains parsed INI settings. api_key is passed to the ranker.
    Each returned row contains question, options and ranked_documents.
    """
    # Open resources once, then reuse them for every question.
    index = LexicalIndex(config_path(config, "index"))
    questions = load_questions(config_path(config, "questions"))
    known_ids = set(index.all_document_ids())
    output_rows: list[dict[str, Any]] = []

    for position, item in enumerate(questions, start=1):
        question = item["question"]
        options = item["options"]
        # 1. Run the method selected by [retrieval] method in the INI file.
        raw_ranked_documents = retrieve_and_rank(
            question,
            options,
            index,
            config,
            api_key,
        )
        # 2. Check the returned IDs and scores before looking up document text.
        ranked_documents = validate_ranked_documents(
            raw_ranked_documents,
            known_ids,
        )
        # 3. Attach text and metadata in the same order, preserving the ranking.
        enriched = _attach_document_fields(index, ranked_documents)
        output_rows.append(
            {
                "question": question,
                "options": options,
                "ranked_documents": enriched,
            }
        )
        print_status(
            f"Retrieved and ranked {len(enriched)} supports for "
            f"question {position}/{len(questions)}"
        )

    # 4. Save one question per line for the judging command to read later.
    output_path = config_path(config, "run_dir") / "retrieved.jsonl"
    write_jsonl(output_path, output_rows)
    print_status(f"Wrote {output_path}")
    return output_rows


def _attach_document_fields(
    index: LexicalIndex,
    ranked_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add 1-based ranks, text and metadata to validated doc_id and score records.

    get_documents returns documents in the requested ID order using one searcher.
    The runner saves these records under ranked_documents for judge_evidence.
    """
    documents = index.get_documents([row["doc_id"] for row in ranked_documents])
    enriched = []
    for rank, (ranked_document, document) in enumerate(
        zip(ranked_documents, documents),
        start=1,
    ):
        enriched.append(
            {
                "rank": rank,
                "doc_id": ranked_document["doc_id"],
                "score": ranked_document["score"],
                "title": document.get("title", ""),
                "text": document.get("text", ""),
                "source": document.get("source", ""),
                "group": document.get("group", ""),
                "url": document.get("url", ""),
            }
        )
    return enriched


def main() -> None:
    """Read command-line arguments and start the retrieval stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--api_key")
    args = parser.parse_args()
    run_retrieval(load_config(args.config), args.api_key)


if __name__ == "__main__":
    main()
