"""Stage 1: build the retrieval resources selected for this run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from whoosh import index as whoosh_index
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import ID, STORED, TEXT, Schema

from common.config import config_path, load_config
from common.io import print_status
from index.text_processing import document_text


def build_resources(config: dict, api_key: str | None = None) -> int:
    """Build the configured Whoosh index and return the number of documents.

    config is the parsed INI dictionary.
    The documents and index paths come from its paths section.
    api_key is available for your own resource setup.
    """
    count = build_index(
        config_path(config, "documents"),
        config_path(config, "index"),
    )
    # Mini and full document vectors are supplied. Dense retrieval opens them when needed.
    return count


def build_index(documents_path: Path, index_path: Path) -> int:
    """Index a UTF-8 document JSONL file and return its document count.

    documents_path contains id, title and content fields.
    index_path is the output directory.
    The Week 08 pattern is Schema -> create_in -> writer -> add_document -> commit.
    This full-data version reads one document at a time.
    """
    # 1. Create the index and a writer, following the Week 08 workshop sequence.
    # Whoosh replaces its own index files and leaves other files intact.
    index_path.mkdir(parents=True, exist_ok=True)
    index = whoosh_index.create_in(index_path, _schema())
    # Other index structures and document text also use memory beyond this pool.
    writer = index.writer(limitmb=256)  # Limit the writer's in-memory posting pool (MB).
    started = time.monotonic()
    try:
        # 2. Read and write one document at a time to limit corpus memory use.
        count = _write_documents(writer, documents_path)
        # 3. Finish writing and merge index segments for subsequent searches.
        # Merging reduces the segments searched later, at the cost of extra build time and disk work.
        writer.commit(optimize=True)
    except Exception:
        # Release the writer if indexing fails, then show the original error.
        writer.cancel()
        raise
    elapsed = time.monotonic() - started
    print_status(f"Built Whoosh index with {count:,} documents in {elapsed:.1f}s")
    print_status(f"Index: {index_path}")
    return count


def _write_documents(writer, documents_path: Path) -> int:
    """Add each non-empty JSONL record to the open writer and return the count.

    Each line becomes one document dictionary. The file is read in order,
    keeping only the current record in memory outside Whoosh's own buffers.
    build_index commits the writer after all records have been added.
    """
    count = 0
    with documents_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            document = json.loads(line)
            fields = _index_fields(document)
            # The dictionary keys match _schema; ** passes them as named arguments.
            writer.add_document(**fields)
            count += 1
            if count % 20000 == 0:
                print_status(f"Indexed {count:,} documents...")
    return count


def _index_fields(document: dict) -> dict[str, str]:
    """Map one corpus record to the fields expected by Whoosh.

    The corpus id becomes doc_id, which links search hits to document vectors
    and judged evidence. document_text joins title and content for searching.
    Original fields are stored separately so judging can choose its input text.
    Missing optional fields become empty strings.
    """
    return {
        "doc_id": str(document["id"]),
        "title": str(document.get("title") or ""),
        "text": document_text(document),
        "content": str(document.get("content") or ""),
        "description": str(document.get("description") or ""),
        "post_content": str(document.get("post_content") or ""),
        "source": str(document.get("source") or ""),
        "group": str(document.get("group") or ""),
        "url": str(document.get("url") or ""),
    }


def _schema() -> Schema:
    """Use searchable text fields and stored metadata needed by later stages.

    B0 uses StandardAnalyzer consistently for indexing, queries and options.
    The Week 08 workshop uses StemmingAnalyzer in its separate BM25 example.
    Search the text field here, which combines title and content.
    The workshop searches its content field.
    """
    return Schema(
        doc_id=ID(stored=True, unique=True, sortable=True),
        title=TEXT(stored=True, analyzer=StandardAnalyzer()),
        text=TEXT(stored=False, analyzer=StandardAnalyzer()),
        content=STORED,
        description=STORED,
        post_content=STORED,
        source=ID(stored=True),
        group=ID(stored=True),
        url=STORED,
    )


def main() -> None:
    """Read command-line arguments and start resource building."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--api_key")
    args = parser.parse_args()
    config = load_config(args.config)
    build_resources(config, args.api_key)


if __name__ == "__main__":
    main()
