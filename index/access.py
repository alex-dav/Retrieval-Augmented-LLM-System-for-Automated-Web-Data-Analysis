"""Open the Whoosh index built by index.build and read indexed documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from whoosh import index as whoosh_index
from index.text_processing import document_text


class LexicalIndex:
    """Provide access to the saved Whoosh index and its documents.

    The runner opens this object once and passes it to retrieve_and_rank as index.
    self.raw_index is a Whoosh index, like ix in the Canvas example.
    schema and searcher(...) access that underlying index.
    get_documents(ids) returns document text and metadata in the requested ID order.
    """

    def __init__(self, index_path: Path | str):
        """Open the existing index at index_path without rebuilding it."""
        self.path = Path(index_path)
        if not whoosh_index.exists_in(self.path):
            raise FileNotFoundError(
                f"Whoosh index not found at {self.path}. Run python -m index.build first."
            )
        self.raw_index = whoosh_index.open_dir(self.path)
        # Keep the Whoosh field definitions available as index.schema.
        self.schema = self.raw_index.schema
        self._document_ids: list[str] | None = None

    def __len__(self) -> int:
        """Return the number of documents in the opened index."""
        return self.raw_index.doc_count()

    def searcher(self, weighting: Any = None):
        """Return a Whoosh searcher with default or supplied scoring weights.

        Use with index.searcher(...) as searcher to close its files after use.
        This is a Whoosh searcher, with the same search(...) calls as the Canvas example.
        """
        if weighting is None:
            return self.raw_index.searcher()
        return self.raw_index.searcher(weighting=weighting)

    def all_document_ids(self) -> list[str]:
        """Return all indexed IDs in string-sorted order.

        The first call scans stored records and keeps their IDs in memory for later questions.
        This order is separate from the row order in the embedding files.
        """
        if self._document_ids is None:
            # Read IDs once. Later questions can reuse this list without scanning again.
            with self.raw_index.searcher() as searcher:
                self._document_ids = sorted(
                    str(fields["doc_id"])
                    for fields in searcher.all_stored_fields()
                )
        return list(self._document_ids)  # Return a copy so callers cannot change the saved list.

    def get_document(self, doc_id: str) -> dict[str, Any]:
        """Return stored fields for one ID, with id, title, content and combined text.

        This accessor renames stored doc_id to id to match the source corpus records.
        The retrieval runner keeps doc_id from the ranked record and attaches this document's text and metadata.
        """
        with self.raw_index.searcher() as searcher:
            fields = searcher.document(doc_id=str(doc_id))
        if fields is None:
            raise KeyError(f"Unknown document ID: {doc_id}")
        document = dict(fields)
        document["id"] = str(document.pop("doc_id"))
        document["text"] = document_text(document)
        return document

    def get_documents(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        """Return dictionaries in doc_ids order, with the same fields as get_document.

        This opens one searcher for the whole list. An unknown ID raises KeyError.
        """
        documents: list[dict[str, Any]] = []
        with self.raw_index.searcher() as searcher:
            for doc_id in doc_ids:
                fields = searcher.document(doc_id=str(doc_id))
                if fields is None:
                    raise KeyError(f"Unknown document ID: {doc_id}")
                document = dict(fields)
                document["id"] = str(document.pop("doc_id"))
                # The schema stores title and content separately. Rejoin them for judging.
                document["text"] = document_text(document)
                documents.append(document)
        return documents
