"""Tokenisation and title-plus-content text shared by the index and B0."""

from __future__ import annotations

from whoosh.analysis import StandardAnalyzer


_ANALYZER = StandardAnalyzer()


def tokenize(text: str) -> list[str]:
    """Return tokens using the same StandardAnalyzer as the B0 index.

    Args:
        text: Document, question or option text.

    Returns:
        A list of lowercase tokens after Whoosh's default stopword filtering.
    """
    return [token.text for token in _ANALYZER(text or "")]


def document_text(document: dict[str, object]) -> str:
    """Return title and content joined by a newline, omitting empty fields.

    document is a corpus record with title and content keys.
    The same string is indexed and passed to judge_evidence as the text field.
    The judging method chooses what to include in its model prompts.
    The supplied mini and full document vectors use this same text construction.
    """
    title = str(document.get("title") or "")
    content = str(document.get("content") or "")
    return "\n".join(part for part in (title, content) if part)
