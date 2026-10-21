"""Read and write question files, intermediate JSONL, and final CSV results."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated option names that json.loads would otherwise overwrite."""
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file into Python objects."""
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    """Save a Python value as UTF-8 JSON, creating its parent folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return one dictionary per non-empty line of a UTF-8 JSONL file.

    All records remain in the returned list in memory.
    The judging stage uses this for retrieved question records.
    index.build reads the full corpus one line at a time.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Save each dictionary on one line, creating the output folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Return [{question: str, options: list[str]}, ...] in file order.

    The full file groups questions under dev and test.
    The mini file contains question entries directly.
    Both become the same list for the runner.
    """
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Question file must contain a JSON object: {path}")

    groups: list[dict[str, Any]]
    # Remove only the split grouping. Keep each question and its option order.
    if "dev" in value or "test" in value:
        groups = []
        for split in ("dev", "test"):
            group = value.get(split, {})
            if not isinstance(group, dict):
                raise ValueError(f"Question split '{split}' must be an object")
            groups.append(group)
    else:
        groups = [value]

    questions: list[dict[str, Any]] = []
    for group in groups:
        for question, payload in group.items():
            if not isinstance(payload, dict) or not isinstance(
                payload.get("distribution"), dict
            ):
                raise ValueError(f"Question has no distribution object: {question}")
            # The supplied distribution keys name the options. Its zero values are not labels.
            questions.append(
                {
                    "question": str(question),
                    "options": [str(option) for option in payload["distribution"]],
                }
            )
    return questions


def read_submission(path: Path) -> list[dict[str, Any]]:
    """Return CSV rows with distribution decoded to a dict and supports to a list.

    This reverses write_submission.
    CSV quoting and JSON decoding are handled separately so commas inside option strings do not split a row into extra cells.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["question", "distribution", "supports"]:
            raise ValueError(
                "Submission columns must be: question, distribution, supports"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Submission row {line_number} must contain three cells")
            # First read the CSV cells, then decode the JSON stored inside two cells.
            try:
                distribution = json.loads(
                    row["distribution"],
                    object_pairs_hook=_reject_duplicate_object_keys,
                )
                supports = json.loads(row["supports"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid JSON in submission row {line_number}"
                ) from error
            rows.append(
                {
                    "question": str(row["question"]),
                    "distribution": distribution,
                    "supports": supports,
                }
            )
    return rows


def write_submission(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save question/distribution/supports rows using Kaggle's CSV quoting.

    distribution is a Python dict and supports is a list before this call.
    Both become JSON strings inside the CSV.
    newline="" prevents extra blank lines on Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question", "distribution", "supports"])
        writer.writeheader()
        for row in rows:
            # json.dumps makes each structured value one string. DictWriter quotes the cell.
            question = row["question"]
            distribution_json = json.dumps(row["distribution"], ensure_ascii=False, separators=(",", ":"))
            supports_json = json.dumps(row["supports"], separators=(",", ":"))
            csv_row = {
                "question": question,
                "distribution": distribution_json,
                "supports": supports_json,
            }
            writer.writerow(csv_row)


def print_status(message: str, *, flush: bool = False) -> None:
    """Print a message, optionally flushing immediately instead of buffering it.

    Some Windows terminals cannot display every Unicode character.
    Escape only unsupported characters in the log.
    Saved UTF-8 data and replies are unchanged.
    """
    try:
        print(message, flush=flush)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(message.encode(encoding, errors="backslashreplace").decode(encoding), flush=flush)
