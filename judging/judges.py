"""Suggested starting points for evidence-judging methods."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from common.config import NON_RELEVANT, config_path, judging_count
from common.io import print_status
from api.chat import chat
from judging.baseline import b0_judge


_SYSTEM_PROMPT = (
    "You are a survey research assistant. For each document, decide which answer "
    "option the document's author would choose, based only on evidence in the text. "
    "Use 'non_relevant' when the document does not indicate a choice among the options. "
    "Reply with one JSON object and nothing else, in the form "
    '{"judgments": [{"doc_id": "...", "option": "...", "reason": "..."}]}. '
    "Give every supplied doc_id exactly once, copy the option strings exactly, and "
    "keep each reason to at most eight words quoting or paraphrasing the evidence. "
    "Never guess: when the text does not clearly show the author's own choice, "
    "use non_relevant."
)

_JSON_REMINDER = (
    "\nReply with only the JSON object described above. No prose before or after it."
)

# Few-shot demonstrations are optional. judge_shots = 0 keeps prompts identical
# to the zero-shot runs, so cached zero-shot replies stay reusable.
_DEFAULT_EXAMPLES_FILE = "judging/few_shot_examples.json"
_MAX_SHOTS = 4
_MAX_EXAMPLE_CHARS = 300


def judge_evidence(
    question: str,
    options: list[str],
    ranked_documents: list[dict[str, Any]],
    config: dict[str, Any],
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return document-level judgments for one survey question.

    Args:
        question: Exact question text from the configured question file.
        options: Exact answer-option strings in their original order.
        ranked_documents: The 100 supports produced by retrieval.
            Each record contains ``rank``, ``doc_id``, ``score``, ``title``, ``text``, ``source``, ``group``, and ``url``.
        config: Parsed settings from the INI file used for this run.
            The ``judging.judge_k`` value is an integer from 1 to 100. The default is 20.
        api_key: Course API key, or ``None`` when the judging method does not need it.

    Returns:
        Exactly ``judge_k`` dictionaries for distinct documents selected from ``ranked_documents``.
        Each dictionary must contain ``doc_id`` and ``option``.
        The option must be one exact string from ``options`` or ``non_relevant``.
        A short ``reason`` is optional.

        Example record:
        {"doc_id": "2WZHJKU27B", "option": "Mobile", "reason": "B0 token overlap"}

    The runner checks the count, document IDs, uniqueness, and option strings.
    You may replace this function or reorganise the internal pipeline.
    """
    # Retrieval and judging have separate method settings, so either can be tested with B0.
    method = str(config["judging"]["method"]).strip().lower()
    judge_k = judging_count(config)

    if method == "b0":
        # B0 always takes the first judge_k supports. Your own method may select another subset.
        return b0_judge(options, ranked_documents[:judge_k])

    if method == "llm_judge":
        return llm_judge(
            question,
            options,
            ranked_documents,
            config,
            api_key,
            judge_k,
        )

    raise ValueError(f"Unknown judging method: {method}")


def llm_judge(
    question: str,
    options: list[str],
    ranked_documents: list[dict[str, Any]],
    config: dict[str, Any],
    api_key: str | None,
    judge_k: int,
) -> list[dict[str, Any]]:
    """Judge the top ``judge_k`` supports with the configured chat model.

    Documents are sent in batches of ``judging.judge_batch_size`` with their text
    truncated to ``judging.max_doc_chars`` characters. A batch that fails is retried
    once with a fresh reply, then split, and finally judged by B0 so the returned
    list always holds one valid judgment per selected document.
    """
    if not api_key:
        raise ValueError("llm_judge requires --api_key")

    settings = config["judging"]
    batch_size = _positive_int(
        settings.get("judge_batch_size", judge_k), "judging.judge_batch_size"
    )
    max_doc_chars = _positive_int(
        settings.get("max_doc_chars", 1200), "judging.max_doc_chars"
    )

    selected = ranked_documents[:judge_k]
    examples = _load_examples(config)
    if examples:
        print_status(f"[judging] Using {len(examples)} few-shot example(s).", flush=True)
    judged: dict[str, dict[str, Any]] = {}
    for start in range(0, len(selected), batch_size):
        chunk = selected[start : start + batch_size]
        for record in _judge_chunk(
            question, options, chunk, config, api_key, max_doc_chars, examples
        ):
            # setdefault keeps the first judgment when a document appears twice.
            judged.setdefault(str(record["doc_id"]), record)

    # A batch reply may omit documents. Ask once more for those alone before B0.
    missing = [
        document for document in selected if str(document["doc_id"]) not in judged
    ]
    if missing:
        print_status(
            f"[judging] Re-asking the model for {len(missing)} unanswered document(s).",
            flush=True,
        )
        for record in _judge_chunk(
            question, options, missing, config, api_key, max_doc_chars, examples
        ):
            judged.setdefault(str(record["doc_id"]), record)
    missing = [
        document for document in selected if str(document["doc_id"]) not in judged
    ]
    if missing:
        print_status(
            f"[judging] Filling {len(missing)} document(s) with B0.",
            flush=True,
        )
        for record in b0_judge(options, missing):
            judged.setdefault(str(record["doc_id"]), record)

    # Return in retrieval-rank order so judgments.jsonl reads top-down.
    return [judged[str(document["doc_id"])] for document in selected]


def _judge_chunk(
    question: str,
    options: list[str],
    chunk: list[dict[str, Any]],
    config: dict[str, Any],
    api_key: str | None,
    max_doc_chars: int,
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Judge one batch, retrying, splitting, or falling back to B0 on failure.

    Returns one record per document in chunk. A usable reply is parsed and returned.
    An unusable reply or a recoverable API error leads to a second attempt with a
    fresh reply, then to smaller batches, and finally to B0 for single documents.
    """
    system_prompt = _SYSTEM_PROMPT
    user_prompt = _user_prompt(question, options, chunk, max_doc_chars, examples)
    doc_ids = {str(document["doc_id"]) for document in chunk}

    # A cached reply that fails to parse must be replaced under its own key, so
    # the second attempt repeats the same prompt with a fresh reply before the
    # stricter reminder wording is tried as a third attempt.
    attempts = (
        (config, user_prompt),
        (_fresh_config(config), user_prompt),
        (_fresh_config(config), user_prompt + _JSON_REMINDER),
    )
    best: list[dict[str, Any]] | None = None
    for attempt_config, prompt in attempts:
        try:
            reply = call_judging_model(system_prompt, prompt, attempt_config, api_key)
        except Exception as error:
            if not _is_recoverable(error):
                raise
            print_status(f"[judging] Recoverable model error: {error}", flush=True)
            continue
        records = _parse_judgments(reply, options, doc_ids)
        if records is None:
            print_status(
                "[judging] Reply did not contain usable judgments; requesting a fresh reply.",
                flush=True,
            )
            continue
        if len(records) == len(chunk):
            return records
        # A partial reply is kept only if nothing better arrives, so the cache
        # entry for this prompt can still be replaced by a fuller reply.
        if best is None or len(records) > len(best):
            best = records
        print_status(
            f"[judging] Reply covered {len(records)}/{len(chunk)} documents; "
            "requesting a fuller reply.",
            flush=True,
        )
    if best is not None:
        return best

    if len(chunk) > 1:
        middle = len(chunk) // 2
        print_status(
            f"[judging] Splitting the failed batch into {middle} and {len(chunk) - middle}.",
            flush=True,
        )
        return _judge_chunk(
            question, options, chunk[:middle], config, api_key, max_doc_chars
        ) + _judge_chunk(
            question, options, chunk[middle:], config, api_key, max_doc_chars
        )

    document = chunk[0]
    print_status(
        f"[judging] B0 fallback for document {document['doc_id']}.",
        flush=True,
    )
    return b0_judge(options, chunk)


def _user_prompt(
    question: str,
    options: list[str],
    chunk: list[dict[str, Any]],
    max_doc_chars: int,
    examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build one user message with optional examples, then the question and batch documents."""
    lines: list[str] = []
    # Demonstrations come first so the model copies the label-and-reason pattern.
    for position, example in enumerate(examples or [], start=1):
        text = str(example.get("text") or "").strip()
        if len(text) > _MAX_EXAMPLE_CHARS:
            text = text[:_MAX_EXAMPLE_CHARS] + " ...[truncated]"
        lines.append(f"Example {position}")
        lines.append(f"Question: {example['question']}")
        lines.append(f"Options: {json.dumps(example['options'], ensure_ascii=False)}")
        lines.append(f"Document:\n{text}")
        lines.append(
            "Judgment: "
            + json.dumps(
                {
                    "doc_id": str(example.get("doc_id") or f"EXAMPLE{position}"),
                    "option": example["option"],
                    "reason": str(example.get("reason") or "").strip(),
                },
                ensure_ascii=False,
            )
        )
        lines.append("")
    if examples:
        lines.append("Now judge the real documents below in the same way.")
    lines.extend(
        [
            f"Question: {question}",
            f"Options: {json.dumps(options, ensure_ascii=False)}",
            "Documents:",
        ]
    )
    for position, document in enumerate(chunk, start=1):
        text = str(document.get("text") or "").strip()
        if len(text) > max_doc_chars:
            text = text[:max_doc_chars] + " ...[truncated]"
        if not text:
            text = "(empty document)"
        lines.append(f"{position}. doc_id={document['doc_id']}\n{text}")
    lines.append(
        'Return the JSON object now: {"judgments": '
        '[{"doc_id": "...", "option": "...", "reason": "..."}]}'
    )
    return "\n".join(lines)


def _load_examples(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the few-shot demonstrations selected by judging.judge_shots.

    Args:
        config: Parsed INI settings. ``judge_shots`` is an integer from 0 to 4;
            the default 0 sends no examples, so prompts stay byte-identical to
            the zero-shot runs and their cached replies remain reusable.
            ``few_shot_file`` names the JSON example list (optional).

    Returns:
        The first ``judge_shots`` example objects, or an empty list when disabled.
    """
    settings = config["judging"]
    raw_shots = str(settings.get("judge_shots", 0)).strip()
    try:
        shots = int(raw_shots)
    except ValueError:
        raise ValueError("judging.judge_shots must be an integer from 0 to 4") from None
    if shots < 0 or shots > _MAX_SHOTS:
        raise ValueError("judging.judge_shots must be an integer from 0 to 4")
    if shots == 0:
        return []
    path = Path(str(settings.get("few_shot_file") or _DEFAULT_EXAMPLES_FILE))
    examples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"{path}: few-shot examples must be a non-empty JSON list")
    selected = examples[:shots]
    for position, example in enumerate(selected, start=1):
        if not isinstance(example, dict):
            raise ValueError(f"{path}: example {position} must be an object")
        for key in ("question", "options", "text", "option"):
            if key not in example:
                raise ValueError(f"{path}: example {position} is missing '{key}'")
        if not isinstance(example["options"], list) or not example["options"]:
            raise ValueError(f"{path}: example {position} options must be a non-empty list")
    return selected


def _parse_judgments(
    reply: str,
    options: list[str],
    doc_ids: set[str],
) -> list[dict[str, Any]] | None:
    """Convert a raw model reply into records, or return None when it is unusable.

    JSON is accepted with or without code fences and with common key names.
    Unknown documents, repeated documents, and unknown option strings are dropped,
    so the caller can fill the remaining documents itself.
    """
    value = _extract_json(reply)
    entries: Any = None
    if isinstance(value, list):
        entries = value
    elif isinstance(value, dict):
        # A single judgment object is accepted as a list of one.
        if "doc_id" in value or "id" in value:
            entries = [value]
        else:
            for key in ("judgments", "results", "items", "labels", "documents"):
                if isinstance(value.get(key), list):
                    entries = value[key]
                    break
    if not isinstance(entries, list):
        return None

    option_lookup = {option.casefold(): option for option in options}
    records: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        doc_id = _entry_value(entry, ("doc_id", "id", "document_id", "docid"))
        option = _entry_value(entry, ("option", "label", "answer", "choice"))
        if doc_id is None or option is None:
            continue
        doc_id = str(doc_id).strip()
        if doc_id not in doc_ids or doc_id in records:
            continue
        option_text = str(option).strip()
        matched = _match_option(option_text, option_lookup, options)
        if matched is None:
            # An unexpected label would fail validation, so leave it for the filler.
            continue
        reason = _entry_value(entry, ("reason", "explanation", "justification"))
        records[doc_id] = {
            "doc_id": doc_id,
            "option": matched,
            "reason": str(reason or "").strip()[:300],
        }
    # An empty parse is treated as a failed reply so a fresh one is requested.
    return list(records.values()) if records else None


def _extract_json(reply: str) -> Any:
    """Return the first JSON object or array found in a reply, or None.

    Models sometimes leave string values unquoted, so a reply that does not
    parse is requoted once and tried again before being given up on.
    """
    if not isinstance(reply, str) or not reply.strip():
        return None
    text = reply.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    for candidate in (text, _requoted(text)):
        decoder = json.JSONDecoder()
        for position, character in enumerate(candidate):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[position:])
            except ValueError:
                continue
            return value
    return None


def _requoted(text: str) -> str:
    """Quote bare word values that follow a colon, as in ``"id": 5EDU37EZZY``.

    Only letter-bearing tokens directly before a comma or brace are quoted, so
    already-quoted strings, arrays, plain numbers, and JSON keys are left alone.
    """
    return re.sub(
        r"(:\s*)([A-Za-z0-9][A-Za-z0-9_ ./&'()-]*)(\s*[,}\]])",
        _quote_bare_value,
        text,
    )


def _quote_bare_value(match: re.Match[str]) -> str:
    """Return one _requoted match, leaving pure numbers unquoted."""
    token = match.group(2).strip()
    if token.replace(".", "", 1).isdigit():
        return match.group(0)
    return f'{match.group(1)}"{token}"{match.group(3)}'


def _entry_value(entry: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first present value among common key spellings."""
    for key in keys:
        if key in entry:
            return entry[key]
    lowered = {str(key).lower(): value for key, value in entry.items()}
    for key in keys:
        if key in lowered:
            return lowered[key]
    return None


def _match_option(
    option_text: str,
    option_lookup: dict[str, str],
    options: list[str],
) -> str | None:
    """Return the option a model label refers to, or None when it is unclear.

    The exact and case-insensitive matches come first, then common "no choice"
    spellings, then a comparison with punctuation removed, so an answer such as
    "Action/adventure" still selects the option "Action-adventure". A label that
    lists several options at once is left unmatched rather than guessed.
    """
    folded = option_text.casefold().strip()
    matched = option_lookup.get(folded)
    if matched is not None:
        return matched
    if _is_non_relevant(option_text):
        return NON_RELEVANT

    def compact(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.casefold())

    compact_label = compact(folded)
    if not compact_label:
        return None
    compact_options = {compact(option): option for option in options}
    matched = compact_options.get(compact_label)
    if matched is not None:
        return matched
    # A single option inside a longer label is accepted; several matches are not.
    candidates = [
        option
        for key, option in compact_options.items()
        if len(key) >= 4 and key in compact_label
    ]
    return candidates[0] if len(candidates) == 1 else None


def _is_non_relevant(option_text: str) -> bool:
    """Recognise the supplied "no choice" label in common spellings."""
    normalised = re.sub(r"[^a-z]+", "_", option_text.casefold()).strip("_")
    return normalised in {
        NON_RELEVANT,
        "nonrelevant",
        "not_relevant",
        "no_choice",
        "none",
        "na",
        "n_a",
        "irrelevant",
        "no_answer",
    }


def _is_recoverable(error: Exception) -> bool:
    """Report whether a model error is worth retrying with a smaller batch.

    Authentication, configuration, and gateway failures are not recoverable here
    and must stop the run. Content filters, truncation, empty replies, and
    unusable cached replies are retried as the README describes.
    """
    if not isinstance(error, ValueError):
        return False
    message = str(error)
    return any(
        marker in message
        for marker in (
            "content filter",
            "truncated",
            "reply was empty",
            "no reply choices",
            "empty or is not text",
        )
    )


def _fresh_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of config that always requests a new API reply."""
    return {**config, "api": {**config["api"], "use_cache": False}}


def _positive_int(value: Any, label: str) -> int:
    """Return a config value as an integer above zero, or raise a labelled error."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive integer") from error
    if number < 1 or number != float(value):
        raise ValueError(f"{label} must be a positive integer")
    return number


def call_judging_model(
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
    api_key: str | None,
) -> str:
    """Send two prompt strings using the judging model and cache settings.

    Args:
        system_prompt: Your instructions for judging evidence.
        user_prompt: Your question, options and selected evidence text.
        config: Uses judging.model, judging.max_tokens, api.base_url, api.use_cache and paths.api_cache.
            max_tokens limits the requested output tokens for this call.
        api_key: Course key passed with --api_key.

    Returns:
        The model's original reply as a string. The API helper also prints it.

    The helper sends the two prompt strings exactly as supplied.
    judge_evidence selects the documents and decides which fields to put in those strings.
    """
    # Keep gateway, model and cache settings in the INI file. Prompts come from your method.
    return chat(
        api_key=api_key,
        base_url=str(config["api"]["base_url"]),
        cache_path=config_path(config, "api_cache"),
        model=str(config["judging"]["model"]),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=int(config["judging"]["max_tokens"]),
        use_cache=bool(config["api"]["use_cache"]),
    )
