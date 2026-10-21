"""Call course chat models, print their raw replies, and save them for reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

from openai import OpenAI

from common.io import print_status

_CACHE_LOCK = Lock()


def chat(
    *,
    api_key: str | None,
    base_url: str,
    cache_path: Path,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    use_cache: bool = True,
) -> str:
    """Print and return a model's original reply as a string.

    Args:
        api_key: Course key supplied with --api_key. A cached reply needs no key.
        base_url: Course gateway URL from config["api"]["base_url"].
        cache_path: JSON file holding request-to-reply entries. Its folder is created.
        model: Approved chat model name.
        system_prompt: Instructions for the model.
        user_prompt: Text to send for this request.
        max_tokens: Maximum number of output tokens requested.
        use_cache: Reuse saved replies when True. False requests and saves fresh replies.

    Returns:
        Raw reply text, also printed with its API or cache source.
        Parse it in your method.
        A cached reply may be incorrect, so inspect it before use.

    Threads in this process share a cache lock.
    Use separate cache files for separate Python processes.
    This call waits for the reply. The lock protects cache file access while requests run outside it.
    Every uncached request uses the API budget.
    """
    # 1. Identify the exact request. Changing a prompt or model creates a different cache key.
    # The API key is deliberately excluded from the saved request identifier.
    payload = {
        "base_url": base_url,
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_tokens": int(max_tokens),
    }
    key = _cache_key(payload)
    # Read the file in both modes so new replies can be merged with its other entries.
    with _CACHE_LOCK:
        cache = _load_cache(cache_path)
    if use_cache and key in cache:
        cached = cache[key]
        print_status(f"[api cache] {cache_path}. Set use_cache = false to refresh saved replies.", flush=True)
        if not isinstance(cached, str) or not cached.strip():
            raise ValueError(
                f"Cached API reply is empty or is not text: {cache_path}. "
                "Set [api] use_cache = false to request and save a fresh reply."
            )
        print_status(cached, flush=True)
        return cached

    # 2. With no reusable response, send the two messages through the course gateway.
    if not api_key:
        raise ValueError("This method requires --api_key")
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": int(max_tokens),
    }
    try:
        response = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
            **request
        )
    except Exception as error:
        if not _is_content_filter_error(error):
            raise
        raise ValueError(
            "The API provider's content filter rejected this request. The supplied corpus was screened, "
            "but the provider uses a separate safety filter and may reject real-world political or social "
            "text by mistake. Retry once. If the error repeats, reduce the judging batch size and retry "
            "smaller groups to identify the triggering document. No reply was saved to the cache."
        ) from error
    # 3. Show the original reply so you can inspect prompt and parsing problems.
    print_status("[api response]", flush=True)
    if not response.choices:
        raise ValueError("The API returned no reply choices. No reply was saved to the cache.")
    choice = response.choices[0]
    text = choice.message.content or ""
    print_status(text, flush=True)
    if choice.finish_reason == "length":
        raise ValueError(
            "The API reply was truncated. Check max_tokens and shorten the prompt or requested output. "
            "No reply was saved to the cache."
        )
    if not text.strip():
        raise ValueError("The API reply was empty. No reply was saved to the cache.")

    # 4. Save new replies in both modes. Your method still needs to interpret and check the text.
    # Reload after the call so concurrent threads keep each other's responses.
    with _CACHE_LOCK:
        cache = _load_cache(cache_path)
        cache[key] = text
        try:
            _save_cache(cache_path, cache)
        except OSError:
            print_status(f"[api cache] Could not save the reply to {cache_path}.", flush=True)
            raise
    return text


def _cache_key(payload: dict[str, Any]) -> str:
    """Identify a request by its model, prompts, output limit and gateway URL."""
    # Sorting keys makes identical settings produce the same identifier.
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_content_filter_error(error: Exception) -> bool:
    """Recognise provider content-filter errors without hiding other API failures."""
    details = [str(error), repr(getattr(error, "body", "")), str(getattr(error, "code", ""))]
    response = getattr(error, "response", None)
    if response is not None:
        details.append(str(getattr(response, "text", "")))
    return "content_filter" in " ".join(details).lower()


def _load_cache(path: Path) -> dict[str, Any]:
    """Read saved raw replies, or start empty when the cache file is absent."""
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read API cache: {path}. Check paths.api_cache and repair or move the file before retrying."
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"API cache must contain a JSON object: {path}. Repair or move the file before retrying.")
    return value


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    """Close the temporary file before replacing the cache, including on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write a complete replacement first, leaving the existing cache intact during writing.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(cache, ensure_ascii=False, indent=2) + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
