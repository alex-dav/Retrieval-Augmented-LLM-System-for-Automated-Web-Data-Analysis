"""Load INI configuration files and resolve their paths."""

from __future__ import annotations

from configparser import ConfigParser, Error as ConfigParserError
from pathlib import Path
from typing import Any


SUPPORTS_K = 100
MAX_JUDGE_K = SUPPORTS_K
NON_RELEVANT = "non_relevant"


def _parse_value(value: str) -> str | int | float | bool:
    """Convert simple INI values while leaving paths and names as strings."""
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    # INI stores text. Try integer, then decimal, otherwise retain the string.
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def load_config(config_file: str) -> dict[str, Any]:
    """Return INI sections as dictionaries with bool, int, float or string values.

    config_file is resolved from the current folder.
    Paths inside it are resolved by config_path relative to the INI file's own folder.
    """
    path = Path(config_file).expanduser().resolve()
    parser = ConfigParser(interpolation=None)
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            parser.read_file(stream)
    except OSError as error:
        raise ValueError(f"Cannot read INI config {path}: {error}") from error
    except ConfigParserError as error:
        raise ValueError(f"Invalid INI config {path}: {error}") from error

    # For example, [judging] judge_k = 20 becomes config["judging"]["judge_k"] = 20.
    config: dict[str, Any] = {}
    for section in parser.sections():
        config[section] = {}
        for key, value in parser.items(section):
            config[section][key] = _parse_value(value)
    for section in ("paths", "retrieval", "judging", "api"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Config section '{section}' is missing: {path}")
    # Remember the location so relative data/output paths work from any working folder.
    config["_config_file"] = str(path)
    config["_config_dir"] = str(path.parent)
    return config


def config_path(config: dict[str, Any], key: str) -> Path:
    """Return an absolute Path for a named entry in config['paths'].

    For example, config_path(config, "run_dir") resolves the run_dir setting relative to the INI file's folder, unless it is already an absolute path.
    """
    value = config["paths"].get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config path '{key}' is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def judging_count(config: dict[str, Any]) -> int:
    """Read the selected document count without silently rounding a decimal."""
    count = config["judging"]["judge_k"]
    # Reject decimals and booleans, rather than silently turning them into a count.
    if type(count) is not int or not 1 <= count <= MAX_JUDGE_K:
        raise ValueError(f"judging.judge_k must be an integer from 1 to {MAX_JUDGE_K}")
    return count
