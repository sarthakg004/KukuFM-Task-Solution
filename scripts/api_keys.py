"""Discover configured Gemini keys without exposing their values."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv


KEY_PATTERN = re.compile(r'^GEMINI_API_KEY(\d*)$')


def discover_gemini_keys(
    names: list[str] | None = None,
    env_path: str | Path | None = None,
) -> list[str]:
    """Find base or numbered Gemini keys, skipping blanks and duplicate values.

    Numbering can have gaps. Environment variables take precedence over .env.
    An optional explicit list remains supported for callers and isolated tests.
    Only variable names are returned; credentials are never printed or saved.
    """
    path = Path(env_path) if env_path is not None else Path(__file__).resolve().parents[1] / '.env'
    load_dotenv(path)

    if names is None:
        names = [name for name in os.environ if KEY_PATTERN.fullmatch(name)]
        names.sort(key=lambda name: int(KEY_PATTERN.fullmatch(name)[1] or -1))

    available = []
    seen = set()
    for name in dict.fromkeys(names):
        value = os.getenv(name, '').strip()
        if not value or value in seen:
            continue
        seen.add(value)
        available.append(name)

    return available
