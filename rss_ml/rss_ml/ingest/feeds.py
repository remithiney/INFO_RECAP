"""Feed list loader."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def normalize_feed(feed: str) -> str:
    """Trim and normalize a feed URL or name."""

    return feed.strip()


def dedupe(items: Iterable[str]) -> List[str]:
    """Deduplicate while preserving order."""

    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_feeds(path: Path) -> List[str]:
    """Load feeds from a text file, normalizing and deduping."""

    if not path.exists():
        raise FileNotFoundError(f"Feeds file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = [line.strip() for line in f.readlines()]

    feeds = [
        normalize_feed(line)
        for line in raw
        if line and not line.startswith("#")
    ]
    return dedupe(feeds)
