"""Lightweight URL cache to avoid refetching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class CacheEntry:
    url: str
    content_hash: str
    last_fetch: datetime

    def to_json(self) -> Dict[str, str]:
        return {
            "url": self.url,
            "content_hash": self.content_hash,
            "last_fetch": self.last_fetch.isoformat(),
        }

    @classmethod
    def from_json(cls, data: Dict[str, str]) -> "CacheEntry":
        return cls(
            url=data["url"],
            content_hash=data["content_hash"],
            last_fetch=datetime.fromisoformat(data["last_fetch"]),
        )


class UrlCache:
    """Very small JSON cache on disk."""

    def __init__(self, path: Path):
        self.path = path
        self._store: Dict[str, CacheEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._store = {url: CacheEntry.from_json(entry) for url, entry in raw.items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {url: entry.to_json() for url, entry in self._store.items()}
        self.path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    def get(self, url: str) -> Optional[CacheEntry]:
        return self._store.get(url)

    def set(self, url: str, content_hash: str, fetched_at: datetime) -> None:
        self._store[url] = CacheEntry(url=url, content_hash=content_hash, last_fetch=fetched_at)
        self._save()
