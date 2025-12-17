"""Extract article content directly from RSS entries (no page scraping)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from bs4 import BeautifulSoup

from rss_ml.ingest.fetch import FeedEntry
from rss_ml.store.cache import UrlCache


@dataclass
class ExtractedContent:
    url: str
    retrieved_at: datetime
    content: str
    main_text: str
    content_hash: str
    error: Optional[str] = None


@dataclass
class ExtractionConfig:
    """Lightweight options for feed-based extraction."""

    cache_ttl_seconds: int = 6 * 3600
    strip_html: bool = True


def _hash_content(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _is_cache_fresh(last_fetch: datetime, ttl_seconds: int) -> bool:
    return datetime.utcnow() - last_fetch < timedelta(seconds=ttl_seconds)


def _html_to_text(raw: str, strip_html: bool) -> str:
    if not raw:
        return ""
    if not strip_html:
        return raw.strip()
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def extract_content(
    entry: FeedEntry,
    cache: Optional[UrlCache] = None,
    config: ExtractionConfig | None = None,
) -> ExtractedContent:
    """Build ExtractedContent from the feed payload; no HTTP requests involved."""

    config = config or ExtractionConfig()
    retrieved_at = datetime.utcnow()

    text = _html_to_text(entry.content, config.strip_html)
    if not text:
        text = entry.title or ""

    content_hash = _hash_content(text or entry.url)

    cached_hash: Optional[str] = None
    if cache:
        cached = cache.get(entry.url)
        if cached and _is_cache_fresh(cached.last_fetch, config.cache_ttl_seconds):
            cached_hash = cached.content_hash

    if cache and cached_hash == content_hash:
        return ExtractedContent(
            url=entry.url,
            retrieved_at=retrieved_at,
            content=text,
            main_text=text,
            content_hash=content_hash,
            error="cached",
        )

    if cache:
        cache.set(entry.url, content_hash, retrieved_at)

    error_message = "empty feed content" if not text else None

    return ExtractedContent(
        url=entry.url,
        retrieved_at=retrieved_at,
        content=text,
        main_text=text,
        content_hash=content_hash,
        error=error_message,
    )
