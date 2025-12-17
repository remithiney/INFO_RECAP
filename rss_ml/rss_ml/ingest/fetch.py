"""Fetch RSS entries with error handling, limits, and progress."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

import feedparser
from rich.console import Console
from rich.progress import Progress, TaskID


@dataclass
class FeedEntry:
    url: str
    title: str
    published: Optional[datetime]
    author: Optional[str]
    feed_name: str
    content: str = ""


@dataclass
class FetchError:
    feed_url: str
    message: str


@dataclass
class FetchResult:
    entries: List[FeedEntry]
    errors: List[FetchError]


def _get_entry_content(entry: feedparser.FeedParserDict) -> str:
    """Return the richest content available from the feed entry."""

    if entry.get("content"):
        parts = entry.content
        if isinstance(parts, list) and parts:
            first = parts[0]
            if isinstance(first, dict):
                return first.get("value") or ""
            return str(first)
        if isinstance(parts, dict):
            return parts.get("value") or ""
        return str(parts)
    return entry.get("summary") or entry.get("description") or ""


def _parse_entry(entry: feedparser.FeedParserDict, feed_title: str) -> FeedEntry:
    published = None
    if "published_parsed" in entry and entry.published_parsed:
        published = datetime(*entry.published_parsed[:6])
    return FeedEntry(
        url=entry.get("link") or "",
        title=entry.get("title") or "",
        published=published,
        author=entry.get("author"),
        feed_name=feed_title,
        content=_get_entry_content(entry),
    )


def fetch_entries(
    feeds: Sequence[str],
    user_agent: str = "rss-ml/0.1",
    max_entries: Optional[int] = None,
    per_feed_limit: Optional[int] = None,
    max_title_chars: Optional[int] = None,
    max_content_chars: Optional[int] = None,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
    console: Optional[Console] = None,
) -> FetchResult:
    """Fetch entries from a list of feed URLs with logging and limits."""

    console = console or Console()
    entries: List[FeedEntry] = []
    errors: List[FetchError] = []

    if progress and task_id is not None:
        progress.update(task_id, total=len(feeds))

    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url, request_headers={"User-Agent": user_agent})
        except Exception as exc:  # noqa: BLE001
            errors.append(FetchError(feed_url=feed_url, message=str(exc)))
            console.log(f"[red]Feed parse error[/red] {feed_url}: {exc}")
            if progress and task_id is not None:
                progress.advance(task_id, 1)
            continue

        if getattr(parsed, "bozo", False):
            bozo_exc = getattr(parsed, "bozo_exception", None)
            msg = str(bozo_exc) if bozo_exc else "unknown parse error"
            errors.append(FetchError(feed_url=feed_url, message=msg))
            console.log(f"[yellow]Feed warning[/yellow] {feed_url}: {msg}")

        feed_title = parsed.feed.get("title", feed_url) if parsed and parsed.feed else feed_url
        count_in_feed = 0

        for entry in getattr(parsed, "entries", []):
            if per_feed_limit and count_in_feed >= per_feed_limit:
                break
            feed_entry = _parse_entry(entry, feed_title)
            if not feed_entry.url:
                errors.append(FetchError(feed_url=feed_url, message="entry missing URL"))
                continue
            if max_title_chars and len(feed_entry.title) > max_title_chars:
                errors.append(
                    FetchError(feed_url=feed_url, message=f"title too long ({len(feed_entry.title)} chars)")
                )
                console.log(f"[yellow]Skipped[/yellow] {feed_url}: title too long")
                continue
            if max_content_chars and len(feed_entry.content) > max_content_chars:
                errors.append(
                    FetchError(feed_url=feed_url, message=f"content too long ({len(feed_entry.content)} chars)")
                )
                console.log(f"[yellow]Skipped[/yellow] {feed_url}: content too long")
                continue
            entries.append(feed_entry)
            count_in_feed += 1

            if max_entries and len(entries) >= max_entries:
                break

        if progress and task_id is not None:
            progress.advance(task_id, 1)

        if max_entries and len(entries) >= max_entries:
            break

    return FetchResult(entries=entries, errors=errors)
