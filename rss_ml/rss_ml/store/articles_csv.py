"""CSV writer for raw articles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from rss_ml.ingest.fetch import FeedEntry
from rss_ml.ingest.extract import ExtractedContent


@dataclass
class ArticleRecord:
    published_at: Optional[str]
    title: str
    content: str
    retrieved_at: str
    authors: str
    feed: str
    url: str
    content_hash: str


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt else None


def build_record(entry: FeedEntry, content: ExtractedContent) -> ArticleRecord:
    """Merge feed entry metadata with extracted content into a CSV row."""

    return ArticleRecord(
        published_at=_iso(entry.published),
        title=entry.title,
        content=content.content,
        retrieved_at=_iso(content.retrieved_at) or "",
        authors=entry.author or "",
        feed=entry.feed_name,
        url=entry.url,
        content_hash=content.content_hash,
    )


def build_records(
    entries: Sequence[FeedEntry],
    contents_by_url: Dict[str, ExtractedContent],
) -> List[ArticleRecord]:
    """Build article records aligning fetched entries and extracted content."""

    records: List[ArticleRecord] = []
    for entry in entries:
        content = contents_by_url.get(entry.url)
        if not content:
            continue
        records.append(build_record(entry, content))
    return records


def to_dataframe(records: Iterable[ArticleRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


def write_articles_csv(path: Path, records: Iterable[ArticleRecord]) -> Path:
    """Persist articles to CSV with the expected schema, appending if file exists."""

    new_df = to_dataframe(records)
    columns = [
        "published_at",
        "title",
        "content",
        "retrieved_at",
        "authors",
        "feed",
        "url",
        "content_hash",
    ]
    new_df = new_df.reindex(columns=columns)

    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["url"], keep="last")
        df_to_write = combined.reindex(columns=columns)
    else:
        df_to_write = new_df

    path.parent.mkdir(parents=True, exist_ok=True)
    df_to_write.to_csv(path, index=False)
    return path
