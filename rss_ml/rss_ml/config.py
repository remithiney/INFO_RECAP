"""Configuration dataclasses for rss_ml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CacheConfig:
    """Cache tuning for fetch/extract steps."""

    path: Path = Path("data") / "cache.sqlite"
    ttl_seconds: int = 6 * 3600


@dataclass
class FeedsConfig:
    """Configuration for feed ingestion."""

    feeds_file: Path = Path("data") / "feeds.txt"
    user_agent: str = "rss-ml/0.1"
    max_entries: Optional[int] = None
    max_title_chars: int = 300
    max_content_chars: int = 20000
    cache: CacheConfig = field(default_factory=CacheConfig)


@dataclass
class OutputPaths:
    """Output locations for intermediate artifacts."""

    articles_raw: Path = Path("data") / "articles_raw.csv"
    articles_features: Path = Path("data") / "articles_features.csv"
    clusters: Path = Path("data") / "clusters.csv"


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    feeds: FeedsConfig = field(default_factory=FeedsConfig)
    outputs: OutputPaths = field(default_factory=OutputPaths)
