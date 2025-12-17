"""Cluster summarization and CSV export."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from rss_ml.nlp.summarize import Summarizer, summarize_batch
from rss_ml.store.clusters_csv import ClusterRecord, write_clusters_csv


@dataclass
class ClusterSummaryConfig:
    max_articles_per_cluster: int = 6
    summary_language: str = "en"
    default_score_per_article: float = 1.0
    created_at: datetime = datetime.utcnow()
    batch_size: int = 4


def _representative_title(titles: Sequence[str]) -> str:
    # Simple heuristic: choose the median-length title
    if not titles:
        return ""
    sorted_titles = sorted(titles, key=len)
    return sorted_titles[len(sorted_titles) // 2]


def _top_urls(urls: Sequence[str], limit: int = 5) -> str:
    return "|".join(urls[:limit])


def _languages_counts(langs: Iterable[str]) -> str:
    counts = Counter(langs)
    return "|".join(f"{lang}:{cnt}" for lang, cnt in counts.most_common())


def summarize_clusters(
    articles_df: pd.DataFrame,
    cluster_assignments: Dict[str, str],
    output_path: Path,
    summarizer: Summarizer | None = None,
    config: ClusterSummaryConfig | None = None,
) -> Path:
    """
    Summarize clusters and write clusters.csv.

    Parameters
    ----------
    articles_df : DataFrame
        Must include columns: id/url, title, summary, language.
    cluster_assignments : mapping article_id -> cluster_id
    output_path : path for clusters.csv
    """

    cfg = config or ClusterSummaryConfig()
    summarizer = summarizer or Summarizer()

    clusters: Dict[str, List[str]] = defaultdict(list)
    for article_id, cluster_id in cluster_assignments.items():
        clusters[cluster_id].append(article_id)

    records: List[ClusterRecord] = []

    for cluster_id, article_ids in clusters.items():
        subset = articles_df[articles_df["id"].isin(article_ids)].copy()
        subset = subset.sort_values("published_at")

        titles = subset["title"].fillna("").tolist()
        urls = subset["url"].fillna("").tolist()
        languages = subset["language"].fillna("und").tolist()
        summaries = subset["summary"].fillna("").tolist()

        # Select up to N summaries/texts for the cluster summary
        texts_for_summary = summaries[: cfg.max_articles_per_cluster]
        if not any(texts_for_summary) and "content" in subset.columns:
            texts_for_summary = subset["content"].fillna("").tolist()[: cfg.max_articles_per_cluster]

        cluster_summary_list = summarize_batch(
            texts_for_summary,
            language=cfg.summary_language,
            summarizer=summarizer,
        )
        cluster_summary = " ".join(cluster_summary_list).strip()

        representative_title = _representative_title(titles)
        cluster_size = len(article_ids)
        cluster_score = cluster_size * cfg.default_score_per_article

        record = ClusterRecord(
            cluster_id=str(cluster_id),
            cluster_size=cluster_size,
            cluster_score=cluster_score,
            cluster_summary=cluster_summary,
            top_urls=_top_urls(urls),
            languages=_languages_counts(languages),
            representative_title=representative_title,
            created_at=cfg.created_at.isoformat(),
        )
        records.append(record)

    write_clusters_csv(output_path, records)
    return output_path
