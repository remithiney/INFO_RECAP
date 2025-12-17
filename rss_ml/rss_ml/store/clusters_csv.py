"""CSV writer for cluster summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class ClusterRecord:
    cluster_id: str
    cluster_size: int
    cluster_score: float
    cluster_summary: str
    top_urls: str
    languages: str
    representative_title: str
    created_at: str


def write_clusters_csv(path: Path, records: Iterable[ClusterRecord]) -> Path:
    df = pd.DataFrame([asdict(r) for r in records])
    columns = [
        "cluster_id",
        "cluster_size",
        "cluster_score",
        "cluster_summary",
        "top_urls",
        "languages",
        "representative_title",
        "created_at",
    ]
    df = df.reindex(columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
