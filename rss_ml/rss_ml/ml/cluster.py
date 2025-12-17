"""Clustering of article embeddings with HDBSCAN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from rss_ml.nlp.embed import Embedder, embed_texts

try:
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover
    hdbscan = None


@dataclass
class ClusterConfig:
    min_cluster_size: int = 2
    min_samples: int | None = None
    metric: str = "euclidean"
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "leaf"


@dataclass
class ClusterResult:
    labels: List[int]
    probabilities: List[float]
    cluster_map: Dict[str, str]  # article_id -> cluster_id (or "noise")
    cluster_sizes: Dict[str, int]


def _label_to_id(label: int) -> str:
    return "noise" if label == -1 else f"c{label}"


def cluster_embeddings(
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    config: ClusterConfig | None = None,
) -> ClusterResult:
    """Run HDBSCAN on embeddings and return cluster mapping and metrics."""

    if hdbscan is None:
        raise ImportError("hdbscan is required for clustering")

    cfg = config or ClusterConfig()
    X = np.array(embeddings)
    model = hdbscan.HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        metric=cfg.metric,
        cluster_selection_epsilon=cfg.cluster_selection_epsilon,
        cluster_selection_method=cfg.cluster_selection_method,
        prediction_data=True,
    )
    model.fit(X)

    labels = model.labels_.tolist()
    probabilities = (
        model.probabilities_.tolist()
        if hasattr(model, "probabilities_")
        else [1.0] * len(labels)
    )

    cluster_map = {str(id_): _label_to_id(label) for id_, label in zip(ids, labels)}

    cluster_sizes: Dict[str, int] = {}
    for label in labels:
        cid = _label_to_id(label)
        cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

    return ClusterResult(
        labels=labels,
        probabilities=probabilities,
        cluster_map=cluster_map,
        cluster_sizes=cluster_sizes,
    )


def cluster_dataframe(
    df: pd.DataFrame,
    text_column: str = "text_for_embedding",
    id_column: str = "id",
    embedder: Embedder | None = None,
    config: ClusterConfig | None = None,
) -> ClusterResult:
    """Embed texts from a DataFrame then cluster."""

    if text_column not in df.columns:
        raise ValueError(f"Missing column '{text_column}' for clustering")

    ids = (
        df[id_column].astype(str).tolist()
        if id_column in df.columns
        else df.index.astype(str).tolist()
    )
    texts = df[text_column].fillna("").tolist()
    embeddings = embed_texts(texts, embedder=embedder)

    return cluster_embeddings(ids, embeddings, config=config)
