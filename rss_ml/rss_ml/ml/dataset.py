"""Dataset builder for downstream ML with cleaning and export options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import pandas as pd


@dataclass
class DatasetConfig:
    min_length: int = 80
    allowed_languages: Optional[Sequence[str]] = None
    lowercase: bool = False
    use_summary: bool = False  # if True, use summary instead of full content
    output_path: Optional[Path] = None  # if set, write parquet/csv


def _filter_languages(df: pd.DataFrame, allowed: Optional[Iterable[str]]) -> pd.DataFrame:
    if not allowed:
        return df
    allowed_set: Set[str] = set(allowed)
    return df[df["language"].isin(allowed_set)]


def _choose_text(df: pd.DataFrame, use_summary: bool) -> pd.Series:
    if use_summary and "summary" in df.columns:
        return df["summary"].fillna("")
    return df["content"].fillna("")


def build_dataset(
    articles_features_csv: Path,
    config: DatasetConfig | None = None,
) -> pd.DataFrame:
    """Load, clean, filter articles, and add text_for_embedding."""

    cfg = config or DatasetConfig()
    df = pd.read_csv(articles_features_csv)
    df = df.drop_duplicates(subset=["content_hash"])

    text_col = _choose_text(df, cfg.use_summary)
    if cfg.lowercase:
        text_col = text_col.str.lower()
    df = df[text_col.str.len() >= cfg.min_length].copy()

    df = _filter_languages(df, cfg.allowed_languages)
    df["text_for_embedding"] = df["title"].fillna("") + "\n\n" + text_col

    if cfg.output_path:
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.output_path.suffix.lower() == ".parquet":
            df.to_parquet(cfg.output_path, index=False)
        else:
            df.to_csv(cfg.output_path, index=False)
    return df
