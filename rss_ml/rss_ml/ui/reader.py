"""CLI reader for clusters with filters and pagination."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Read and filter clusters.")


def _load_clusters(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Clusters file not found: {path}")
    return pd.read_csv(path)


def _paginate(df: pd.DataFrame, page: int, page_size: int) -> pd.DataFrame:
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return df.iloc[start:end]


@app.command("read-clusters")
def read_clusters(
    clusters_csv: Path = typer.Argument(Path("data") / "clusters.csv", help="Path to clusters CSV."),
    min_size: int = typer.Option(1, help="Filter clusters by minimum size."),
    language: Optional[str] = typer.Option(None, help="Filter by language code substring."),
    query: Optional[str] = typer.Option(None, help="Keyword substring to filter summaries or titles."),
    sort_by: str = typer.Option("cluster_score", help="Column to sort by."),
    descending: bool = typer.Option(True, help="Sort descending."),
    page: int = typer.Option(1, help="Page number (1-based)."),
    page_size: int = typer.Option(20, help="Rows per page."),
    show_summary_chars: int = typer.Option(220, help="Trim summary display length."),
):
    """Display clusters in a paginated Rich table."""

    console = Console()
    df = _load_clusters(clusters_csv)

    if min_size > 1 and "cluster_size" in df.columns:
        df = df[df["cluster_size"] >= min_size]
    if language and "languages" in df.columns:
        df = df[df["languages"].str.contains(language, case=False, na=False)]
    if query:
        df = df[
            df["cluster_summary"].str.contains(query, case=False, na=False)
            | df["representative_title"].str.contains(query, case=False, na=False)
        ]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending)

    total_rows = len(df)
    df = _paginate(df, page=page, page_size=page_size)

    table = Table(title=f"Clusters page {page} (showing {len(df)}/{total_rows})")
    for col in [
        "cluster_id",
        "cluster_size",
        "cluster_score",
        "representative_title",
        "cluster_summary",
        "top_urls",
        "languages",
        "created_at",
    ]:
        if col in df.columns:
            table.add_column(col)

    for _, row in df.iterrows():
        summary = str(row.get("cluster_summary", ""))[:show_summary_chars]
        table.add_row(
            str(row.get("cluster_id", "")),
            str(row.get("cluster_size", "")),
            str(row.get("cluster_score", "")),
            str(row.get("representative_title", "")),
            summary,
            str(row.get("top_urls", "")),
            str(row.get("languages", "")),
            str(row.get("created_at", "")),
        )

    console.print(table)
