"""CLI orchestration for rss-ml pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import typer
from rich.console import Console

from rss_ml.config import PipelineConfig
from rss_ml.ingest.extract import ExtractedContent, ExtractionConfig, extract_content
from rss_ml.ingest.fetch import FetchResult, fetch_entries
from rss_ml.ingest.feeds import load_feeds
from rss_ml.ml.cluster import ClusterConfig, cluster_dataframe
from rss_ml.ml.cluster_summarize import ClusterSummaryConfig, summarize_clusters
from rss_ml.ml.dataset import DatasetConfig, build_dataset
from rss_ml.nlp.classify import Classifier, ClassifyConfig, classify_batch
from rss_ml.nlp.language import LanguageDetector, detect_languages
from rss_ml.nlp.summarize import SummarizeConfig, Summarizer, summarize_batch
from rss_ml.store.articles_csv import ArticleRecord, build_records, write_articles_csv
from rss_ml.store.cache import UrlCache
from rss_ml.store.clusters_csv import write_clusters_csv
from rss_ml.ui.progress import PipelineProgress

app = typer.Typer(help="rss-ml pipeline CLI.")


def _ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = df.index.astype(str)
    return df


def _log_errors(console: Console, label: str, errors: List[str]) -> None:
    if errors:
        console.log(f"[yellow]{label}: {len(errors)}[/yellow]")
        for err in errors[:5]:
            console.log(f"  {err}")


@app.command("ingest")
def ingest(
    feeds_file: Path = typer.Option(Path("data") / "feeds.txt", help="Feeds list."),
    articles_out: Path = typer.Option(Path("data") / "articles_raw.csv", help="Output CSV for articles."),
    max_entries: Optional[int] = typer.Option(None, help="Max total entries."),
    per_feed_limit: Optional[int] = typer.Option(None, help="Max entries per feed."),
):
    """Fetch feeds, extract articles, and write raw CSV."""

    console = Console()
    cfg = PipelineConfig()
    cache = UrlCache(cfg.feeds.cache.path)
    extraction_cfg = ExtractionConfig()

    feeds = load_feeds(feeds_file)
    console.log(f"Loaded {len(feeds)} feeds.")

    with PipelineProgress(console) as pp:
        pp.set_total("fetch_entries", len(feeds))
        fetch_result: FetchResult = fetch_entries(
            feeds,
            user_agent=cfg.feeds.user_agent,
            max_entries=max_entries,
            per_feed_limit=per_feed_limit,
            max_title_chars=cfg.feeds.max_title_chars,
            max_content_chars=cfg.feeds.max_content_chars,
            progress=pp.progress,
            task_id=pp.tasks["fetch_entries"],
            console=console,
        )

        entries = fetch_result.entries
        pp.set_total("extract_content", len(entries))

        contents: Dict[str, ExtractedContent] = {}
        extract_errors: List[str] = []
        for entry in entries:
            content = extract_content(entry, cache=cache, config=extraction_cfg)
            contents[entry.url] = content
            if content.error and content.error != "cached":
                extract_errors.append(f"{entry.url}: {content.error}")
            pp.advance("extract_content")

        records: List[ArticleRecord] = build_records(entries, contents)
        write_articles_csv(articles_out, records)
        pp.update_completed("write_articles")

    _log_errors(console, "Fetch errors", [f"{e.feed_url}: {e.message}" for e in fetch_result.errors])
    _log_errors(console, "Extract errors", extract_errors)
    console.log(f"[green]Ingestion done[/green]: {len(records)} articles -> {articles_out}")


@app.command("featurize")
def featurize(
    articles_csv: Path = typer.Option(Path("data") / "articles_raw.csv", help="Input raw articles CSV."),
    features_out: Path = typer.Option(Path("data") / "articles_features.csv", help="Output features CSV."),
):
    """Apply language detection, summarization, and classification."""

    console = Console()
    df = pd.read_csv(articles_csv)
    df = _ensure_id_column(df)

    with PipelineProgress(console) as pp:
        # Language detection
        pp.set_total("nlp_language", len(df))
        detector = LanguageDetector()
        lang_results = detect_languages(df["content"].fillna("").tolist(), detector=detector)
        df["language"] = [res.language for res in lang_results]
        df["language_score"] = [res.score for res in lang_results]
        pp.update_completed("nlp_language")

        # Summarization per language group to pick right model
        pp.set_total("nlp_summarize", len(df))
        summarizer = Summarizer(SummarizeConfig())
        summaries: List[str] = [""] * len(df)
        for lang in df["language"].unique():
            idxs = df.index[df["language"] == lang].tolist()
            texts = df.loc[idxs, "content"].fillna("").tolist()
            lang_code = "fr" if str(lang).startswith("fr") else "en"
            outs = summarize_batch(texts, language=lang_code, summarizer=summarizer)
            for idx, summary in zip(idxs, outs):
                summaries[idx] = summary
                pp.advance("nlp_summarize")

        df["summary"] = summaries

        # Classification on summaries (fallback to content if empty)
        pp.set_total("nlp_classify", len(df))
        classifier = Classifier(ClassifyConfig())
        classify_texts = [
            s if s else c for s, c in zip(df["summary"].fillna(""), df["content"].fillna(""))
        ]
        class_results = classify_batch(classify_texts, classifier=classifier)
        df["label"] = [r.label for r in class_results]
        df["label_score"] = [r.score for r in class_results]
        pp.update_completed("nlp_classify")

    df.to_csv(features_out, index=False)
    console.log(f"[green]Featurization done[/green]: {len(df)} rows -> {features_out}")


@app.command("cluster")
def cluster(
    features_csv: Path = typer.Option(Path("data") / "articles_features.csv", help="Input features CSV."),
    dataset_out: Path = typer.Option(Path("data") / "dataset.parquet", help="Output dataset with clusters."),
    min_length: int = typer.Option(80, help="Min content length."),
    languages: Optional[List[str]] = typer.Option(None, help="Allowed languages (repeatable)."),
    min_cluster_size: int = typer.Option(2, help="HDBSCAN min cluster size."),
):
    """Embed articles and cluster with HDBSCAN."""

    console = Console()
    cfg = DatasetConfig(min_length=min_length, allowed_languages=languages, output_path=dataset_out)
    df = build_dataset(features_csv, config=cfg)
    df = _ensure_id_column(df)

    with PipelineProgress(console) as pp:
        pp.set_total("embedding", len(df))
        # embed inside cluster_dataframe, so we update progress after
        cluster_result = cluster_dataframe(
            df,
            text_column="text_for_embedding",
            id_column="id",
            config=ClusterConfig(min_cluster_size=min_cluster_size),
        )
        pp.update_completed("embedding")
        pp.update_completed("clustering")

    df["cluster_id"] = df["id"].map(cluster_result.cluster_map)
    df["cluster_proba"] = cluster_result.probabilities
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    if dataset_out.suffix.lower() == ".parquet":
        df.to_parquet(dataset_out, index=False)
    else:
        df.to_csv(dataset_out, index=False)

    console.log(
        f"[green]Clustering done[/green]: {len(df)} rows, "
        f"{len(cluster_result.cluster_sizes)} clusters (incl noise) -> {dataset_out}"
    )


@app.command("summarize-clusters")
def summarize_clusters_cmd(
    dataset_path: Path = typer.Option(Path("data") / "dataset.parquet", help="Dataset with cluster_id."),
    clusters_out: Path = typer.Option(Path("data") / "clusters.csv", help="Output clusters CSV."),
):
    """Summarize clusters and write clusters.csv."""

    console = Console()
    if dataset_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    df = _ensure_id_column(df)

    if "cluster_id" not in df.columns:
        raise ValueError("dataset must contain 'cluster_id' column")

    # Build mapping id -> cluster_id
    cluster_map = dict(zip(df["id"].astype(str), df["cluster_id"].astype(str)))

    with PipelineProgress(console) as pp:
        pp.update_completed("cluster_summaries")
        summarize_cfg = ClusterSummaryConfig(summary_language="en")
        summarize_clusters(df, cluster_assignments=cluster_map, output_path=clusters_out, config=summarize_cfg)
        pp.update_completed("write_clusters")

    console.log(f"[green]Cluster summaries written[/green]: {clusters_out}")


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
    """Proxy to the UI reader for convenience."""

    from rss_ml.ui.reader import read_clusters as read_clusters_cli  # lazy import

    read_clusters_cli(
        clusters_csv=clusters_csv,
        min_size=min_size,
        language=language,
        query=query,
        sort_by=sort_by,
        descending=descending,
        page=page,
        page_size=page_size,
        show_summary_chars=show_summary_chars,
    )


@app.command("run")
def run_all(
    feeds_file: Path = typer.Option(Path("data") / "feeds.txt"),
    articles_out: Path = typer.Option(Path("data") / "articles_raw.csv"),
    features_out: Path = typer.Option(Path("data") / "articles_features.csv"),
    dataset_out: Path = typer.Option(Path("data") / "dataset.parquet"),
    clusters_out: Path = typer.Option(Path("data") / "clusters.csv"),
    max_entries: Optional[int] = typer.Option(None),
    per_feed_limit: Optional[int] = typer.Option(None),
    min_length: int = typer.Option(80),
    languages: Optional[List[str]] = typer.Option(None),
    min_cluster_size: int = typer.Option(2),
):
    """Run ingest -> featurize -> cluster -> summarize."""

    ingest(feeds_file=feeds_file, articles_out=articles_out, max_entries=max_entries, per_feed_limit=per_feed_limit)
    featurize(articles_csv=articles_out, features_out=features_out)
    cluster(
        features_csv=features_out,
        dataset_out=dataset_out,
        min_length=min_length,
        languages=languages,
        min_cluster_size=min_cluster_size,
    )
    summarize_clusters_cmd(dataset_path=dataset_out, clusters_out=clusters_out)


if __name__ == "__main__":
    app()
