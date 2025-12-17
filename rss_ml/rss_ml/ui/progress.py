"""Helpers for Rich progress displays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn, TimeElapsedColumn


@dataclass(frozen=True)
class PipelineTasks:
    """Named tasks used across the pipeline."""

    load_feeds: str = "Load feeds"
    fetch_entries: str = "Fetch entries"
    extract_content: str = "Extract content"
    write_articles: str = "Write articles CSV"
    nlp_language: str = "NLP: language"
    nlp_summarize: str = "NLP: summarize"
    nlp_classify: str = "NLP: classify"
    embedding: str = "Embeddings"
    clustering: str = "Clustering"
    cluster_summaries: str = "Cluster summaries"
    write_clusters: str = "Write clusters CSV"


def create_progress(console: Console | None = None) -> Progress:
    """Create a consistent Rich Progress instance."""

    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def add_pipeline_tasks(progress: Progress) -> Dict[str, TaskID]:
    """Register a single shared task, keyed for compatibility."""

    labels = PipelineTasks()
    task_id = progress.add_task("Starting", total=1)
    return {key: task_id for key in labels.__dict__.keys()}


class PipelineProgress:
    """Helper to track pipeline tasks with Rich progress."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()
        self.progress = create_progress(self.console)
        self.tasks = add_pipeline_tasks(self.progress)
        self.labels = PipelineTasks()

    def set_total(self, task_key: str, total: int) -> None:
        if task_key in self.tasks:
            desc = getattr(self.labels, task_key, task_key)
            self.progress.reset(self.tasks[task_key], total=total, completed=0, description=desc)

    def advance(self, task_key: str, advance: int = 1) -> None:
        if task_key in self.tasks:
            self.progress.advance(self.tasks[task_key], advance)

    def update_completed(self, task_key: str) -> None:
        if task_key in self.tasks:
            self.progress.update(self.tasks[task_key], completed=self.progress.tasks[self.tasks[task_key]].total)

    def __enter__(self) -> "PipelineProgress":
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.progress.__exit__(exc_type, exc, tb)


def pipeline_progress(console: Optional[Console] = None) -> Iterator[PipelineProgress]:
    """Context manager yielding a PipelineProgress with pre-registered tasks."""

    with PipelineProgress(console) as pp:
        yield pp
