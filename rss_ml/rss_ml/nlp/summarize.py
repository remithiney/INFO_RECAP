"""Summarization using HF models with FR/EN selection and batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

try:
    from transformers import pipeline  # type: ignore
except ImportError:  # pragma: no cover
    pipeline = None


@dataclass
class SummarizeConfig:
    en_model: str = "facebook/bart-large-cnn"
    fr_model: str = "csebuetnlp/mT5_multilingual_XLSum"
    max_length: int = 180
    min_length: int = 40
    batch_size: int = 4
    device: Optional[int] = None  # -1 for CPU, int for GPU


class Summarizer:
    """Language-aware summarizer that batches inference."""

    def __init__(self, config: SummarizeConfig | None = None) -> None:
        self.config = config or SummarizeConfig()
        self._pipelines = {}

    def _get_pipeline(self, lang: str):
        if pipeline is None:
            return None
        key = "fr" if lang.startswith("fr") else "en"
        if key in self._pipelines:
            return self._pipelines[key]
        model_name = self.config.fr_model if key == "fr" else self.config.en_model
        self._pipelines[key] = pipeline(
            task="summarization",
            model=model_name,
            device=self.config.device if self.config.device is not None else -1,
        )
        return self._pipelines[key]

    def summarize(
        self,
        texts: Sequence[str],
        language: Optional[str] = None,
    ) -> List[str]:
        lang = language or "en"
        pipe = self._get_pipeline(lang)
        if pipe is None:  # transformers missing, fallback to truncate
            return [_truncate(t, self.config.max_length) for t in texts]

        summaries: List[str] = []
        bs = max(1, self.config.batch_size)
        for i in range(0, len(texts), bs):
            batch = list(texts[i : i + bs])
            try:
                outputs = pipe(
                    batch,
                    max_length=self.config.max_length,
                    min_length=self.config.min_length,
                    truncation=True,
                )
            except Exception:  # noqa: BLE001
                summaries.extend([_truncate(t, self.config.max_length) for t in batch])
                continue
            for out, original in zip(outputs, batch):
                summary = out.get("summary_text") if isinstance(out, dict) else None
                summaries.append(summary or _truncate(original, self.config.max_length))
        return summaries


def _truncate(text: str, max_len: int) -> str:
    if not text:
        return ""
    return text[:max_len]


def summarize_batch(
    texts: Iterable[str],
    language: Optional[str] = None,
    summarizer: Optional[Summarizer] = None,
) -> List[str]:
    """Summarize a batch of texts with language-specific models."""

    summarizer = summarizer or Summarizer()
    return summarizer.summarize(list(texts), language=language)
