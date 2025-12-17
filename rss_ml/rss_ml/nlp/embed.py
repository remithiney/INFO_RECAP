"""Multilingual embeddings via sentence-transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None


@dataclass
class EmbedConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str | None = None  # e.g. "cpu" or "cuda"
    batch_size: int = 16
    normalize: bool = True


class Embedder:
    """Sentence-transformers embedder with lazy loading."""

    def __init__(self, config: EmbedConfig | None = None) -> None:
        self.config = config or EmbedConfig()
        self._model = None

    def _ensure_model(self):
        if self._model or SentenceTransformer is None:
            return
        self._model = SentenceTransformer(self.config.model_name, device=self.config.device)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if SentenceTransformer is None:
            return [[0.0, 0.0, 0.0] for _ in texts]
        self._ensure_model()
        return self._model.encode(  # type: ignore[operator]
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        ).tolist()


def embed_texts(texts: Iterable[str], embedder: Embedder | None = None) -> List[List[float]]:
    """Embed texts using a multilingual MiniLM model by default."""

    embedder = embedder or Embedder()
    return embedder.embed(list(texts))
