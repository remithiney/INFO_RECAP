"""Language detection wrapper (fastText lid.176 or HF model)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:
    import fasttext  # type: ignore
except ImportError:  # pragma: no cover
    fasttext = None

try:
    from transformers import pipeline  # type: ignore
except ImportError:  # pragma: no cover
    pipeline = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class LanguageResult:
    language: str
    score: float


class LanguageDetector:
    """Detect languages with fastText lid.176 or HF fallback."""

    def __init__(
        self,
        fasttext_path: Path | str = Path("models") / "lid.176.bin",
        hf_model: str = "papluca/xlm-roberta-base-language-detection",
        device: Optional[int] = None,
        max_chars: int = 4000,
    ) -> None:
        self.fasttext_path = Path(fasttext_path)
        self.hf_model = hf_model
        self.device = device if device is not None else self._auto_device()
        self.max_chars = max_chars
        self._fasttext_model = None
        self._hf_pipeline = None
        self.backend = self._choose_backend()

    def _choose_backend(self) -> str:
        if fasttext and self.fasttext_path.exists():
            return "fasttext"
        if pipeline is not None:
            return "hf"
        return "heuristic"

    def _ensure_fasttext(self) -> None:
        if self._fasttext_model or not fasttext:
            return
        self._fasttext_model = fasttext.load_model(str(self.fasttext_path))

    def _ensure_hf(self) -> None:
        if self._hf_pipeline or pipeline is None:
            return
        self._hf_pipeline = pipeline(
            task="text-classification",
            model=self.hf_model,
            device=self.device if self.device is not None else -1,
            truncation=True,
            max_length=512,
        )

    def detect(self, texts: Sequence[str]) -> List[LanguageResult]:
        if self.backend == "fasttext":
            return self._detect_fasttext(texts)
        if self.backend == "hf":
            return self._detect_hf(texts)
        return self._detect_heuristic(texts)

    def _detect_fasttext(self, texts: Sequence[str]) -> List[LanguageResult]:
        self._ensure_fasttext()
        results: List[LanguageResult] = []
        for text in texts:
            if not text:
                results.append(LanguageResult(language="und", score=0.0))
                continue
            labels, scores = self._fasttext_model.predict(text)  # type: ignore[operator]
            lang = labels[0].replace("__label__", "") if labels else "und"
            score = float(scores[0]) if scores else 0.0
            results.append(LanguageResult(language=lang, score=score))
        return results

    def _detect_hf(self, texts: Sequence[str]) -> List[LanguageResult]:
        self._ensure_hf()
        trimmed = [t[: self.max_chars] if t else "" for t in texts]
        outputs = self._hf_pipeline(list(trimmed))  # type: ignore[operator]
        results: List[LanguageResult] = []
        for out in outputs:
            label = out.get("label", "und")
            score = float(out.get("score", 0.0))
            results.append(LanguageResult(language=label, score=score))
        return results

    def _detect_heuristic(self, texts: Sequence[str]) -> List[LanguageResult]:
        return [
            LanguageResult(language="und" if not text else "en", score=0.1)
            for text in texts
        ]

    @staticmethod
    def _auto_device() -> int:
        """Prefer GPU if available, else CPU (-1)."""

        if torch is not None and torch.cuda.is_available():
            return 0
        return -1


def detect_languages(texts: Iterable[str], detector: Optional[LanguageDetector] = None) -> List[LanguageResult]:
    """Detect languages for a batch of texts."""

    detector = detector or LanguageDetector()
    return detector.detect(list(texts))
