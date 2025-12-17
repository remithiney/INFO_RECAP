"""Zero-shot or finetuned classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

try:
    from transformers import pipeline  # type: ignore
except ImportError:  # pragma: no cover
    pipeline = None


DEFAULT_LABELS = ["announcement", "review", "update", "other"]


@dataclass
class ClassificationResult:
    label: str
    score: float


@dataclass
class ClassifyConfig:
    zero_shot_model: str = "facebook/bart-large-mnli"
    multi_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    labels: Sequence[str] = tuple(DEFAULT_LABELS)
    hypothesis_template: str = "This text is a {}."
    device: Optional[int] = None  # -1 CPU, int GPU
    batch_size: int = 4
    use_multilingual: bool = True


class Classifier:
    """Zero-shot classifier with optional multilingual model."""

    def __init__(self, config: ClassifyConfig | None = None) -> None:
        self.config = config or ClassifyConfig()
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline or pipeline is None:
            return
        model_name = self.config.multi_model if self.config.use_multilingual else self.config.zero_shot_model
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.config.device if self.config.device is not None else -1,
        )

    def classify(self, texts: Sequence[str]) -> List[ClassificationResult]:
        if pipeline is None:
            return [ClassificationResult(label="other" if t else "announcement", score=0.25) for t in texts]

        self._ensure_pipeline()
        results: List[ClassificationResult] = []
        bs = max(1, self.config.batch_size)
        labels = list(self.config.labels)

        for i in range(0, len(texts), bs):
            batch = list(texts[i : i + bs])
            try:
                outputs = self._pipeline(  # type: ignore[operator]
                    batch,
                    candidate_labels=labels,
                    hypothesis_template=self.config.hypothesis_template,
                    multi_label=False,
                )
            except Exception:  # noqa: BLE001
                results.extend([ClassificationResult(label="other", score=0.0) for _ in batch])
                continue

            # transformers returns dict or list depending on input size
            if isinstance(outputs, dict):
                outputs = [outputs]

            for out in outputs:
                lbls = out.get("labels", [])
                scores = out.get("scores", [])
                label = lbls[0] if lbls else "other"
                score = float(scores[0]) if scores else 0.0
                results.append(ClassificationResult(label=label, score=score))

        return results


def classify_batch(
    texts: Iterable[str],
    classifier: Optional[Classifier] = None,
) -> List[ClassificationResult]:
    """Classify a batch of texts using zero-shot (or fallback stub)."""

    classifier = classifier or Classifier()
    return classifier.classify(list(texts))
