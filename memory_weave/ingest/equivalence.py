"""NLI-backed equivalence decisions for candidate memory records."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

import numpy as np

from memory_weave.config import EquivalenceConfig

EquivalenceVerdict = Literal["same", "contradicts", "distinct"]

# cross-encoder/nli-deberta-v3-small exposes [contradiction, entailment, neutral].
_DEFAULT_CONTRADICTION_LABEL = 0
_DEFAULT_ENTAILMENT_LABEL = 1
_NLI_LABEL_COUNT = 3
_VERDICTS: frozenset[EquivalenceVerdict] = frozenset({"same", "contradicts", "distinct"})


class EquivalenceJudge(Protocol):
    """Classify the relationship between two memory claims."""

    def judge(self, first: str, second: str) -> EquivalenceVerdict:
        """Return whether claims are the same, contradictory, or distinct."""

    def entails(self, premise: str, hypothesis: str) -> float:
        """Return how strongly one statement supports another statement."""


class FakeJudge:
    """Return configured verdicts and entailment scores for deterministic tests."""

    def __init__(
        self,
        verdicts: Mapping[tuple[str, str], EquivalenceVerdict] | None = None,
        entailments: Mapping[tuple[str, str], float] | None = None,
    ) -> None:
        self._verdicts: dict[frozenset[str], EquivalenceVerdict] = {}
        self._entailments = dict(entailments or {})
        self.entail_calls: list[tuple[str, str]] = []
        for (first, second), verdict in (verdicts or {}).items():
            self.set_verdict(first, second, verdict)

    def set_verdict(self, first: str, second: str, verdict: EquivalenceVerdict) -> None:
        """Set a symmetric verdict for one pair of claims."""

        if verdict not in _VERDICTS:
            raise ValueError(f"Unsupported equivalence verdict: {verdict}")
        self._verdicts[frozenset((first, second))] = verdict

    def judge(self, first: str, second: str) -> EquivalenceVerdict:
        """Return the configured symmetric verdict or ``distinct`` by default."""

        return self._verdicts.get(frozenset((first, second)), "distinct")

    def set_entailment(self, premise: str, hypothesis: str, score: float) -> None:
        """Set one directed entailment score for a test."""

        if not 0.0 <= score <= 1.0:
            raise ValueError("Entailment scores must be between 0 and 1.")
        self._entailments[(premise, hypothesis)] = score

    def entails(self, premise: str, hypothesis: str) -> float:
        """Return a configured score and otherwise treat the supplied evidence as supporting the claim."""

        self.entail_calls.append((premise, hypothesis))
        return self._entailments.get((premise, hypothesis), 1.0)


class NLICrossEncoderJudge:
    """Judge claim equivalence with a lazily loaded NLI cross-encoder."""

    def __init__(self, config: EquivalenceConfig, *, model_factory: Callable[[], Any] | None = None) -> None:
        self._entail_floor = config.entail_floor
        self._contradict_floor = config.contradict_floor
        self._model_factory = model_factory or _cross_encoder_factory(config.model)
        self._model: Any | None = None
        self._contradiction_label = _DEFAULT_CONTRADICTION_LABEL
        self._entailment_label = _DEFAULT_ENTAILMENT_LABEL
        self._model_load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Return whether the local cross-encoder has been instantiated."""

        with self._model_load_lock:
            return self._model is not None

    def judge(self, first: str, second: str) -> EquivalenceVerdict:
        """Score both claim directions and apply the configured NLI floors."""

        scores = self._predict([[first, second], [second, first]])
        if scores.shape != (2, _NLI_LABEL_COUNT):
            raise ValueError(f"NLI model returned shape {scores.shape}, expected (2, {_NLI_LABEL_COUNT}).")
        if not np.isfinite(scores).all():
            raise ValueError("NLI model scores must contain only finite values.")
        if np.all(scores[:, self._entailment_label] >= self._entail_floor):
            return "same"
        if np.any(scores[:, self._contradiction_label] >= self._contradict_floor):
            return "contradicts"
        return "distinct"

    def entails(self, premise: str, hypothesis: str) -> float:
        """Return the NLI entailment probability for a directed evidence-to-claim comparison."""

        scores = self._predict([[premise, hypothesis]])
        if scores.shape != (1, _NLI_LABEL_COUNT):
            raise ValueError(f"NLI model returned shape {scores.shape}, expected (1, {_NLI_LABEL_COUNT}).")
        return float(scores[0, self._entailment_label])

    def _predict(self, pairs: list[list[str]]) -> np.ndarray:
        model = self._load_model()
        with self._inference_lock:
            predicted = model.predict(pairs, apply_softmax=True, convert_to_numpy=True, show_progress_bar=False)
        scores = np.asarray(predicted, dtype=np.float32)
        if not np.isfinite(scores).all():
            raise ValueError("NLI model scores must contain only finite values.")
        return scores

    def _load_model(self) -> Any:
        with self._model_load_lock:
            if self._model is None:
                model = self._model_factory()
                contradiction_label, entailment_label = _nli_label_indices(model)
                self._model = model
                self._contradiction_label = contradiction_label
                self._entailment_label = entailment_label
            return self._model


def _nli_label_indices(model: Any) -> tuple[int, int]:
    """Resolve contradiction and entailment score positions from model metadata when available."""

    config = getattr(model, "config", None)
    if config is None:
        config = getattr(getattr(model, "model", None), "config", None)
    id2label = getattr(config, "id2label", None)
    if id2label is None:
        return _DEFAULT_CONTRADICTION_LABEL, _DEFAULT_ENTAILMENT_LABEL
    if not isinstance(id2label, Mapping):
        raise ValueError("NLI model config.id2label must be a mapping when it is present.")

    labels: dict[str, int] = {}
    for raw_index, raw_label in id2label.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"NLI model config.id2label has a non-integer label index: {raw_index!r}.") from exc
        label = str(raw_label).strip().lower()
        if label in labels:
            raise ValueError(f"NLI model config.id2label repeats the {label!r} label.")
        labels[label] = index

    try:
        contradiction_label = labels["contradiction"]
        entailment_label = labels["entailment"]
    except KeyError as exc:
        raise ValueError("NLI model config.id2label must name both 'contradiction' and 'entailment' labels.") from exc
    if not 0 <= contradiction_label < _NLI_LABEL_COUNT or not 0 <= entailment_label < _NLI_LABEL_COUNT:
        raise ValueError(f"NLI model config.id2label indices must be between 0 and {_NLI_LABEL_COUNT - 1}.")
    return contradiction_label, entailment_label


def _cross_encoder_factory(model_name: str) -> Callable[[], Any]:
    def load() -> Any:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:
            message = "NLI judging requires sentence-transformers. Install it with: uv sync --extra local-models"
            raise RuntimeError(message) from exc
        return CrossEncoder(model_name)

    return load
