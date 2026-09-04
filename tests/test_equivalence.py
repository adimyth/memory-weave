from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from time import sleep
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from memory_weave.config import EquivalenceConfig
from memory_weave.ingest.equivalence import FakeJudge, NLICrossEncoderJudge


def test_fake_judge_uses_symmetric_configured_verdicts_and_a_distinct_default() -> None:
    judge = FakeJudge(
        {
            ("Aditya prefers concise answers.", "Aditya likes short replies."): "same",
            ("Aditya prefers concise answers.", "Aditya prefers detailed answers."): "contradicts",
        }
    )

    assert judge.judge("Aditya likes short replies.", "Aditya prefers concise answers.") == "same"
    assert judge.judge("Aditya prefers detailed answers.", "Aditya prefers concise answers.") == "contradicts"
    assert judge.judge("Aditya prefers concise answers.", "Aditya lives in Bangalore.") == "distinct"


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        (np.array([[0.01, 0.90, 0.09], [0.01, 0.85, 0.14]], dtype=np.float32), "same"),
        (np.array([[0.82, 0.10, 0.08], [0.01, 0.60, 0.39]], dtype=np.float32), "contradicts"),
        (np.array([[0.10, 0.60, 0.30], [0.20, 0.50, 0.30]], dtype=np.float32), "distinct"),
    ],
)
def test_nli_judge_loads_lazily_scores_both_directions_and_applies_floors(scores: np.ndarray, expected: str) -> None:
    model = _StubCrossEncoder(scores)
    judge = NLICrossEncoderJudge(
        EquivalenceConfig(entail_floor=0.80, contradict_floor=0.80),
        model_factory=lambda: model,
    )

    assert judge.is_loaded is False
    assert judge.judge("first claim", "second claim") == expected
    assert judge.is_loaded is True
    assert model.calls == [[["first claim", "second claim"], ["second claim", "first claim"]]]


def test_nli_judge_uses_model_label_metadata_when_the_score_order_is_permuted() -> None:
    model = _StubCrossEncoder(
        np.array([[0.90, 0.02, 0.08], [0.85, 0.01, 0.14]], dtype=np.float32),
        id2label={0: "entailment", 1: "contradiction", 2: "neutral"},
    )
    judge = NLICrossEncoderJudge(
        EquivalenceConfig(entail_floor=0.80, contradict_floor=0.80),
        model_factory=lambda: model,
    )

    assert judge.judge("first claim", "second claim") == "same"


def test_nli_judge_rejects_model_metadata_without_nli_label_names() -> None:
    model = _StubCrossEncoder(
        np.array([[0.90, 0.02, 0.08], [0.85, 0.01, 0.14]], dtype=np.float32),
        id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
    )
    judge = NLICrossEncoderJudge(EquivalenceConfig(), model_factory=lambda: model)

    with pytest.raises(ValueError, match="must name both 'contradiction' and 'entailment'"):
        judge.judge("first claim", "second claim")


def test_nli_judge_loads_one_model_when_two_threads_arrive_together() -> None:
    model = _StubCrossEncoder(np.array([[0.01, 0.90, 0.09], [0.01, 0.85, 0.14]], dtype=np.float32))
    factory_calls = 0
    factory_lock = Lock()

    def factory() -> _StubCrossEncoder:
        nonlocal factory_calls
        with factory_lock:
            factory_calls += 1
        sleep(0.02)
        return model

    judge = NLICrossEncoderJudge(EquivalenceConfig(), model_factory=factory)
    with ThreadPoolExecutor(max_workers=2) as executor:
        verdicts = list(executor.map(lambda _: judge.judge("first claim", "second claim"), range(2)))

    assert factory_calls == 1
    assert verdicts == ["same", "same"]


def test_nli_judge_serializes_model_inference() -> None:
    model = _BlockingCrossEncoder(np.array([[0.01, 0.90, 0.09], [0.01, 0.85, 0.14]], dtype=np.float32))
    judge = NLICrossEncoderJudge(EquivalenceConfig(), model_factory=lambda: model)
    second_started = Event()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(judge.judge, "first claim", "second claim")
        assert model.entered.wait(timeout=1)
        second = executor.submit(_judge_claims, judge, second_started)
        assert second_started.wait(timeout=1)
        try:
            assert model.overlap.wait(timeout=0.05) is False
        finally:
            model.release.set()
        assert first.result(timeout=1) == "same"
        assert second.result(timeout=1) == "same"

    assert model.max_active == 1


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("MEMORY_WEAVE_INTEGRATION") != "1",
    reason="set MEMORY_WEAVE_INTEGRATION=1 to run local-model integration tests",
)
def test_nli_judge_classifies_same_contradictory_and_distinct_claims() -> None:
    judge = NLICrossEncoderJudge(EquivalenceConfig())

    assert judge.judge("Aditya prefers concise answers.", "Aditya likes short replies.") == "same"
    assert judge.judge("Aditya prefers concise answers.", "Aditya prefers detailed answers.") == "contradicts"
    assert judge.judge("Aditya prefers concise answers.", "Aditya lives in Bangalore.") == "distinct"


class _StubCrossEncoder:
    def __init__(self, scores: np.ndarray, *, id2label: dict[int, str] | None = None) -> None:
        self._scores = scores
        self.calls: list[list[list[str]]] = []
        if id2label is not None:
            self.model = SimpleNamespace(config=SimpleNamespace(id2label=id2label))

    def predict(self, pairs: list[list[str]], **kwargs: Any) -> np.ndarray:
        self.calls.append(pairs)
        assert kwargs == {
            "apply_softmax": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        return self._scores


def _judge_claims(judge: NLICrossEncoderJudge, started: Event) -> str:
    started.set()
    return judge.judge("first claim", "second claim")


class _BlockingCrossEncoder(_StubCrossEncoder):
    def __init__(self, scores: np.ndarray) -> None:
        super().__init__(scores)
        self.entered = Event()
        self.overlap = Event()
        self.release = Event()
        self._active = 0
        self._active_lock = Lock()
        self.max_active = 0

    def predict(self, pairs: list[list[str]], **kwargs: Any) -> np.ndarray:
        with self._active_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if self._active > 1:
                self.overlap.set()
            self.entered.set()
        try:
            assert self.release.wait(timeout=1)
            return super().predict(pairs, **kwargs)
        finally:
            with self._active_lock:
                self._active -= 1
