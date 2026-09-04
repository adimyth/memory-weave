from __future__ import annotations

import os
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
    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]], **kwargs: Any) -> np.ndarray:
        self.calls.append(pairs)
        assert kwargs == {
            "apply_softmax": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        return self._scores
