from __future__ import annotations

import pytest

from memory_weave.util import Timer, normalize_alias, normalize_ws, uuid7


def test_timer_records_ordered_stage_durations_and_total() -> None:
    ticks = iter([1.0, 1.002, 1.007])
    timer = Timer(warm=True, clock=lambda: next(ticks))

    assert timer.mark("embed") == pytest.approx(2.0)
    assert timer.mark("dense") == pytest.approx(5.0)
    assert timer.as_dict() == {"embed": pytest.approx(2.0), "dense": pytest.approx(5.0), "total": pytest.approx(7.0)}
    assert timer.warm is True


def test_timer_rejects_duplicate_stage_names() -> None:
    ticks = iter([1.0, 1.001, 1.002])
    timer = Timer(warm=False, clock=lambda: next(ticks))
    timer.mark("embed")

    with pytest.raises(ValueError, match="already recorded"):
        timer.mark("embed")


def test_normalize_text_helpers() -> None:
    assert normalize_ws("  a\n\t  b  ") == "a b"
    assert normalize_alias("  Adítya  MISHRA ") == "aditya mishra"


def test_uuid7_values_sort_in_creation_order() -> None:
    values = [uuid7() for _ in range(100)]

    assert values == sorted(values)
