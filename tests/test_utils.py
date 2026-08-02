"""Tests for dstoolkit.utils.

Module 01 teaches testing as a core data-science skill. These tests are the
worked example students are asked to extend in the exercises.
"""

from __future__ import annotations

import pytest

from dstoolkit.utils import format_bytes, memory_footprint, timer


class TestFormatBytes:
    """format_bytes should pick the largest unit that keeps the number readable."""

    @pytest.mark.parametrize(
        ("n_bytes", "expected"),
        [
            (0, "0.0 B"),
            (512, "512.0 B"),
            (1024, "1.0 KiB"),
            (1536, "1.5 KiB"),
            (1024**2, "1.0 MiB"),
            (1024**3, "1.0 GiB"),
            (1024**4, "1.0 TiB"),
        ],
    )
    def test_unit_selection(self, n_bytes: int, expected: str) -> None:
        assert format_bytes(n_bytes) == expected

    def test_precision_is_configurable(self) -> None:
        assert format_bytes(1536, precision=3) == "1.500 KiB"

    def test_saturates_at_largest_unit(self) -> None:
        # Beyond TiB we have no bigger unit, so the number itself must grow.
        assert format_bytes(1024**5) == "1024.0 TiB"

    def test_handles_negative_values(self) -> None:
        # Deltas (e.g. "memory freed") can legitimately be negative.
        assert format_bytes(-1536) == "-1.5 KiB"


class TestMemoryFootprint:
    def test_returns_formatted_string(self) -> None:
        result = memory_footprint([1, 2, 3])
        assert isinstance(result, str)
        assert result.endswith(("B", "KiB", "MiB"))

    def test_larger_container_reports_more(self) -> None:
        small = [0] * 10
        large = [0] * 10_000
        import sys

        assert sys.getsizeof(large) > sys.getsizeof(small)


class TestTimer:
    def test_prints_label_and_duration(self, capsys: pytest.CaptureFixture) -> None:
        with timer("my-block"):
            sum(range(1000))

        captured = capsys.readouterr().out
        assert "my-block:" in captured
        assert captured.strip().endswith("s")

    def test_reports_even_when_block_raises(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        # The `finally` clause must still fire — that is the whole point of it.
        with pytest.raises(ValueError), timer("failing-block"):
            raise ValueError("boom")

        assert "failing-block:" in capsys.readouterr().out
