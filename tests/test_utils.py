"""Tests for utils.utils helpers (get_region, rgetattr)."""

import pytest

from utils.utils import get_region, rgetattr


class TestGetRegion:
    """get_region maps (monkey, array index) -> brain region.

    aggregate.collect_results depends on this mapping, so it is worth
    pinning down directly.
    """

    @pytest.mark.parametrize(
        "array, expected",
        [(0, "V1"), (7, "V1"), (8, "IT"), (12, "IT"), (13, "V4"), (15, "V4")],
    )
    def test_monkeyF_ranges(self, array, expected):
        assert get_region("monkeyF", array) == expected

    @pytest.mark.parametrize(
        "array, expected",
        [(0, "V1"), (7, "V1"), (8, "V4"), (11, "V4"), (12, "IT"), (15, "IT")],
    )
    def test_monkeyN_ranges(self, array, expected):
        assert get_region("monkeyN", array) == expected

    def test_monkeyF_and_monkeyN_differ(self):
        # Array 8 is IT for monkeyF but V4 for monkeyN -- the mapping is
        # per-monkey, not shared.
        assert get_region("monkeyF", 8) == "IT"
        assert get_region("monkeyN", 8) == "V4"

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="Invalid array index"):
            get_region("monkeyF", 16)

    def test_unknown_monkey_raises(self):
        # Unknown monkey -> empty mapping -> no range matches -> ValueError.
        with pytest.raises(ValueError, match="Invalid array index"):
            get_region("monkeyX", 0)


class TestRgetattr:
    """rgetattr resolves a whitespace-separated attribute path with a default.

    Note: the current implementation splits on whitespace (path.split()),
    not on ".", so these tests document the actual behavior.
    """

    class _Node:
        pass

    def _nested(self):
        root = self._Node()
        root.child = self._Node()
        root.child.value = 42
        return root

    def test_single_attribute(self):
        obj = self._Node()
        obj.value = 7
        assert rgetattr(obj, "value", default=-1) == 7

    def test_nested_attribute_whitespace_path(self):
        root = self._nested()
        assert rgetattr(root, "child value", default=-1) == 42

    def test_missing_attribute_returns_default(self):
        obj = self._Node()
        assert rgetattr(obj, "does_not_exist", default="fallback") == "fallback"

    def test_missing_nested_attribute_returns_default(self):
        root = self._nested()
        assert rgetattr(root, "child missing", default=None) is None
