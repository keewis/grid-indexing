import pytest
import grid_indexes


def test_sum_as_string():
    assert grid_indexes.sum_as_string(1, 1) == "2"
