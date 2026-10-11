import numpy as np
import pytest

from pymoors import (
    InvalidParameterError,
    generational_distance,
    hypervolume,
    inverted_generational_distance,
    spacing,
    spread,
)


def test_hypervolume_single_point():
    front = np.array([[1.0, 1.0]])
    reference = np.array([3.0, 3.0])
    assert hypervolume(front, reference) == pytest.approx(4.0)


def test_hypervolume_dominated_point_contributes_nothing():
    front = np.array([[1.0, 1.0], [2.0, 2.0]])
    reference = np.array([3.0, 3.0])
    assert hypervolume(front, reference) == pytest.approx(4.0)


def test_hypervolume_raises_on_shape_mismatch():
    front = np.array([[1.0, 1.0]])
    reference = np.array([3.0, 3.0, 3.0])
    with pytest.raises(InvalidParameterError):
        hypervolume(front, reference)


def test_generational_distance_zero_when_subset():
    reference = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    front = np.array([[0.0, 0.0], [2.0, 2.0]])
    assert generational_distance(front, reference) == pytest.approx(0.0)


def test_inverted_generational_distance_penalizes_missing_coverage():
    reference = np.array([[0.0, 0.0], [10.0, 10.0]])
    front = np.array([[0.0, 0.0]])
    expected = np.sqrt(10.0**2 * 2.0) / 2.0
    assert inverted_generational_distance(front, reference) == pytest.approx(expected)


def test_spacing_uniform_front_is_zero():
    front = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
    assert spacing(front) == pytest.approx(0.0, abs=1e-9)


def test_spacing_raises_with_fewer_than_two_points():
    front = np.array([[0.0, 4.0]])
    with pytest.raises(InvalidParameterError):
        spacing(front)


def test_spread_uniform_front_matching_extremes_is_zero():
    front = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
    assert spread(front, front) == pytest.approx(0.0, abs=1e-9)


def test_spread_raises_for_three_objectives():
    front = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    with pytest.raises(InvalidParameterError):
        spread(front, front)


def test_spread_raises_with_fewer_than_two_points():
    front = np.array([[0.0, 4.0]])
    with pytest.raises(InvalidParameterError):
        spread(front, front)
