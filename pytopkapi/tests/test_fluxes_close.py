"""Equivalence tests for fluxes._close against numpy.isclose.

The scalar form replaces np.isclose in the hottest line of the solver, so
it has to agree with numpy on every input the model can produce, not
merely on typical ones.

"""
import math

import numpy as np
import pytest

from pytopkapi.fluxes import _close, Qout_computing

SPECIAL = (0.0, -0.0, 1.0, -1.0, 1e-30, -1e-30, 1e-8, 1e-5,
           1e30, -1e30, 5e-324, 1.7976931348623157e308,
           math.inf, -math.inf, math.nan)


@pytest.mark.parametrize('a', SPECIAL)
@pytest.mark.parametrize('b', SPECIAL)
def test_agrees_with_numpy_on_special_values(a, b):
    assert bool(_close(a, b)) is bool(np.isclose(a, b))


def test_agrees_with_numpy_on_a_random_sweep():
    rng = np.random.default_rng(20260910)
    # magnitudes spanning the range of volumes and flows in a real run
    scale = 10.0 ** rng.uniform(-12, 9, size=20000)
    left = scale * rng.normal(size=20000)
    # right sits a controlled relative distance away, straddling the
    # 1e-5 threshold where the two forms could disagree
    offset = 10.0 ** rng.uniform(-9, -2, size=20000)
    right = left * (1.0 + offset * rng.choice([-1.0, 1.0], size=20000))

    for a, b in zip(left.tolist(), right.tolist()):
        assert bool(_close(a, b)) is bool(np.isclose(a, b))


def test_agrees_with_numpy_exactly_on_the_threshold():
    """The boundary case: b displaced by precisely atol + rtol*abs(b)."""
    for b in (1.0, -3.5, 1e-6, 4e7):
        edge = 1e-8 + 1e-5 * abs(b)
        for a in (b + edge, b - edge,
                  b + np.nextafter(edge, np.inf),
                  b - np.nextafter(edge, np.inf)):
            assert bool(_close(a, b)) is bool(np.isclose(a, b))


def test_numpy_scalars_behave_like_python_floats():
    a = np.float64(1.0)
    b = np.float64(1.0 + 1e-9)
    assert bool(_close(a, b)) is bool(np.isclose(a, b))
    assert bool(_close(float(a), float(b))) is bool(_close(a, b))


def test_arrays_fall_back_to_numpy():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.5, 3.0])
    assert np.array_equal(_close(a, b), np.isclose(a, b))


def test_nan_is_close_to_nothing():
    assert bool(_close(math.nan, math.nan)) is False
    assert bool(_close(math.nan, 1.0)) is False


def test_infinity_is_close_only_to_itself():
    assert bool(_close(math.inf, math.inf)) is True
    assert bool(_close(math.inf, -math.inf)) is False
    assert bool(_close(math.inf, 1e308)) is False


def test_qout_still_zeroes_a_balanced_store():
    """The behaviour the call site depends on, stated directly."""
    assert Qout_computing(100.0, 100.0 + 2.0 * 3600.0, 2.0, 3600.0) == 0
    assert Qout_computing(100.0, 100.0, 2.0, 3600.0) == pytest.approx(2.0)
