"""Which scalar functions ode.py uses, and why.

`ode.py` ends with ``from numpy import *``, so every bare ``sqrt``,
``exp`` and ``isfinite`` in it is numpy's array version operating on a
single number. Swapping those for the ``math`` equivalents is tempting
(they are ten to fifty times faster on scalars) but only one of the three
is safe. These tests pin that reasoning down so it does not have to be
rediscovered.

"""
import math

import numpy as np
import pytest

from pytopkapi import ode


# ---- what ode.py should be bound to ---------------------------------

def test_isfinite_is_the_scalar_version():
    assert ode.isfinite is math.isfinite


def test_sqrt_and_exp_are_still_numpy():
    """Deliberate. The next two tests say why."""
    assert ode.sqrt is np.sqrt
    assert ode.exp is np.exp


# ---- why exp and sqrt were left alone --------------------------------

def test_math_exp_does_not_match_numpy_exp():
    """A guard against a well-meant future swap.

    ``exp`` is not required by IEEE 754 to be correctly rounded, and the
    two implementations disagree by one bit on a few per cent of inputs.
    One bit here feeds a non-linear reservoir and an adaptive stepper, so
    it does not stay one bit. If this test ever starts failing, the
    libraries have converged and the swap becomes available.
    """
    rng = np.random.default_rng(7)
    values = rng.uniform(-700.0, 700.0, size=200000)
    numpy_side = np.exp(values)
    math_side = np.array([math.exp(v) for v in values.tolist()])
    assert not np.array_equal(numpy_side, math_side)


def test_math_sqrt_raises_where_numpy_returns_nan():
    """The discriminant in ode.solution can go negative.

    numpy yields nan, which flows through to the isfinite guard in `qas`
    and triggers the Runge-Kutta fallback. math.sqrt would raise instead
    and take the run down.
    """
    assert math.isnan(np.sqrt(-1.0))
    with pytest.raises(ValueError):
        math.sqrt(-1.0)


def test_sqrt_itself_agrees_exactly():
    """For the record: sqrt is exact in both, so only the raising differs."""
    rng = np.random.default_rng(13)
    values = rng.uniform(0.0, 1e12, size=200000)
    assert np.array_equal(np.sqrt(values),
                          np.array([math.sqrt(v) for v in values.tolist()]))


# ---- the swap that was made, proved harmless -------------------------

SPECIAL = (0.0, -0.0, 1.0, -1.0, 5e-324, 1.7976931348623157e308,
           math.inf, -math.inf, math.nan)


@pytest.mark.parametrize('value', SPECIAL)
def test_isfinite_agrees_on_special_values(value):
    assert bool(math.isfinite(value)) is bool(np.isfinite(value))
    assert bool(math.isfinite(np.float64(value))) is \
        bool(np.isfinite(np.float64(value)))


def test_isfinite_agrees_over_a_sweep():
    rng = np.random.default_rng(11)
    values = np.concatenate([rng.uniform(-1e12, 1e12, 200000),
                             rng.uniform(-1e-300, 1e-300, 50000)])
    theirs = np.isfinite(values)
    ours = np.array([math.isfinite(v) for v in values.tolist()])
    assert np.array_equal(theirs, ours)


def test_qas_is_unchanged_by_the_swap():
    """Drive the quasi-analytical solver both ways and compare bits.

    The four-cell continuity fixture runs with solve_*=0, so it never
    reaches `qas`. This covers the path the real catchment takes.
    """
    rng = np.random.default_rng(20260910)
    cases = []
    for _ in range(20000):
        cases.append((float(10 ** rng.uniform(-6, 1)),      # a, inflow
                      float(10 ** rng.uniform(-8, 0)),      # b
                      float(rng.choice([1.6666667, 2.5, 3.0])),  # alpha
                      float(10 ** rng.uniform(-4, 4)),      # V0
                      3600.0))

    ours = [ode.qas(*case) for case in cases]
    ode.isfinite = np.isfinite
    try:
        theirs = [ode.qas(*case) for case in cases]
    finally:
        ode.isfinite = math.isfinite

    assert len(ours) == len(cases)
    assert sum(v is not None for v in ours) > 0.5 * len(cases), \
        'the sweep should mostly reach a solution, not bail early'
    for left, right in zip(ours, theirs):
        assert repr(left) == repr(right)


def test_adjust_2points_line_still_returns_the_same_pair():
    """The removed line built a closure and threw it away."""
    alpha0, beta0 = ode.adjust_2points_line(0.01, 1e-5, 2.5, 40.0, 3600.0)
    y1 = 40.0 + 3600.0 * 0.01
    expected_beta = (y1 ** 1.5 - 40.0 ** 1.5) / (y1 - 40.0)
    assert beta0 == expected_beta
    assert alpha0 == 40.0 ** 1.5 - expected_beta * 40.0
