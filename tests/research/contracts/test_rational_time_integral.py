"""Dimensional, uncancelled coefficient bounds for smooth face integrands."""
import math

import numpy as np
import pytest

from research.experiments.material_top_band.rational_time_integral import RationalTimePolynomial


@pytest.mark.parametrize('q', [-.02, 0., .02])
def test_reciprocal_lambda_integral(q):
    function = RationalTimePolynomial([1.], q).over_lambda(1)
    result = function.integrate(.03)
    exact = .03 if q == 0. else .03 * math.log1p(q) / q
    assert abs(result['value'] - exact) <= result['roundoff'] + result['truncation']
    assert result['degree'] == 0 and result['power'] == 1
    assert result['truncation'] >= 0.


def test_uncancelled_scale_and_dimensional_tail():
    function = RationalTimePolynomial([1., 2.], .02)
    cancelled = (function - function).over_lambda(8)
    assert cancelled.integrate(.02)['scale'] > 0.
    base = function.over_lambda(8).integrate(.02)
    scaled = (function * 3000.).over_lambda(8).integrate(.02)
    expected_tail = math.comb(31, 7) * .02**24 / (1. - 32./25. * .02)
    assert base['truncation'] > 0.
    assert base['truncation'] == pytest.approx(2. * .02 * 3. * expected_tail, abs=0., rel=3e-15)
    assert scaled['truncation'] == pytest.approx(3000. * base['truncation'], abs=0., rel=3e-15)
    assert scaled['scale'] == pytest.approx(3000. * base['scale'], abs=0., rel=3e-15)
    longer = function.over_lambda(8).integrate(.04)
    assert longer['truncation'] == pytest.approx(2. * base['truncation'], abs=0., rel=3e-15)


def test_degree_and_power_are_executed_guards():
    with pytest.raises(ValueError, match='degree'):
        RationalTimePolynomial(np.ones(10), .02).integrate(.02)
    with pytest.raises(ValueError, match='power'):
        RationalTimePolynomial([1.], .02).over_lambda(9).integrate(.02)
    with pytest.raises(ValueError, match='frozen'):
        RationalTimePolynomial(np.ones(10), .02).integrate(.02, degree_limit=10)
    with pytest.raises(ValueError, match='majorant'):
        RationalTimePolynomial([1.], .02, majorant=[0.])


@pytest.mark.parametrize('bad', [np.nan, np.inf, 1j, True, .021])
def test_invalid_time_ratio(bad):
    with pytest.raises(ValueError):
        RationalTimePolynomial([1.], bad)


@pytest.mark.parametrize('coefficients', [[np.nan], [1j], np.ma.array([1.], mask=[True])])
def test_real_finite_unmasked_coefficients_required(coefficients):
    with pytest.raises(ValueError):
        RationalTimePolynomial(coefficients, .02)
