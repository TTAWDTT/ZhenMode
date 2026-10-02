"""Audited smooth time integrals with dimensional, uncancelled majorants."""
import math

import numpy as np
from numpy.polynomial import Polynomial

from .real_geometry import _array, _scalar


class RationalTimePolynomial:
    def __init__(self, coefficients, q, *, power=0, majorant=None):
        self.q = _scalar(q, 'alpha dt')
        if abs(self.q) > .02:
            raise ValueError('alpha dt exceeds frozen smooth range')
        coefficients = _array(coefficients, 'coefficients')
        if coefficients.ndim != 1 or not coefficients.size:
            raise ValueError('nonempty coefficient vector required')
        self.numerator = Polynomial(coefficients)
        majorant = abs(coefficients) if majorant is None else _array(majorant, 'majorant')
        if majorant.ndim != 1 or majorant.size < coefficients.size or np.any(majorant[:coefficients.size] < abs(coefficients)):
            raise ValueError('coefficient majorant must cover actual numerator')
        self.majorant = Polynomial(majorant)
        if np.any(self.majorant.coef < 0) or not np.isfinite(self.majorant.coef).all():
            raise ValueError('finite nonnegative coefficient majorant required')
        if isinstance(power, bool) or not isinstance(power, int) or power < 0:
            raise ValueError('nonnegative integral power required')
        self.power = power
        self.numerator.coef.setflags(write=False)
        self.majorant.coef.setflags(write=False)

    def _coerce(self, other):
        if isinstance(other, RationalTimePolynomial):
            if other.q != self.q:
                raise ValueError('different time ratios')
            return other
        return RationalTimePolynomial([_scalar(other, 'scalar')], self.q)

    def __add__(self, other):
        other = self._coerce(other)
        power = max(self.power, other.power)
        lam, major = Polynomial([1., self.q]), Polynomial([1., abs(self.q)])
        numerator = self.numerator * lam**(power-self.power) + other.numerator * lam**(power-other.power)
        scale = self.majorant * major**(power-self.power) + other.majorant * major**(power-other.power)
        return RationalTimePolynomial(numerator.coef, self.q, power=power, majorant=scale.coef)

    __radd__ = __add__

    def __neg__(self):
        return RationalTimePolynomial(-self.numerator.coef, self.q, power=self.power, majorant=self.majorant.coef)

    def __sub__(self, other):
        return self + -self._coerce(other)

    def __rsub__(self, other):
        return self._coerce(other) + -self

    def __mul__(self, other):
        other = self._coerce(other)
        return RationalTimePolynomial((self.numerator * other.numerator).coef, self.q, power=self.power+other.power,
                                      majorant=(self.majorant * other.majorant).coef)

    __rmul__ = __mul__

    def over_lambda(self, power=1):
        return RationalTimePolynomial(self.numerator.coef, self.q, power=self.power+power, majorant=self.majorant.coef)

    def integrate(self, duration, *, degree_limit=8, power_limit=8):
        dt = _scalar(duration, 'duration', positive=True)
        if any(isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 8 for limit in (degree_limit, power_limit)):
            raise ValueError('degree and power guards cannot exceed frozen limits')
        degree = max(self.numerator.degree(), self.majorant.degree())
        if degree > degree_limit:
            raise ValueError('executed numerator degree exceeds protocol')
        if self.power > power_limit:
            raise ValueError('executed denominator power exceeds protocol')
        nodes, weights = np.polynomial.legendre.leggauss(16)
        y, weights = .5 * (nodes + 1.), .5 * weights
        value = dt * math.fsum(weights * self.numerator(y) / (1. + self.q * y)**self.power)
        coefficient_scale = float(np.sum(self.majorant.coef))
        scale = dt * coefficient_scale / (1. - abs(self.q))**self.power
        tail = math.comb(31, 7) * abs(self.q)**24 / (1. - 32./25. * abs(self.q))
        truncation = dt * coefficient_scale * (2. * tail)
        if not all(math.isfinite(number) for number in (value, scale, coefficient_scale, tail, truncation)):
            raise ValueError('finite integral and coefficient scale required')
        return dict(value=value, scale=scale, roundoff=512. * np.finfo(float).eps * max(1., scale),
                    truncation=truncation, degree=int(degree), power=self.power)
