import numpy as np
import scipy.special
from numpy.polynomial.chebyshev import Chebyshev


def erf_poly(k: float, epsilon: float, guaranteed: bool = False) -> Chebyshev:
    """
    For k > 0, 0 < epsilon <= 1 / 5, returns a polynomial in the Chebyshev
    basis that approximates the function erf(k * x) with maximum absolute
    error epsilon on the interval [-1, 1].

    Implements corollary 4 from https://arxiv.org/abs/1707.05391

    :param k: The multiplicative factor for the argument of the error function, requires k > 0
    :param epsilon: The required absolute error bound, requires 0 < epsilon <= 1 / 5
    :param guaranteed: If the accuracy should be guaranteed using analytical bounds (ignoring numerical errors).
        If this is set to false, the function will return polynomials of lower degrees.
    :return: The polynomial approximating erf(k * x)
    """
    assert k > 0
    assert epsilon <= 1 / 5
    epsilon *= np.sqrt(np.pi) * np.e / 2
    max_j = np.sqrt(2 * np.ceil(np.maximum(k**2 / 2 * np.e**2, np.log(2 / epsilon))) * np.log(4 / epsilon))
    if not guaranteed:
        # The analytical bounds generally yield higher accuracies than required. If the bounds don't
        # need to be guaranteed, this heuristic reduces the degree of the polynomial while usually still
        # providing the required precision most of the time.
        max_j *= 0.4
    degree = 2 * int(np.ceil(max_j)) + 1
    coefficients = np.zeros(degree + 1)

    coefficients[1] += scipy.special.i0e(k**2 / 2)

    for j in range(1, (degree - 1) // 2 + 1):
        factor = scipy.special.ive(j, k**2 / 2) * (-1) ** j
        coefficients[2 * j + 1] += factor / (2 * j + 1)
        coefficients[2 * j - 1] -= factor / (2 * j - 1)

    coefficients *= 2 * k / np.sqrt(np.pi)

    if np.any(np.isnan(coefficients)):
        raise RuntimeError("nan")

    return Chebyshev(coefficients)


def sign_poly(delta: float, epsilon: float, guaranteed: bool = False) -> Chebyshev:
    """
    For delta > 0, 0 < epsilon <= 2 / 5, returns a polynomial in the Chebyshev basis
    that approximates the sign function sign(x) with maximum absolute error epsilon
    in the intervals [-domain, -delta] and [delta, domain].

    Implements Lemma 25 from https://arxiv.org/abs/1806.01838,
    based on Corollary 6 from https://arxiv.org/abs/1707.05391.

    :param delta: The size of the region (-delta, delta) in which the approximation
        need not hold, requires delta > 0
    :param epsilon: The maximum absolute error of the approximation, requires 0 < epsilon <= 2 / 5
    :param guaranteed: If the accuracy should be guaranteed using analytical bounds (ignoring numerical errors).
        If this is set to false, the function will return polynomials of lower degrees.
    :return: The polynomial approximating sign(x)
    """
    k = 1 / (np.sqrt(2) * delta) * np.sqrt(np.log(2 / (np.pi * (epsilon / 2) ** 2)))
    return erf_poly(k, epsilon / 2, guaranteed)


def rect_poly(t: float, delta: float, epsilon: float, guaranteed: bool = False) -> Chebyshev:
    """
    For t >= 0, delta > 0 and 0 < epsilon <= 2 / 5, returns a polynomial P in the
    Chebyshev basis that approximates the rectangle function such that |P(x)| <= epsilon
    for x in [-1, -width - delta] and x in [width + delta, 1] and |P(x) - 1| <= epsilon
    for x in [-width + delta, width - delta].

    Implements Lemma 29 from https://arxiv.org/abs/1806.01838.

    :param t: The scaling of the rectangle function, requires t >= 0
    :param delta: The size of the regions [+-width - delta, +-width + delta] in which the approximation
        need not hold, requires delta > 0
    :param epsilon: The maximum absolute error of the approximation, requires 0 < epsilon <= 2 / 5
    :param guaranteed: If the accuracy should be guaranteed using analytical bounds (ignoring numerical errors).
        If this is set to false, the function will return polynomials of lower degrees.
    :return: The polynomial approximating rect(x / w)
    """
    k = 1 / (np.sqrt(2) * delta) * np.sqrt(np.log(2 / (np.pi * (epsilon / 2) ** 2)))
    base = erf_poly(k * (1 + t), epsilon / 2, guaranteed)
    p1 = base(Chebyshev(-1 / (1 + t) * np.array([-t, 1])))
    p2 = base(Chebyshev(1 / (1 + t) * np.array([t, 1])))
    return (p1 + p2) / 2


def trunc_linear_poly(gamma: float, delta: float, epsilon: float, guaranteed: bool = False) -> Chebyshev:
    """
    For gamma >= 1, delta > 0, 0 < epsilon <= 2 / 5, returns a polynomial P in the
    Chebyshev basis that approximates the truncated linear function gamma * x
    maximum relative error epsilon on the interval [-(1 - delta) / gamma, (1 - delta) / gamma]
    while being bounded by 1 on the entire interval [-1, 1].

    Implements the polynomial used in the proof of Theorem 30 from https://arxiv.org/abs/1806.01838.

    :param gamma: The gradient of the truncated linear function around the origin, requires gamma >= 1
    :param delta: The distance from the limit 1 / gamma until which the approximation needs to hold, requires delta > 0
    :param epsilon: The maximum relative error of the approximation, requires 0 < epsilon <= 2 / 5
    :param guaranteed: If the accuracy should be guaranteed using analytical bounds (ignoring numerical errors).
        If this is set to false, the function will return polynomials of lower degrees.
    :return: The polynomial approximating the truncated linear function gamma * x
    """
    t = (1 - delta / 2) / gamma
    delta = delta / (2 * gamma)
    epsilon = epsilon / gamma

    x = Chebyshev([0, 1])
    if guaranteed:
        return gamma * x * (1 - epsilon / 2) * rect_poly(t, delta, epsilon / 2, guaranteed)
    else:
        return gamma * x * rect_poly(t, delta, epsilon, guaranteed)


def _unscaled_inverse_poly(kappa: float, epsilon: float) -> Chebyshev:
    """
    For kappa > 1 and 0 < epsilon < 1, returns a polynomial in the Chebyshev basis
    that approximates the function 1 / x on the interval [-1, 1] \ (-1 / kappa, 1 / kappa).

    Implements Lemma 40 from https://arxiv.org/abs/1806.01838

    :param kappa: The approximation is only valid for |x| >= 1 / kappa, requires kappa > 1.
    :param epsilon: The maximum error of the approximation, requires 0 < epsilon < 1.
    :return: The polynomial approximating 1 / x
    """
    assert kappa > 1
    assert 0 < epsilon < 1

    epsilon /= 2  # because we have two epsilon / 2 approximations
    b = int(np.ceil(kappa**2 * np.log(kappa / epsilon)))
    j = int(np.ceil(np.sqrt(b * np.log(4 * b / epsilon))))
    coefficients = np.zeros(2 * j + 2)
    for i in range(min(b, j) + 1):
        coefficients[2 * i + 1] += (-1) ** i * 4 * scipy.special.bdtrc(b + i, 2 * b, 0.5)
    return Chebyshev(coefficients)


def inverse_poly(delta: float, epsilon: float, guaranteed: bool = False) -> Chebyshev:
    """
    For 0 < epsilon < 2 * delta <= 1 / 2, returns a polynomial P which epsilon-approximates
    delta / x on the domain [-1, 1] \ (-delta, delta) and for which |P| <= 1.

    Implements Theorem 41 from https://arxiv.org/abs/1806.01838, but with delta scaled by 1 / 2.

    :param delta: The approximation is only valid for |x| >= delta, requires epsilon / 2 < delta <= 1 / 4.
    :param epsilon: The maximum error of the approximation, requires 0 < epsilon < 2 * delta.
    :param guaranteed: If the accuracy should be guaranteed using analytical bounds (ignoring numerical errors).
        If this is set to false, the function will return polynomials of lower degrees.
    :return: The polynomial approximating delta / x
    """
    assert 0 < epsilon < 2 * delta <= 1 / 2

    poly = delta * _unscaled_inverse_poly(1 / delta, (epsilon / 3) / delta)

    maxima = poly.deriv().roots()
    maxima = maxima[np.abs((np.imag(maxima)) < 1e-6) & (np.abs(maxima) <= 1)]
    pmax = np.max(np.abs(poly(np.concatenate((maxima, [-1, 1])))))

    rect = rect_poly(delta / 2, delta / 2, min(epsilon / 3, 1 / pmax), guaranteed)

    return poly * (1 - rect)
