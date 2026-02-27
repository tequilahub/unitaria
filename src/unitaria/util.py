import numpy as np


def poly_sup_norm(poly, range=(-1.0, 1.0)):
    """
    Computes the sup norm of a polynomial in the range [-1, 1]
    """

    extrema = poly.deriv().roots()
    extrema = extrema[np.logical_and(extrema > range[0], extrema < range[1])]
    extrema = extrema[np.abs(np.imag(extrema)) < 1e-6]
    extrema = np.concatenate((extrema, [range[0], range[1]]))

    return np.max(np.abs(poly(extrema)))

