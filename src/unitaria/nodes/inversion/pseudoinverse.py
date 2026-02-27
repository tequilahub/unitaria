import numpy as np

from unitaria.nodes import Node
from unitaria.nodes import ProxyNode
from unitaria.nodes.basic.adjoint import Adjoint
from unitaria.nodes.qsvt.qsvt import QSVT
from unitaria.poly_approx import inverse_poly


class Pseudoinverse(ProxyNode):
    """
    This node implements the Moore-Penrose pseudoinverse of ``A`` with the
    given tolerance, if `condition` is the ratio between ``A.normalization`` and
    ``A``s smallest singular value.

    Implements Theorem 41 from https://arxiv.org/abs/1806.01838

    :param A:
        The node to be inverted
    :param condition:
        An upper bound on the inverse of nonzero singular values of A. This is
        the same as the usual definition of the condition of a matrix iff the
        largest singular value is its normalization, i.e. if the block-encoding
        has optimal subnormalization.
    :param tolerance:
        The absolute error tolerance
    :param guaranteed:
        Determines if the accuracy should be guaranteed using analytical
        bounds (ignoring numerical errors). If this is set to false, this
        function will use a heuristic which will result in polynomials of
        lower degrees while usually still providing the requested precision.
    """

    # TODO: The condition should actually be the condition of the matrix,
    #   the dependence on the subnormalization should be hidden from the user.
    def __init__(self, A: Node, condition: float, tolerance: float, guaranteed: bool = False):
        super().__init__(A.dimension_in, A.dimension_out)
        assert condition >= 1
        assert 0 < tolerance < 0.5
        self.A = A
        self.condition = condition
        self.tolerance = tolerance
        self.guaranteed = guaranteed

        # This is to make certain assumptions fit.
        # It is ok to make the condition larger and the error tolerance smaller
        # than what the user supplied
        condition = max(condition, 4)
        tolerance = min(tolerance, 2 / condition)

        assert 0 < tolerance <= 2 / condition <= 0.5

        self.poly = inverse_poly(1 / condition, tolerance, False) * condition

        # Rescale the polynomial, so that it fits the input normalization
        X = np.polynomial.Chebyshev([0, 1])
        self.poly = self.poly(X / A.normalization) / A.normalization

    def definition(self):
        return QSVT(A=Adjoint(self.A), coefficients=self.poly.coef, format="chebyshev")

    def children(self) -> list[Node]:
        return [self.A]

    def parameters(self) -> dict:
        return {"condition": self.condition, "tolerance": self.tolerance, "guaranteed": self.guaranteed}
