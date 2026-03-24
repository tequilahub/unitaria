from unitaria.nodes import Node
from unitaria.nodes import ProxyNode
from unitaria.nodes.basic.adjoint import Adjoint
from unitaria.nodes.basic.scale import Scale
from unitaria.nodes.qsvt.qsvt import QSVT
from unitaria.poly_approx import inverse_poly


class Pseudoinverse(ProxyNode):
    """
    For 0 < epsilon <= 2 * delta <= 0.5, assuming the smallest nonzero singular
    value of A is at least delta, this node implements the Moore-Penrose
    pseudoinverse of A.

    Implements Theorem 41 from https://arxiv.org/abs/1806.01838

    :param A:
        The node to be inverted
    :param condition:
        An upper bound on the inverse of nonzero singular values of A.
        This is the same as the usual definition of the condition of a matrix iff
        the largest singular value is one, i.e. if the block-encoding has optimal
        subnormalization.
    :param epsilon:
        The maximum absolute error
    :param guaranteed:
        Determines if the accuracy should be guaranteed using analytical
        bounds (ignoring numerical errors). If this is set to false, this
        function will use a heuristic which will result in polynomials of
        lower degrees while usually still providing the requested precision.
    """

    # TODO: The condition should actually be the condition of the matrix,
    #   the dependence on the subnormalization should be hidden from the user.
    def __init__(self, A: Node, condition: float, epsilon: float, guaranteed: bool = False):
        assert 0 < epsilon <= 2 / condition <= 0.5

        super().__init__(A.dimension_in, A.dimension_out)

        self.A = A
        self.condition = condition
        self.poly = inverse_poly(1 / condition, epsilon, guaranteed)

    def definition(self):
        return Scale(QSVT(A=Adjoint(self.A), coefficients=self.poly.coef, format="chebyshev"), self.condition)
