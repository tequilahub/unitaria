from unitaria import Node
from unitaria.nodes import ProxyNode
from unitaria.nodes.amplification.poly_approx import trunc_linear_poly
from unitaria.nodes.qsvt.qsvt import QSVT


class SingularValueAmplification(ProxyNode):
    """
    A node that uniformly amplifies all singular values of
    another node using the QSVT to improve the normalization
    of a block-encoding without changing anything else (up
    to approximation errors).

    Implements Theorem 30 from https://arxiv.org/abs/1806.01838.

    :param A:
        The node that should be amplified
    :param amplification:
        The factor by which the singular values should be multiplied.
    :param delta:
        The distance of the largest singular value after amplification
        from 1, i.e. the caller guarantees that, if x is the largest
        singular value of A, then x * amplification < 1 - delta, where
        delta must be positive.
    :param accuracy:
        The maximum absolute error of this amplification
    :param guaranteed:
        Determines if the accuracy should be guaranteed using analytical
        bounds (ignoring numerical errors). If this is set to false, the
        function will return polynomials of lower degrees while usually
        still providing the required precision most of the time.
    """

    def __init__(self, A: Node, amplification: float, delta: float, accuracy: float, guaranteed: bool = False):
        assert amplification > 1.0
        assert delta > 0.0
        assert accuracy > 0.0

        super().__init__(A.dimension_in, A.dimension_out)

        self.A = A
        self.poly = trunc_linear_poly(amplification, delta, accuracy, guaranteed)

    def definition(self) -> Node:
        return QSVT(A=self.A, coefficients=self.poly.coef, format="chebyshev")
