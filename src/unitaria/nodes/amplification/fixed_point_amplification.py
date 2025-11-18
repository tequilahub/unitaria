from unitaria import Node
from unitaria.nodes import ProxyNode
from unitaria.nodes.amplification.poly_approx import sign_poly
from unitaria.nodes.qsvt.qsvt import QSVT


class FixedPointAmplification(ProxyNode):
    """
    A node that applies fixed point amplification to a vector node,
    i.e. it can amplify the norm of that vector node close to 1
    without knowing the exact value, given only a lower bound.
    This is achieved by using the QSVT and while it is more expensive
    than Grover-style amplification, there is no risk of "overshooting"
    and getting a lower norm.

    Implements Theorem 27 from https://arxiv.org/abs/1806.01838.

    :param A:
        The vector that should be amplified
    :param min_norm:
        A lower bound for the norm of this vector.
    :param accuracy:
        The maximum absolute error of this amplification
    :param guaranteed:
        Determines if the accuracy should be guaranteed using analytical
        bounds (ignoring numerical errors). If this is set to false, the
        function will return polynomials of lower degrees while usually
        still providing the required precision most of the time.
    """

    def __init__(self, A: Node, min_norm: float, accuracy: float, guaranteed: bool = False):
        assert A.is_vector()
        assert min_norm > 0.0
        assert accuracy > 0.0

        super().__init__(A.dimension_in, A.dimension_out)

        self.A = A
        self.poly = sign_poly(min_norm, accuracy, guaranteed)

    def definition(self) -> Node:
        return QSVT(A=self.A, coefficients=self.poly.coef, format="chebyshev")
