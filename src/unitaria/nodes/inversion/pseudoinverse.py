from unitaria import Node
from unitaria.nodes import ProxyNode
from unitaria.nodes.basic.scale import Scale
from unitaria.nodes.qsvt.qsvt import QSVT
from unitaria.poly_approx import inverse_poly


class Pseudoinverse(ProxyNode):
    """
    Implements Theorem 41 from https://arxiv.org/abs/1806.01838
    """

    def __init__(self, A: Node, delta: float, epsilon: float, guaranteed: bool = False):
        # TODO: Recheck these bounds, and also the bounds in the poly_approx functions
        assert 0 < epsilon <= delta <= 0.25

        super().__init__(A.dimension_in, A.dimension_out)

        self.A = A
        self.delta = delta
        self.poly = inverse_poly(delta, epsilon, guaranteed)

    def definition(self):
        return Scale(QSVT(A=self.A, coefficients=self.poly.coef, format="chebyshev"), 1 / self.delta)
