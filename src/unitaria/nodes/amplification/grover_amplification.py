import numpy as np

from unitaria import Node
from unitaria.nodes import ProxyNode
from unitaria.nodes.qsvt.qsvt import QSVT


class GroverAmplification(ProxyNode):
    def __init__(self, A: Node, iterations: int):
        assert A.is_vector()
        assert iterations >= 0

        super().__init__(A.dimension_in, A.dimension_out)

        self.A = A
        self.iterations = iterations

    def definition(self) -> Node:
        # Angles for the Chebyshev polynomial
        angles = np.pi / 2 * np.ones(2 * self.iterations + 2)
        angles[0] = -self.iterations * np.pi / 2
        angles[-1] = -self.iterations * np.pi / 2

        # The Chebyshev polynomial has negative derivative at zero if
        # `self.iterations` is odd
        if self.iterations % 2 == 1:
            angles[0] += np.pi / 2
            angles[-1] += np.pi / 2

        return QSVT(A=self.A, coefficients=angles, format="angles")
