import numpy as np

from unitaria import Node
from unitaria.nodes import ProxyNode
from unitaria.nodes.qsvt.qsvt import QSVT


class GroverAmplification(ProxyNode):
    def __init__(self, A: Node, iterations: int):
        assert A.is_vector()
        assert iterations > 0

        super().__init__(A.dimension_in, A.dimension_out)

        self.A = A
        self.iterations = iterations

    def definition(self) -> Node:
        # Choose coefficients such that they become [pi, ..., pi] in Wx basis
        coefficients = np.append(
            np.full(2 * self.iterations + 1, np.pi, dtype=np.complex128), [-(2 * self.iterations + 2) / 2 * np.pi]
        )
        return QSVT(A=self.A, coefficients=coefficients, format="angles")
