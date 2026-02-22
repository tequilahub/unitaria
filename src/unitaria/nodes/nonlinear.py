from typing import Sequence

import numpy as np
import tequila as tq

from unitaria.nodes.node import Node
from unitaria.nodes.multilinear_node import MultilinearNode
from unitaria.subspace import Subspace
from unitaria.circuit import Circuit


class ComponentwiseMul(MultilinearNode):
    """
    Node implementing the (bilinear) componentwise multiplication operator

    More specifically this implements the bilinear map ``(x, y) -> [x_1 * y_1,
    ..., x_n * y_n]``. Usually you will want the elementwise product of two
    vectors, in which case the correct result will be obtained by building the
    tensor product of the vectors and then multiplying it with this operation,
    i.e. ``Mul(Tensor(a, b), ComponentwiseMul(a.subspace_out()))``

    :param subspace:
        The vector space in which to perform the element-wise operation
    """

    subspace: Subspace

    def __init__(self, subspace_or_first: Subspace | Node, second: Node | None = None):
        if isinstance(subspace_or_first, Subspace):
            self.subspace = subspace_or_first
            super().__init__([self.subspace.dimension] * 2, self.subspace.dimension)
        else:
            self.subspace = subspace_or_first.subspace_out
            super().__init__([self.subspace.dimension] * 2, self.subspace.dimension, subspace_or_first, second)

    def definition(self) -> Node:
        return ComponentwiseMulMultilinear(self.subspace)


class ComponentwiseMulMultilinear(Node):
    """
    :no-index:

    Internal class for implementing `ComponentwiseMul`, see also `MultilinearNode`
    """

    subspace: Subspace

    def __init__(self, subspace: Subspace):
        super().__init__(subspace.dimension**2, subspace.dimension)
        self.subspace = subspace

    def _subspace_in(self) -> Subspace:
        return Subspace(self.subspace.registers * 2)

    def _subspace_out(self) -> Subspace:
        return Subspace(registers=self.subspace.registers, zero_qubits=self.subspace.total_qubits)

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        shape = list(input.shape[:-1])
        dim = self.subspace.dimension
        input_reshaped = input.reshape(shape + [dim, dim])
        return np.diagonal(input_reshaped, axis1=-2, axis2=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 1:
            result = np.diag(input)
        else:
            result = np.zeros(list(input.shape[:-1]) + [input.shape[-1]] * 2, dtype=np.complex128)
            indices = np.arange(input.shape[-1])
            result[:, indices, indices] = input[:, indices]
        return result.reshape(list(input.shape[:-1]) + [input.shape[-1] ** 2])

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        circuit = Circuit()

        for i in range(self.subspace.total_qubits):
            circuit += tq.gates.CNOT(target[i], target[i + self.subspace.total_qubits])

        return circuit

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 0
