import numpy as np
import tequila as tq

from bequem.nodes import Node
from bequem.subspace.subspace import Subspace
from bequem.circuit import Circuit


class ComponentwiseMul(Node):
    """
    Node implementing the (bilinear) componentwise multiplication operator

    More specifically this implements the bilinear map ``(x, y) -> [x_1 * y_1,
    ..., x_n * y_n]``. Usually you will want the elementwise product of two
    vectors, in which case the correct result will be obtained by building the
    tensor product of the vectors and then multiplying it with this operation,
    i.e. ``Mul(Tensor(a, b), ComponentwiseMul(a.subspace_out()))``

    :ivar qubits:
        The vector space in which to perform the element-wise operation
    """

    subspace: Subspace

    def __init__(self, subspace: Subspace):
        self.subspace = subspace

    def _subspace_in(self) -> Subspace:
        return Subspace(self.subspace.registers * 2)

    def _subspace_out(self) -> Subspace:
        return Subspace(self.subspace.registers, self.subspace.total_qubits)

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

    def _circuit(self) -> Circuit:
        circuit = Circuit()

        for i in range(self.subspace.total_qubits):
            circuit.tq_circuit += tq.gates.CNOT(i, i + self.subspace.total_qubits)

        return circuit
