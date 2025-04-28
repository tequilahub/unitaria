import numpy as np
import tequila as tq

from bequem.nodes import Node
from bequem.qubit_map import QubitMap
from bequem.circuit import Circuit


class ComponentwiseMul(Node):
    """
    Node implementing the (bilinear) componentwise multiplication operator

    More specifically this implements the bilinear map ´´(x, y) -> [x_1
    y_1, ..., x_n, y_n]´´. Usually you will want the elementwise product of two
    vectors, in which case the correct result will be obtained by building the
    tensor product of the vectors and then multiplying it with this operation,
    i.e. ``Mul(Tensor(a, b), CompnentwiseMul(a.qubits_out()))``

    :ivar qubits:
        The vector space in which to perform the element-wise operation
    """
    qubits: QubitMap

    def __init__(self, qubits: QubitMap):
        self.qubits = qubits

    def qubits_in(self) -> QubitMap:
        return QubitMap(self.qubits.registers * 2, self.qubits.zero_qubits * 2)

    def qubits_out(self) -> QubitMap:
        return QubitMap(
            self.qubits.registers, self.qubits.zero_qubits + self.qubits.total_qubits
        )

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        shape = list(input.shape[:-1])
        dim = self.qubits.dimension
        input_reshaped = input.reshape(shape + [dim, dim])
        return np.diagonal(input_reshaped, axis1=-2, axis2=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 1:
            result = np.diag(input)
        else:
            result = np.zeros(list(input.shape[:-1]) + [input.shape[-1]] * 2)
            indices = np.arange(input.shape[-1])
            result[:, indices, indices] = input[:, indices]
        return result.reshape(list(input.shape[:-1]) + [input.shape[-1] ** 2])

    def circuit(self) -> Circuit:
        circuit = Circuit()
        for i in reversed(
                range(self.qubits.total_qubits - self.qubits.zero_qubits)):
            qubit1 = self.qubits.total_qubits + i
            qubit2 = self.qubits.total_qubits - self.qubits.zero_qubits + i
            if qubit1 != qubit2:
                circuit.tq_circuit += tq.gates.SWAP(qubit1, qubit2)

        for i in range(self.qubits.total_qubits):
            circuit.tq_circuit += tq.gates.CNOT(i, i + self.qubits.total_qubits)

        return circuit
