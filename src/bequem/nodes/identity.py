import numpy as np

from bequem.circuit import Circuit
from bequem.qubit_map import QubitMap
from bequem.nodes.node import Node


class Identity(Node):
    """
    Node representing the identity matrix on a given vectorspace

    :ivar qubits:
        The domain of the identity matrix
    """
    qubits: QubitMap

    def __init__(self, qubits: QubitMap):
        """
        :param qubits:
            The domain of the identity matrix
        """
        self.qubits = qubits

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return { "qubits": self.qubits }

    def qubits_in(self) -> QubitMap:
        return self.qubits

    def qubits_out(self) -> QubitMap:
        return self.qubits

    def normalization(self) -> float:
        return 1

    def phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def circuit(self) -> Circuit:
        circuit = Circuit()
        # TODO: Hacky because tequila does not really support circuits without qubits
        if self.qubits.total_qubits > 0:
            circuit.tq_circuit.n_qubits = self.qubits.total_qubits
        return circuit
