import numpy as np

from bequem.circuit import Circuit
from bequem.qubit_map import QubitMap, Qubit
from bequem.nodes.node import Node


class Identity(Node):
    """
    Node representing the identity matrix on a given vectorspace

    :ivar qubits:
        The domain of the identity matrix
    """
    qubits: QubitMap

    def __init__(self, qubits: QubitMap, project_to: QubitMap | None = None):
        """
        :param qubits:
            The domain of the identity matrix
        """
        self.qubits = qubits
        self.project_to = project_to

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        params = {}
        params["qubits"] = self.qubits
        if self.project_to is not None:
            params["project_to"] = self.project_to
        return params

    def qubits_in(self) -> QubitMap:
        return self.qubits

    def qubits_out(self) -> QubitMap:
        return self.project_to or self.qubits

    def normalization(self) -> float:
        return 1

    def phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        if self.project_to is None:
            return input
        else:
            expanded = np.zeros(2 ** self.qubits.total_qubits)
            expanded[self.qubits.enumerate_basis()] = input
            return expanded[self.project_to.enumerate_basis()]

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        if self.project_to is None:
            return input
        else:
            expanded = np.zeros(2 ** self.qubits.total_qubits)
            expanded[self.project_to.enumerate_basis()] = input
            return expanded[self.qubits.enumerate_basis()]

    def circuit(self) -> Circuit:
        circuit = Circuit()
        # TODO: Hacky because tequila does not really support circuits without qubits
        if self.qubits.total_qubits > 0:
            circuit.tq_circuit.n_qubits = self.qubits.total_qubits
        return circuit

    def controlled(self) -> Node:
        if self.project_to is None:
            return Identity(QubitMap([Qubit(QubitMap(self.qubits.total_qubits), self.qubits)]))
        else:
            return Identity(
                QubitMap([Qubit(QubitMap(self.qubits.total_qubits), self.qubits)]),
                QubitMap([Qubit(QubitMap(self.qubits.total_qubits), self.project_to)])
            )
