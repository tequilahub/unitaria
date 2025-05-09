import numpy as np

from bequem.circuit import Circuit
from bequem.qubit_map import Subspace
from bequem.nodes.node import Node


class Identity(Node):
    """
    Node representing the identity matrix on a given vectorspace

    :ivar qubits:
        The domain of the identity matrix
    """
    subspace: Subspace

    def __init__(self, subspace: Subspace, project_to: Subspace | None = None):
        """
        :param qubits:
            The domain of the identity matrix
        """
        self.subspace = subspace
        self.project_to = project_to

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        params = {}
        params["qubits"] = self.subspace
        if self.project_to is not None:
            params["project_to"] = self.project_to
        return params

    def _subspace_in(self) -> Subspace:
        return self.subspace

    def _subspace_out(self) -> Subspace:
        return self.project_to or self.subspace

    def _normalization(self) -> float:
        return 1

    def _phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        if self.project_to is None:
            return input
        else:
            expanded = np.zeros(2 ** self.subspace.total_qubits)
            expanded[self.subspace.enumerate_basis()] = input
            return expanded[self.project_to.enumerate_basis()]

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        if self.project_to is None:
            return input
        else:
            expanded = np.zeros(2 ** self.subspace.total_qubits)
            expanded[self.project_to.enumerate_basis()] = input
            return expanded[self.subspace.enumerate_basis()]

    def _circuit(self) -> Circuit:
        circuit = Circuit()
        # TODO: Hacky because tequila does not really support circuits without qubits
        if self.subspace.total_qubits > 0:
            circuit.tq_circuit.n_qubits = self.subspace.total_qubits
        return circuit
