import numpy as np

from bequem.circuit import Circuit
from bequem.subspace.subspace import Subspace
from bequem.nodes.node import Node


class Identity(Node):
    """
    Node representing the identity matrix on a given vectorspace

    :ivar qubits:
        The domain of the identity matrix
    """

    subspace: Subspace

    def __init__(self, subspace: Subspace | int, project_to: Subspace | None = None):
        """
        :param qubits:
            The domain of the identity matrix
        """
        if not isinstance(subspace, Subspace):
            subspace = Subspace(subspace)
        self.subspace = subspace
        self.project_to = project_to
        if project_to is not None and subspace.total_qubits != project_to.total_qubits:
            raise ValueError("Subspaces must have same number of qubits")

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

    def compute(self, input: np.ndarray) -> np.ndarray:
        if self.project_to is None:
            return input
        else:
            outer_shape = list(input.shape[:-1])
            input = input.reshape([-1, self.subspace_in.dimension])
            expanded = np.zeros((input.shape[0], 2**self.subspace_in.total_qubits), dtype=np.complex128)
            expanded[:, self.subspace_in.enumerate_basis()] = input
            return expanded[:, self.subspace_out.enumerate_basis()].reshape(outer_shape + [-1])

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        if self.project_to is None:
            return input
        else:
            outer_shape = list(input.shape[:-1])
            input = input.reshape([-1, self.subspace_out.dimension])
            expanded = np.zeros((input.shape[0], 2**self.subspace_out.total_qubits), dtype=np.complex128)
            expanded[:, self.subspace_out.enumerate_basis()] = input
            return expanded[:, self.subspace_in.enumerate_basis()].reshape(outer_shape + [-1])

    def _circuit(self) -> Circuit:
        circuit = Circuit()
        # TODO: Hacky because tequila does not really support circuits without qubits
        if self.subspace.total_qubits > 0:
            circuit.tq_circuit.n_qubits = self.subspace.total_qubits
        return circuit
