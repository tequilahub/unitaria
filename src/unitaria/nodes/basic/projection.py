import numpy as np

from unitaria.circuit import Circuit
from unitaria.subspace import Subspace
from unitaria.nodes.node import Node


class Projection(Node):
    """
    Node representing a projection matrix

    Concretely, this is the identity matrix on the full state restricted to this
    subspaces ``subspace_from`` and ``subspace_to``.

    :param from:
        The domain of the identity matrix
    """

    subspace_from: Subspace
    subspace_to: Subspace

    def __init__(self, subspace_from: Subspace | int, subspace_to: Subspace | int):
        if not isinstance(subspace_from, Subspace):
            subspace_from = Subspace(subspace_from)
        if not isinstance(subspace_to, Subspace):
            subspace_to = Subspace(subspace_to)
        super().__init__(subspace_from.dimension, subspace_to.dimension)
        self.subspace_from = subspace_from
        self.subspace_to = subspace_to
        if subspace_from.total_qubits != subspace_to.total_qubits:
            raise ValueError("Subspaces must have same number of qubits")

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        params = {}
        params["subspace_from"] = self.subspace_from
        params["subspace_to"] = self.subspace_to
        return params

    def _subspace_in(self) -> Subspace:
        return self.subspace_from

    def _subspace_out(self) -> Subspace:
        return self.subspace_to

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        input = input.reshape([-1, self.dimension_in])
        expanded = np.zeros((input.shape[0], 2**self.subspace_in.total_qubits), dtype=np.complex128)
        expanded[:, self.subspace_in.enumerate_basis()] = input
        return expanded[:, self.subspace_out.enumerate_basis()].reshape(outer_shape + [-1])

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        input = input.reshape([-1, self.dimension_out])
        expanded = np.zeros((input.shape[0], 2**self.subspace_out.total_qubits), dtype=np.complex128)
        expanded[:, self.subspace_out.enumerate_basis()] = input
        return expanded[:, self.subspace_in.enumerate_basis()].reshape(outer_shape + [-1])

    def _circuit(self) -> Circuit:
        circuit = Circuit()
        # TODO: Hacky because tequila does not really support circuits without qubits
        if self.subspace_from.total_qubits > 0:
            circuit.tq_circuit.n_qubits = self.subspace_from.total_qubits
        return circuit
