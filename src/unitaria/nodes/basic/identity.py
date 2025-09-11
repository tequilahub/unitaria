import numpy as np

from unitaria.circuit import Circuit
from unitaria.subspace import Subspace
from unitaria.nodes.node import Node


class Identity(Node):
    """
    Node representing the identity matrix on a given vectorspace

    :param subspace:
        The domain of the identity matrix
    """

    subspace: Subspace

    def __init__(self, subspace: Subspace | int):
        if not isinstance(subspace, Subspace):
            subspace = Subspace(subspace)
        super().__init__(subspace.dimension, subspace.dimension)
        self.subspace = subspace

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        params = {}
        params["subspace"] = self.subspace
        return params

    def _subspace_in(self) -> Subspace:
        return self.subspace

    def _subspace_out(self) -> Subspace:
        return self.subspace

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def _circuit(self) -> Circuit:
        circuit = Circuit()
        # TODO: Hacky because tequila does not really support circuits without qubits
        if self.subspace.total_qubits > 0:
            circuit.tq_circuit.n_qubits = self.subspace.total_qubits
        return circuit
