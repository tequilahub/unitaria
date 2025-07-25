import numpy as np

from unitaria.circuit import Circuit
from unitaria.circuits.state_prep import prepare_state
from unitaria.nodes.node import Node
from unitaria.subspace import Subspace


class ConstantVector(Node):
    """
    Node representing the given vector

    :param vec:
        The vector represented by this node
    """

    vec: np.ndarray

    def __init__(self, vec: np.ndarray):
        super().__init__(1, vec.shape[0])
        self.n_qubits = round(np.log2(vec.shape[0]))
        assert 2**self.n_qubits == vec.shape[0]
        self.vec = vec

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"vec": self.vec}

    def _subspace_in(self) -> Subspace:
        return Subspace(0, self.n_qubits)

    def _subspace_out(self) -> Subspace:
        return Subspace(self.n_qubits)

    def _normalization(self) -> float:
        return np.linalg.norm(self.vec)

    def compute(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 1:
            return self.vec * input[0]
        else:
            return (np.array([self.vec]).T @ input.T).T

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return (np.array([np.conj(self.vec)]) @ input.T).T

    def _circuit(self) -> Circuit:
        normalized = self.vec / self.normalization
        # reversed because prepare_state expects MSB ordering
        tq_circuit = prepare_state(normalized, list(reversed(range(self.n_qubits))))
        return Circuit(tq_circuit)
