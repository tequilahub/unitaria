from typing import Sequence

import numpy as np

from unitaria.circuit import Circuit
from unitaria.circuits.state_prep import prepare_state
from unitaria.nodes.node import Node
from unitaria.subspace import Subspace


class ConstantVector(Node):
    """
    Node representing the given vector

    :param vec: The vector represented by this node
    """

    vec: np.ndarray

    def __init__(self, vec: np.ndarray):
        """
        Initialize a ConstantVector node.

        :param vec: The vector to be represented/prepared.
        :raises AssertionError: If the vector length is not a power of 2.
        """
        super().__init__(1, vec.shape[0])
        self.n_qubits = int(np.ceil(np.log2(vec.shape[0])))
        self.vec = vec

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"vec": self.vec}

    def _subspace_in(self) -> Subspace:
        return Subspace("0" * self.n_qubits)

    def _subspace_out(self) -> Subspace:
        return Subspace.from_dim(self.vec.shape[0], bits=self.n_qubits)

    def _normalization(self) -> float:
        return np.linalg.norm(self.vec)

    def compute(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 1:
            return self.vec * input[0]
        else:
            return (np.array([self.vec]).T @ input.T).T

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return (np.array([np.conj(self.vec)]) @ input.T).T

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        if self.vec.shape[0] == 1:
            return Circuit()
        if self.normalization < 1e-8:
            circuit = Circuit()
            circuit.n_qubits = self.n_qubits
            return circuit
        normalized = self.vec / self.normalization
        normalized = np.concatenate((normalized, np.zeros(2**self.n_qubits - self.vec.shape[0])))
        tq_circuit = prepare_state(normalized, target)
        return Circuit(tq_circuit)

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 0
