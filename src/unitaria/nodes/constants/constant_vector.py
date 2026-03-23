from typing import Sequence

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
        """
        Initialize a ConstantVector node.

        Args:
            vec (np.ndarray): The vector to be represented/prepared.
        Raises:
            AssertionError: If the vector length is not a power of 2.
        """
        super().__init__(1, vec.shape[0])
        self.n_qubits = round(np.log2(vec.shape[0]))
        assert 2**self.n_qubits == vec.shape[0]
        self.vec = vec

    def children(self) -> list[Node]:
        """
        Returns the child nodes (none for ConstantVector).

        Returns:
            list[Node]: An empty list.
        """
        return []

    def parameters(self) -> dict:
        """
        Returns the parameters of the node.

        Returns:
            dict: Dictionary containing the vector.
        """
        return {"vec": self.vec}

    def _subspace_in(self) -> Subspace:
        """
        Returns the input subspace for this node.

        Returns:
            Subspace: The input subspace (all zero qubits).
        """
        return Subspace(bits=0, zero_qubits=self.n_qubits)

    def _subspace_out(self) -> Subspace:
        """
        Returns the output subspace for this node.

        Returns:
            Subspace: The output subspace (n_qubits bits).
        """
        return Subspace(bits=self.n_qubits)

    def _normalization(self) -> float:
        """
        Returns the normalization factor for this node (norm of the vector).

        Returns:
            float: The norm of the vector.
        """
        return np.linalg.norm(self.vec)

    def compute(self, input: np.ndarray) -> np.ndarray:
        """
        Computes the output by multiplying the input with the constant vector.

        Args:
            input (np.ndarray): Input state vector.

        Returns:
            np.ndarray: Output state vector after applying the constant vector.
        """
        if input.ndim == 1:
            return self.vec * input[0]
        else:
            return (np.array([self.vec]).T @ input.T).T

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        """
        Computes the adjoint operation for the constant vector.

        Args:
            input (np.ndarray): Input state vector.

        Returns:
            np.ndarray: Output state vector after applying the adjoint operation.
        """
        return (np.array([np.conj(self.vec)]) @ input.T).T

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        """
        Returns the quantum circuit preparing the constant vector state.

        Args:
            target (Sequence[int]): Target qubits.
            clean_ancillae (Sequence[int]): Clean ancilla qubits (currently unused).
            borrowed_ancillae (Sequence[int]): Borrowed ancilla qubits (currently unused).

        Returns:
            Circuit: The constructed quantum circuit.
        """
        normalized = self.vec / self.normalization
        tq_circuit = prepare_state(normalized, target)
        return Circuit(tq_circuit)

    def clean_ancilla_count(self) -> int:
        """
        Returns the number of clean ancilla qubits required (always 0).

        Returns:
            int: Number of clean ancilla qubits (0).
        """
        return 0

    def borrowed_ancilla_count(self) -> int:
        """
        Returns the number of borrowed ancilla qubits required (always 0).

        Returns:
            int: Number of borrowed ancilla qubits (0).
        """
        return 0
