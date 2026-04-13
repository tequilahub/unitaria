from typing import Sequence

import numpy as np

from unitaria.circuit import Circuit
from unitaria.circuits.generic_unitary import generic_unitary
from unitaria.nodes.node import Node
from unitaria.subspace import Subspace
from unitaria.util import is_unitary


class ConstantUnitary(Node):
    """
    Node representing the given unitary

    :param unitary:
        The unitary which should be implemented.
    """

    unitary: np.ndarray

    def __init__(self, unitary: np.ndarray):
        """
        Initialize a ConstantUnitary node.

        Args:
            unitary (np.ndarray): The unitary matrix to be applied. Can be rectangular; will be extended to square if needed.
        Raises:
            ValueError: If the provided matrix is square but not unitary.
        """
        super().__init__(unitary.shape[1], unitary.shape[0])
        assert unitary.ndim == 2
        self.unitary = unitary
        n, m = unitary.shape
        extended_unitary = np.zeros((max(n, m), max(n, m)), dtype=complex)
        extended_unitary[:n, :m] = unitary
        if n != m:
            swap = n < m
            if swap:
                n, m = m, n
                extended_unitary = extended_unitary.T

            for i in range(m, n):
                _extend_basis_by_one(extended_unitary, i)

            if swap:
                extended_unitary = extended_unitary.T
        self.extended_unitary = extended_unitary
        if not is_unitary(self.extended_unitary):
            raise ValueError("The provided matrix is not unitary.")
        self.bits = int(np.ceil(np.log2(extended_unitary.shape[0])))
        assert 2**self.bits == extended_unitary.shape[0]

    @staticmethod
    def _is_unitary(U: np.ndarray, tol: float = 1e-8) -> bool:
        """
        Check if a matrix is unitary within a given tolerance.

        Args:
            U (np.ndarray): Matrix to check.
            tol (float): Tolerance for unitarity check.

        Returns:
            bool: True if U is unitary, False otherwise.
        """
        identity = np.eye(U.shape[0])
        return np.allclose(U @ np.conj(U.T), identity, atol=tol) and np.allclose(np.conj(U.T) @ U, identity, atol=tol)

    def parameters(self) -> dict:
        """
        Returns the parameters of the node.

        Returns:
            dict: Dictionary containing the unitary matrix.
        """
        return {"unitary": self.unitary}

    def _subspace_in(self) -> Subspace:
        """
        Returns the input subspace for this node.

        Returns:
            Subspace: The input subspace.
        """
        return Subspace(dim=self.unitary.shape[1], bits=self.bits)

    def _subspace_out(self) -> Subspace:
        """
        Returns the output subspace for this node.

        Returns:
            Subspace: The output subspace.
        """
        return Subspace(dim=self.unitary.shape[0], bits=self.bits)

    def _normalization(self) -> float:
        """
        Returns the normalization factor for this node (always 1).

        Returns:
            float: The normalization factor (1.0).
        """
        return 1

    def is_guaranteed_unitary(self) -> bool:
        n, m = self.unitary.shape
        return n == m

    def compute(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the unitary matrix to the input state vector.

        Args:
            input (np.ndarray): Input state vector.

        Returns:
            np.ndarray: Output state vector after applying the unitary.
        """
        return (self.unitary @ input.T).T

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the adjoint of the unitary to the input state vector.

        Args:
            input (np.ndarray): Input state vector.

        Returns:
            np.ndarray: Output state vector after applying the adjoint unitary.
        """
        return (np.conj(self.unitary.T) @ input.T).T

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        """
        Returns the quantum circuit implementing this constant unitary operation.

        Args:
            target (Sequence[int]): Target qubits.
            clean_ancillae (Sequence[int]): Clean ancilla qubits (unused).
            borrowed_ancillae (Sequence[int]): Borrowed ancilla qubits (unused).

        Returns:
            Circuit: The constructed quantum circuit.
        """
        return Circuit(generic_unitary(U=self.extended_unitary, target=target))

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


def _extend_basis_by_one(U: np.array, n: int):
    """
    Extends the basis of a (possibly rectangular) unitary matrix by one column/row.

    Args:
        U (np.ndarray): The matrix to extend (in-place).
        n (int): The index at which to extend the basis.
    """
    candidates = np.eye(U.shape[0]) - U[:, :n] @ np.conj(U.T)[:n, :]
    norms = np.linalg.norm(candidates, ord=2, axis=0)
    best = np.argmax(norms)
    U[:, n] = candidates[:, best] / norms[best]
