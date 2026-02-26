from typing import Sequence

import numpy as np

from unitaria.circuit import Circuit
from unitaria.circuits.generic_unitary import generic_unitary
from unitaria.nodes.node import Node
from unitaria.subspace import Subspace


class ConstantUnitary(Node):
    """
    Node representing the given unitary

    :param unitary:
        The unitary which should be implemented.
    """

    unitary: np.ndarray

    def __init__(self, unitary: np.ndarray):
        super().__init__(unitary.shape[1], unitary.shape[0])
        assert unitary.ndim == 2
        if unitary.shape[0] == unitary.shape[1]:
            if not self._is_unitary(unitary):
                raise ValueError("The provided matrix is not unitary.")
        self.unitary = unitary
        n, m = unitary.shape
        extended_unitary = np.zeros((max(n, m), max(n, m)))
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
        self.bits = int(np.ceil(np.log2(extended_unitary.shape[0])))
        assert 2**self.bits == extended_unitary.shape[0]

    @staticmethod
    def _is_unitary(U: np.ndarray, tol: float = 1e-8) -> bool:
        identity = np.eye(U.shape[0])
        return np.allclose(U @ np.conj(U.T), identity, atol=tol) and np.allclose(np.conj(U.T) @ U, identity, atol=tol)

    def parameters(self) -> dict:
        return {"unitary": self.unitary}

    def _subspace_in(self) -> Subspace:
        return Subspace(dim=self.unitary.shape[1], bits=self.bits)

    def _subspace_out(self) -> Subspace:
        return Subspace(dim=self.unitary.shape[0], bits=self.bits)

    def _normalization(self) -> Subspace:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return (self.unitary @ input.T).T

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return (np.conj(self.unitary.T) @ input.T).T

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        return Circuit(generic_unitary(U=self.extended_unitary, target=target))

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 0


def _extend_basis_by_one(U: np.array, n: int):
    candidates = np.eye(U.shape[0]) - U[:, :n] @ np.conj(U.T)[:n, :]
    norms = np.linalg.norm(candidates, ord=2, axis=0)
    best = np.argmax(norms)
    U[:, n] = candidates[:, best] / norms[best]
