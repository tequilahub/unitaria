import numpy as np

from bequem.circuit import Circuit
from bequem.circuits.state_prep import prepare_state
from bequem.nodes.node import Node
from bequem.subspace import Subspace


class ConstantVector(Node):
    """
    Node representing the given vector

    :ivar vec:
        The vector represented by this node
    """
    vec: np.ndarray

    def __init__(self, vec: np.ndarray):
        """
        :param vec:
            The vector represented by this node
        """
        self.n_qubits = round(np.log2(vec.shape[0]))
        assert 2 ** self.n_qubits == vec.shape[0]
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

    def _phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray | None = None) -> np.ndarray:
        if input is None:
            return self.vec
        elif input.ndim == 1:
            return self.vec * input[0]
        else:
            return (np.array([self.vec]).T @ input.T).T

    def compute_adjoint(self, input: np.ndarray | None = None) -> np.ndarray:
        return np.array([self.vec]) @ input

    def _circuit(self) -> Circuit:
        normalized = self.vec / self.normalization
        # reversed because prepare_state expects MSB ordering
        tq_circuit = prepare_state(normalized, list(reversed(range(self.n_qubits))))
        return Circuit(tq_circuit)


class ConstantUnitary(Node):
    """
    Node representing the given unitary
    """

    unitary: np.ndarray

    def __init__(self, unitary: np.ndarray):
        assert unitary.ndim == 2
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

    def parameters(self) -> dict:
        return {"unitary": self.unitary}

    def _subspace_in(self) -> Subspace:
        return Subspace.from_dim(self.unitary.shape[1], self.bits)

    def _subspace_out(self) -> Subspace:
        return Subspace.from_dim(self.unitary.shape[0], self.bits)

    def _normalization(self) -> Subspace:
        return 1

    def _phase(self) -> Subspace:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        return (self.unitary @ input.T).T

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return (np.conj(self.unitary.T) @ input.T).T

    def _circuit(self) -> Circuit:
        from qiskit.circuit.library import UnitaryGate

        qiskit_circuit = UnitaryGate(self.extended_unitary).definition

        return Circuit.from_qiskit(qiskit_circuit)

def _extend_basis_by_one(U: np.array, n: int):
    candidates = np.eye(U.shape[0]) - U[:, :n] @ np.conj(U.T)[:n, :]
    norms = np.linalg.norm(candidates, ord=2, axis=0)
    best = np.argmax(norms)
    U[:, n] = candidates[:, best] / norms[best]
