import numpy as np

from bequem.circuit import Circuit
from bequem.circuits.state_prep import prepare_state
from bequem.nodes.node import Node
from bequem.qubit_map import Subspace


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
        self._normalization = np.linalg.norm(vec)
        self._phase = np.mean(np.angle(vec))

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"vec": self.vec}

    def qubits_in(self) -> Subspace:
        return Subspace(0, self.n_qubits)

    def qubits_out(self) -> Subspace:
        return Subspace(self.n_qubits)

    def normalization(self) -> float:
        return self._normalization

    def phase(self) -> float:
        return self._phase

    def compute(self, input: np.ndarray | None = None) -> np.ndarray:
        return self.vec

    def compute_adjoint(self, input: np.ndarray | None = None) -> np.ndarray:
        return self.vec.T @ input

    def circuit(self) -> Circuit:
        normalized = self.vec / self.normalization()
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
        assert unitary.shape[0] == unitary.shape[1]
        self.bits = int(np.ceil(np.log2(unitary.shape[0])))
        assert 2**self.bits == unitary.shape[0]
        self.unitary = unitary

    def parameters(self) -> dict:
        return {"unitary": self.unitary}

    def qubits_in(self) -> Subspace:
        return Subspace(self.bits)

    def qubits_out(self) -> Subspace:
        return Subspace(self.bits)

    def normalization(self) -> Subspace:
        return 1

    def phase(self) -> Subspace:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        if input is None:
            input = np.array([1])
        return self.unitary @ input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return np.conj(self.unitary.T) @ input

    def circuit(self) -> Circuit:
        from qiskit.circuit.library import UnitaryGate

        qiskit_circuit = UnitaryGate(self.unitary).definition

        return Circuit.from_qiskit(qiskit_circuit)
