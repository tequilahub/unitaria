import numpy as np

from bequem.circuit import Circuit
from bequem.circuits.state_prep import prepare_state
from bequem.nodes.node import Node
from bequem.qubit_map import QubitMap


class ConstantVector(Node):
    def __init__(self, vec: np.ndarray):
        self.n_qubits = round(np.log2(vec.shape[0]))
        assert 2 ** self.n_qubits == vec.shape[0]
        self.vec = vec

    def qubits_in(self) -> QubitMap:
        return QubitMap(0, self.n_qubits)

    def qubits_out(self) -> QubitMap:
        return QubitMap(self.n_qubits)

    def normalization(self) -> float:
        return np.linalg.norm(self.vec)

    def compute(self, input: np.ndarray | None = None) -> np.ndarray:
        return self.vec

    def compute_adjoint(self, input: np.ndarray | None = None) -> np.ndarray:
        return self.vec.T @ input

    def circuit(self) -> Circuit:
        normalized = self.vec / self.normalization()
        # reversed because prepare_state expects MSB ordering
        tq_circuit = prepare_state(normalized, list(reversed(range(self.n_qubits))))
        return Circuit(tq_circuit)
