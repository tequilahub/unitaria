import numpy as np
from tequila_building_blocks.state_prep import prepare_state

from bequem.circuit import Circuit
from bequem.node import Node
from bequem.qubit_map import QubitMap, ZeroBit, IdBit


class ConstantMatrix(Node):

    def __init__(self, vec: np.array):
        self.n_qubits = round(np.log2(vec.shape[0]))
        assert 2**self.n_qubits == vec.shape[0]
        self.vec = vec

    def qubits_in(self) -> QubitMap:
        return QubitMap([ZeroBit() for _ in range(self.n_qubits)])

    def qubits_out(self) -> QubitMap:
        return QubitMap([IdBit() for _ in range(self.n_qubits)])

    def normalization(self) -> float:
        return np.linalg.norm(self.vec)

    def compute(self, input: np.array | None = None) -> np.array:
        assert input is None
        return self.vec / self.normalization()

    def circuit(self) -> Circuit:
        tq_circuit = prepare_state(self.vec, range(self.n_qubits))
        return Circuit(tq_circuit)
