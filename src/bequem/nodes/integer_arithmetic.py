import numpy as np
import tequila as tq
from .node import Node
from bequem.qubit_map import QubitMap, IdBit
from bequem.circuit import Circuit


class Increment(Node):

    def __init__(self, bits: int):
        self.bits = bits

    def qubits_in(self) -> QubitMap:
        return QubitMap([IdBit for _ in range(self.bits)])

    def qubits_out(self) -> QubitMap:
        return QubitMap([IdBit for _ in range(self.bits)])

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return np.roll(input, 1)

    def circuit(self) -> Circuit:
        circuit = tq.QCircuit()
        for i in range(self.bits - 1):
            circuit += tq.gates.CNOT(target=self.bits - i - 1,
                               control=tuple(range(0, self.bits - i - 1)))
        circuit += tq.gates.X(target=0)
        return Circuit(circuit)
