import numpy as np
import tequila as tq
from .node import Node
from bequem.qubit_map import QubitMap, ID
from bequem.circuit import Circuit


class Increment(Node):
    def __init__(self, bits: int):
        if bits < 1:
            raise ValueError()
        self.bits = bits

    def qubits_in(self) -> QubitMap:
        return QubitMap([ID for _ in range(self.bits)])

    def qubits_out(self) -> QubitMap:
        return QubitMap([ID for _ in range(self.bits)])

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return np.roll(input, 1)

    def circuit(self) -> Circuit:
        circuit = tq.QCircuit()
        for i in reversed(range(self.bits)):
            circuit += tq.gates.X(target=i, control=list(range(i)))
        return Circuit(circuit)
