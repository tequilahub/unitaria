import numpy as np
import tequila as tq
from node import Node
from qubit_map import QubitMap, IdBit
from circuit import Circuit


class ConstantIntegerAddition(Node):

    def __init__(self, bits: int, c: int):
        self.bits = bits
        self.c = c

    def qubits_in(self) -> QubitMap:
        return [IdBit for _ in range(self.bits)]

    def qubits_out(self) -> QubitMap:
        return [IdBit for _ in range(self.bits)]

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.array) -> np.array:
        return np.roll(input, self.c)

    def circuit(self) -> Circuit:
        if self.c == 1:
            circuit = tq.QCircuit()
            for i in range(self.bits - 1):
                circuit += tq.CNOT(target=self.bits-i, control=tuple(range(0, self.bits-i)))
            circuit += tq.X(target=0)
            return Circuit(circuit)
        else:
            raise NotImplementedError
