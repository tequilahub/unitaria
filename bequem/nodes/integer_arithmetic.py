from node import Node
from qubit_map import QubitMap, IdBit
from circuit import Circuit

class ConstantIntegerAddition(Node):

    def __init__(self, bits: int, c: int):
        self.bits = bits
        self.c = c

    def qubits_in(self) -> QubitMap:
        return [IdBit for _ in range(bits)]

    def qubits_out(self) -> QubitMap:
        return [IdBit for _ in range(bits)]

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.array) -> np.array:
        output = np.zeros_like(input)
        output[c:] = input[:-c]
        output[:c] = input[-c:]
        return output

    def circuit(self) -> Circuit:
        raise NotImplementedError
