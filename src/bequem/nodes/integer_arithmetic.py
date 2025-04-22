import numpy as np

from .node import Node
from bequem.qubit_map import QubitMap, ID
from bequem.circuit import Circuit
from bequem.circuits.arithmetic import increment_circuit_single_ancilla, addition_circuit


class Increment(Node):
    def __init__(self, bits: int):
        if bits < 1:
            raise ValueError()
        self.bits = bits

    def qubits_in(self) -> QubitMap:
        return QubitMap(self.bits, 1)

    def qubits_out(self) -> QubitMap:
        return QubitMap(self.bits, 1)

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return np.roll(input, 1, axis=-1)

    def circuit(self) -> Circuit:
        circuit = increment_circuit_single_ancilla(target=list(reversed(range(self.bits))), ancilla=self.bits)
        return Circuit(circuit)


class IntegerAddition(Node):
    def __init__(self, source_bits: int, target_bits: int):
        # TODO: Restriction is because the ancilla free construction needs two source bits.
        #  I know how to fix this but haven't implemented it yet.
        if source_bits < 2 or target_bits < source_bits:
            raise ValueError()
        self.source_bits = source_bits
        self.target_bits = target_bits

    def qubits_in(self) -> QubitMap:
        return QubitMap(self.source_bits + self.target_bits, 1)

    def qubits_out(self) -> QubitMap:
        return QubitMap(self.source_bits + self.target_bits, 1)

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        old_shape = input.shape
        N = 2 ** self.source_bits
        M = 2 ** self.target_bits
        input = input.reshape((-1, M, N))
        result = np.zeros_like(input)
        for val in range(N):
            result[:, :, val] += np.roll(input[:, :, val], val, axis=-1)
        return result.reshape(old_shape)

    def circuit(self) -> Circuit:
        source = list(reversed(range(self.source_bits)))
        target = list(reversed(range(self.source_bits, self.source_bits + self.target_bits)))
        circuit = addition_circuit(source, target)
        return Circuit(circuit)
