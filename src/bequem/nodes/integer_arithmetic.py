import numpy as np
import tequila as tq

from .node import Node
from bequem.qubit_map import QubitMap
from bequem.circuit import Circuit
from bequem.circuits.arithmetic import increment_circuit_single_ancilla, addition_circuit


class Increment(Node):
    def __init__(self, bits: int, control_bits: int=0):
        if bits < 1:
            raise ValueError()
        self.bits = bits
        self.control_bits = control_bits

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return { "bits": self.bits }

    def qubits_in(self) -> QubitMap:
        total_bits = self.bits + self.control_bits
        if total_bits <= 3:
            return QubitMap(total_bits)
        else:
            return QubitMap(total_bits, 1)

    def qubits_out(self) -> QubitMap:
        total_bits = self.bits + self.control_bits
        if total_bits <= 3:
            return QubitMap(total_bits)
        else:
            return QubitMap(total_bits, 1)

    def normalization(self) -> float:
        return 1

    def phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        remainder, controlled = np.split(input, [-2**self.bits], axis=-1)
        controlled = np.roll(controlled, 1, axis=-1)
        return np.concatenate((remainder, controlled), axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        remainder, controlled = np.split(input, [-2**self.bits], axis=-1)
        controlled = np.roll(controlled, -1, axis=-1)
        return np.concatenate((remainder, controlled), axis=-1)

    def circuit(self) -> Circuit:
        total_bits = self.bits + self.control_bits
        if total_bits <= 3:
            circuit = tq.QCircuit()
            for i in reversed(range(total_bits)):
                circuit += tq.gates.X(target=i, control=list(range(i)))
        else:
            circuit = increment_circuit_single_ancilla(target=list(reversed(range(total_bits))), ancilla=total_bits)

        if self.control_bits == 0:
            return Circuit(circuit)
        else:
            controlled_circuit = Circuit()
            qubit_map = dict(
                [(i, i + self.bits) for i in range(self.control_bits)] +
                [(i + self.control_bits, i) for i in range(self.bits)] + 
                [(total_bits, total_bits)]
            )
            controlled_circuit.tq_circuit += circuit.map_qubits(qubit_map)
            controlled_circuit.tq_circuit += tq.gates.X(target=list(range(self.bits, total_bits)))
            return controlled_circuit

    def controlled(self) -> Node:
        return Increment(self.bits, self.control_bits + 1)


class IntegerAddition(Node):
    def __init__(self, source_bits: int, target_bits: int):
        # TODO: Restriction is because the ancilla free construction needs two source bits.
        #  I know how to fix this but haven't implemented it yet.
        if source_bits < 2 or target_bits < source_bits:
            raise ValueError()
        self.source_bits = source_bits
        self.target_bits = target_bits

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return { "source_bits": self.source_bits, "target_bits": self.target_bits }

    def qubits_in(self) -> QubitMap:
        return QubitMap(self.source_bits + self.target_bits)

    def qubits_out(self) -> QubitMap:
        return QubitMap(self.source_bits + self.target_bits)

    def normalization(self) -> float:
        return 1

    def phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        old_shape = input.shape
        N = 2 ** self.source_bits
        M = 2 ** self.target_bits
        input = input.reshape((-1, M, N))
        result = np.zeros_like(input)
        for val in range(N):
            result[:, :, val] += np.roll(input[:, :, val], val, axis=-1)
        return result.reshape(old_shape)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        old_shape = input.shape
        N = 2 ** self.source_bits
        M = 2 ** self.target_bits
        input = input.reshape((-1, M, N))
        result = np.zeros_like(input)
        for val in range(N):
            result[:, :, val] += np.roll(input[:, :, val], -val, axis=-1)
        return result.reshape(old_shape)

    def circuit(self) -> Circuit:
        source = list(reversed(range(self.source_bits)))
        target = list(reversed(range(self.source_bits, self.source_bits + self.target_bits)))
        circuit = addition_circuit(source, target)
        return Circuit(circuit)
