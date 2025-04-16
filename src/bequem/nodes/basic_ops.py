from __future__ import annotations
import numpy as np
import tequila as tq

from bequem.circuit import Circuit
from bequem.qubit_map import QubitMap
from bequem.nodes.node import Node


class Mul(Node):
    def __init__(self, A: Node, B: Node):
        assert A.qubits_out().simplify().reduce() == B.qubits_in().simplify().reduce()
        self.A = A
        self.B = B

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        input = self.A.compute(input)
        input = self.B.compute(input)
        return input

    def circuit(self) -> Circuit:
        circuit = Circuit()
        circuit.append(self.A.circuit())
        raise NotImplementedError
        # TODO: Qubit permutation
        circuit.append(self.B.circuit())

        return circuit

    def qubits_in(self) -> QubitMap:
        self.A.qubits_in()  # TODO: Stimmt noch nicht

    def qubits_out(self) -> QubitMap:
        self.B.qubits_out()

    def normalization(self) -> float:
        return self.A.normalization() * self.B.normalization()


class Tensor(Node):
    def __init__(self, A: Node, B: Node):
        self.A = A
        self.B = B

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        batch_shape = list(input.shape[:-1])
        input = input.reshape(
            batch_shape + [self.B.qubits_in().dimension, self.A.qubits_in().dimension]
        )
        input = self.A.compute(input)
        input = np.swapaxes(input, -1, -2)
        input = self.B.compute(input)
        input = np.swapaxes(input, -1, -2)
        return np.reshape(input, batch_shape + [-1])

    def circuit(self) -> Circuit:
        qubits_in_A = self.A.qubits_in()
        qubits_in_B = self.B.qubits_in()

        circuit = Circuit()

        for i in reversed(range(qubits_in_B.total_qubits - qubits_in_B.zero_qubits)):
            qubit1 = qubits_in_A.total_qubits + i
            qubit2 = qubits_in_A.total_qubits - qubits_in_A.zero_qubits + i
            if qubit1 != qubit2:
                circuit.tq_circuit += tq.gates.SWAP(qubit1, qubit2)

        circuit_A = self.A.circuit().tq_circuit
        circuit.tq_circuit += circuit_A
        qubit_map_B = dict(
            [(i, i + qubits_in_A.total_qubits) for i in range(qubits_in_B.total_qubits)]
        )
        circuit_B = self.B.circuit().tq_circuit.map_qubits(qubit_map_B)
        circuit.tq_circuit += circuit_B

        qubits_out_A = self.A.qubits_out()
        qubits_out_B = self.B.qubits_out()

        for i in range(qubits_out_B.total_qubits - qubits_out_B.zero_qubits):
            qubit1 = qubits_out_A.total_qubits + i
            qubit2 = qubits_out_A.total_qubits - qubits_out_A.zero_qubits + i
            if qubit1 != qubit2:
                circuit.tq_circuit += tq.gates.SWAP(qubit1, qubit2)

        return circuit

    def qubits_in(self) -> QubitMap:
        qubits_A = self.A.qubits_in()
        qubits_B = self.B.qubits_in()
        return QubitMap(
            qubits_A.registers + qubits_B.registers,
            qubits_A.zero_qubits + qubits_B.zero_qubits,
        )

    def qubits_out(self) -> QubitMap:
        qubits_A = self.A.qubits_out()
        qubits_B = self.B.qubits_out()
        return QubitMap(
            qubits_A.registers + qubits_B.registers,
            qubits_A.zero_qubits + qubits_B.zero_qubits,
        )

    def normalization(self) -> float:
        return self.A.normalization() * self.B.normalization()


Node.__and__ = lambda A, B: Tensor(A, B)


class Adjoint(Node):
    pass


class Scale(Node):
    def __init__(
        self,
        A: Node,
        scale: float = 1,
        remove_efficiency: float = 1,
        scale_absolute=False,
    ):
        self.scale = scale
        self.remove_efficiency = remove_efficiency
        self.scale_absolute = scale_absolute
