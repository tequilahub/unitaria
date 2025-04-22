from __future__ import annotations
import numpy as np
import tequila as tq

from bequem.circuit import Circuit
from bequem.qubit_map import QubitMap
from bequem.nodes.node import Node
from bequem.permutation import find_permutation


class Mul(Node):
    def __init__(self, A: Node, B: Node):
        self.perm_A, self.perm_B, self.qubits_middle = find_permutation(
            A.qubits_out(), B.qubits_in()
        )
        self.qubits_middle = QubitMap(
            self.qubits_middle.registers, self.qubits_middle.zero_qubits + 1
        )
        qubits_in_A = A.qubits_in()
        self._qubits_in = QubitMap(
            qubits_in_A.registers,
            self.qubits_middle.total_qubits
            - qubits_in_A.total_qubits
            + qubits_in_A.zero_qubits,
        )
        qubits_out_B = B.qubits_out()
        self._qubits_out = QubitMap(
            qubits_out_B.registers,
            self.qubits_middle.total_qubits
            - qubits_out_B.total_qubits
            + qubits_out_B.zero_qubits,
        )
        self.A = A
        self.B = B

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        input = self.A.compute(input)
        input = self.B.compute(input)
        return input

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        input = self.B.compute_adjoint(input)
        input = self.A.compute_adjoint(input)
        return input

    def circuit(self) -> Circuit:
        circuit = Circuit()
        circuit += self.A.circuit()
        circuit += self.perm_A
        circuit += self.qubits_middle.circuit()
        circuit.tq_circuit += self.perm_B.tq_circuit.dagger()
        circuit += self.B.circuit()

        return circuit

    def qubits_in(self) -> QubitMap:
        return self._qubits_in

    def qubits_out(self) -> QubitMap:
        return self._qubits_out

    def normalization(self) -> float:
        return self.A.normalization() * self.B.normalization()


Node.__matmul__ = lambda A, B: Mul(A, B)


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

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        batch_shape = list(input.shape[:-1])
        input = input.reshape(
            batch_shape + [self.B.qubits_out().dimension, self.A.qubits_out().dimension]
        )
        input = self.A.compute_adjoint(input)
        input = np.swapaxes(input, -1, -2)
        input = self.B.compute_adjoint(input)
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
    def __init__(self, A: Node):
        self.A = A

    def qubits_in(self) -> QubitMap:
        return self.A.qubits_out()

    def qubits_out(self) -> QubitMap:
        return self.A.qubits_in()

    def normalization(self) -> float:
        # TODO: Should normalization be a complex number
        return self.A.normalization()

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        return self.A.compute_adjoint(input)

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        return self.A.compute(input)

    def circuit(self) -> Circuit:
        return self.A.circuit().adjoint()


class Scale(Node):
    def __init__(
        self,
        A: Node,
        scale: float = 1,
        remove_efficiency: float = 1,
        scale_absolute=False,
    ):
        self.A = A
        self.scale = scale
        # TODO: assert_efficiency not implemented yet
        assert remove_efficiency == 1
        self.remove_efficiency = remove_efficiency
        # TODO: scale_absolute not implemented yet
        assert not scale_absolute
        self.scale_absolute = scale_absolute

    def qubits_in(self) -> QubitMap:
        return self.A.qubits_in()

    def qubits_out(self) -> QubitMap:
        return self.A.qubits_out()

    def normalization(self) -> float:
        return self.scale * self.A.normalization()

    def compute(self, input: np.ndarray | None = None) -> np.ndarray:
        return self.scale * self.A.compute(input)

    def compute_adjoint(self, input: np.ndarray | None = None) -> np.ndarray:
        return self.scale * self.A.compute(input)

    def circuit(self) -> Circuit:
        return self.A.circuit()
