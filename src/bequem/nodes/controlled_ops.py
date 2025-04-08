import numpy as np
import tequila as tq

from bequem.qubit_map import QubitMap, Controlled, Qubit
from bequem.circuit import Circuit
from .node import Node


class BlockDiagonal(Node):
    def __init__(self, A: Node, B: Node):
        # TODO: Automatically use maximum normalization using Scale
        assert np.isclose(A.normalization(), B.normalization())
        # TODO
        assert A.qubits_in().total_qubits() == B.qubits_in().total_qubits()
        self.A = A
        self.B = B

    def qubits_in(self) -> QubitMap:
        return QubitMap([Controlled(self.A.qubits_in(), self.B.qubits_in())])

    def qubits_out(self) -> QubitMap:
        return QubitMap([Controlled(self.A.qubits_out(), self.B.qubits_out())])

    def normalization(self) -> float:
        return self.A.normalization()

    def compute(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_in().dimension()
        result_A = self.A.compute(input[:dim_A])
        result_B = self.B.compute(input[dim_A:])
        return np.concatenate((result_A, result_B))

    def circuit(self) -> Circuit:
        circuit_A = self.A.circuit().tq_circuit
        circuit_B = self.B.circuit().tq_circuit
        control_qubit = len(circuit_A.qubits)
        circuit_A.add_controls(control_qubit, inpl=True)
        circuit_B.add_controls(control_qubit, inpl=True)

        circuit = tq.QCircuit()
        circuit += tq.gates.X(target=control_qubit)
        circuit += circuit_A
        circuit += tq.gates.X(target=control_qubit)
        circuit += circuit_B

        return Circuit(circuit)


class Add(Node):
    def __init__(self, A: Node, B: Node):
        # TODO
        # assert A.qubits_in().reduce() == B.qubits_in().reduce()
        # assert A.qubits_out().reduce() == B.qubits_out().reduce()
        assert A.qubits_in().simplify() == B.qubits_in().simplify()
        assert A.qubits_out().simplify() == B.qubits_out().simplify()
        self.A = A
        self.B = B

    def qubits_in(self) -> QubitMap:
        return QubitMap(self.A.qubits_in().registers + [Qubit.ZERO])

    def qubits_out(self) -> QubitMap:
        return QubitMap(self.A.qubits_out().registers + [Qubit.ZERO])

    def normalization(self) -> float:
        return self.A.normalization() + self.B.normalization()

    def compute(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute(input) + self.B.compute(input)

    def circuit(self) -> Circuit:
        circuit_A = self.A.circuit().tq_circuit
        circuit_B = self.B.circuit().tq_circuit
        control_qubit = len(circuit_A.qubits)
        circuit_A.add_controls(control_qubit, inpl=True)
        circuit_B.add_controls(control_qubit, inpl=True)
        angle = np.arctan2(np.sqrt(self.B.normalization()), np.sqrt(self.A.normalization()))

        circuit = tq.QCircuit()
        circuit += tq.gates.Ry(2 * angle, target=control_qubit)
        circuit += tq.gates.X(target=control_qubit)
        circuit += circuit_A
        circuit += tq.gates.X(target=control_qubit)
        circuit += circuit_B
        circuit += tq.gates.Ry(-2 * angle, target=control_qubit)

        return Circuit(circuit)

class BlockHorizontal(Node):
    pass


class BlockVertical(Node):
    pass
