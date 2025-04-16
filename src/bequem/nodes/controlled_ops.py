import numpy as np
import tequila as tq

from bequem.qubit_map import QubitMap, Qubit
from bequem.circuit import Circuit
from bequem.nodes.node import Node


class BlockDiagonal(Node):
    def __init__(self, A: Node, B: Node):
        # TODO: Automatically use maximum normalization using Scale
        assert np.isclose(A.normalization(), B.normalization())
        # TODO
        assert A.qubits_in().total_qubits == B.qubits_in().total_qubits
        self.A = A
        self.B = B

    def qubits_in(self) -> QubitMap:
        return QubitMap([Qubit(self.A.qubits_in(), self.B.qubits_in())])

    def qubits_out(self) -> QubitMap:
        return QubitMap([Qubit(self.A.qubits_out(), self.B.qubits_out())])

    def normalization(self) -> float:
        return self.A.normalization()

    def compute(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_in().dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        result_A = self.A.compute(input_A)
        result_B = self.B.compute(input_B)
        return np.concatenate((result_A, result_B), axis=-1)

    def circuit(self) -> Circuit:
        circuit_A = self.A.circuit().tq_circuit
        circuit_B = self.B.circuit().tq_circuit
        control_qubit = circuit_A.n_qubits
        circuit_A.add_controls(control_qubit, inpl=True)
        circuit_B.add_controls(control_qubit, inpl=True)

        circuit = tq.QCircuit()
        circuit += tq.gates.X(target=control_qubit)
        circuit += circuit_A
        circuit += tq.gates.X(target=control_qubit)
        circuit += circuit_B

        return Circuit(circuit)


Node.__or__ = lambda A, B: BlockDiagonal(A, B)


class Add(Node):
    def __init__(self, A: Node, B: Node):
        # TODO
        assert A.qubits_in() == B.qubits_in()
        assert A.qubits_out() == B.qubits_out()
        self.A = A
        self.B = B

    def qubits_in(self) -> QubitMap:
        # TODO: Stimmt so nicht
        return QubitMap(self.A.qubits_in().registers, 1)

    def qubits_out(self) -> QubitMap:
        # TODO: Stimmt so nicht
        return QubitMap(self.A.qubits_out().registers, 1)

    def normalization(self) -> float:
        return self.A.normalization() + self.B.normalization()

    def compute(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute(input) + self.B.compute(input)

    def circuit(self) -> Circuit:
        circuit_A = self.A.circuit().tq_circuit
        circuit_B = self.B.circuit().tq_circuit
        control_qubit = circuit_A.n_qubits
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


Node.__add__ = lambda A, B: Add(A, B)


class BlockHorizontal(Node):
    pass


class BlockVertical(Node):
    pass
