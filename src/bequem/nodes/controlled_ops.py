import numpy as np
import tequila as tq

from bequem.qubit_map import QubitMap, Qubit
from bequem.circuit import Circuit
from bequem.nodes.node import Node
from bequem.permutation import find_permutation


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
        control_qubit = max(circuit_A.n_qubits, circuit_B.n_qubits)
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
        self.perm_in_A, self.perm_in_B, qubits_in = find_permutation(
            A.qubits_in(), B.qubits_in()
        )
        self._qubits_in = QubitMap(qubits_in.registers, qubits_in.zero_qubits + 1)
        self.perm_out_A, self.perm_out_B, qubits_out = find_permutation(
            A.qubits_out(), B.qubits_out()
        )
        self._qubits_out = QubitMap(qubits_out.registers, qubits_out.zero_qubits + 1)
        self.A = A
        self.B = B

    def qubits_in(self) -> QubitMap:
        return self._qubits_in

    def qubits_out(self) -> QubitMap:
        return self._qubits_out

    def normalization(self) -> float:
        # TODO: This seems to be necessary because Tequila's phase gate seems to implement
        #  diag(exp(-i theta / 2), exp(i theta / 2)) instead of diag(1, exp(i theta)).
        phase = (np.angle(self.A.normalization()) + np.angle(self.B.normalization())) / 2
        return np.exp(phase * 1j) * (np.abs(self.A.normalization()) + np.abs(self.B.normalization()))

    def compute(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute(input) + self.B.compute(input)

    def circuit(self) -> Circuit:
        circuit_A = (
            self.perm_in_A.tq_circuit
            + self.A.circuit().tq_circuit
            + self.perm_out_A.tq_circuit
        )
        circuit_B = (
            self.perm_in_B.tq_circuit
            + self.B.circuit().tq_circuit
            + self.perm_out_B.tq_circuit
        )
        control_qubit = self.qubits_in().total_qubits - 1
        circuit_A.add_controls(control_qubit, inpl=True)
        circuit_B.add_controls(control_qubit, inpl=True)
        norm_A = self.A.normalization()
        norm_B = self.B.normalization()
        angle = np.arctan2(
            np.sqrt(np.abs(norm_B)),
            np.sqrt(np.abs(norm_A))
        )
        phase = np.angle(norm_B) - np.angle(norm_A)

        circuit = tq.QCircuit()
        circuit += tq.gates.Ry(2 * angle, target=control_qubit)
        circuit += tq.gates.X(target=control_qubit)
        circuit += circuit_A
        circuit += tq.gates.X(target=control_qubit)
        circuit += tq.gates.Phase(angle=phase, target=control_qubit)
        circuit += circuit_B
        circuit += tq.gates.Ry(-2 * angle, target=control_qubit)

        return Circuit(circuit)


Node.__add__ = lambda A, B: Add(A, B)


class BlockHorizontal(Node):
    pass


class BlockVertical(Node):
    pass
