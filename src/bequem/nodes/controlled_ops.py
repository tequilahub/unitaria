import numpy as np
import tequila as tq

from bequem.qubit_map import QubitMap, Qubit
from bequem.circuit import Circuit
from bequem.nodes.node import Node


class BlockDiagonal(Node):
    def __init__(self, A: Node, B: Node):
        # TODO: Automatically use maximum normalization using Scale
        assert np.isclose(A.normalization(), B.normalization())
        self.max_qubits = max(A.qubits_in().total_qubits, B.qubits_in().total_qubits)
        qubits_in_A = A.qubits_in()
        qubits_in_A = QubitMap(
            qubits_in_A.registers,
            self.max_qubits
            - qubits_in_A.total_qubits
            + qubits_in_A.zero_qubits,
        )
        qubits_out_A = A.qubits_out()
        qubits_out_A = QubitMap(
            qubits_out_A.registers,
            self.max_qubits
            - qubits_out_A.total_qubits
            + qubits_out_A.zero_qubits,
        )
        qubits_in_B = B.qubits_in()
        qubits_in_B = QubitMap(
            qubits_in_B.registers,
            self.max_qubits
            - qubits_in_B.total_qubits
            + qubits_in_B.zero_qubits,
        )
        qubits_out_B = B.qubits_out()
        qubits_out_B = QubitMap(
            qubits_out_B.registers,
            self.max_qubits
            - qubits_out_B.total_qubits
            + qubits_out_B.zero_qubits,
        )
        self._qubits_in = QubitMap([Qubit(qubits_in_A, qubits_in_B)])
        self._qubits_out = QubitMap([Qubit(qubits_out_A, qubits_out_B)])

        self.A = A
        self.B = B

    def qubits_in(self) -> QubitMap:
        return self._qubits_in

    def qubits_out(self) -> QubitMap:
        return self._qubits_out

    def normalization(self) -> float:
        return self.A.normalization()

    def compute(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_in().dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        result_A = self.A.compute(input_A)
        result_B = self.B.compute(input_B)
        return np.concatenate((result_A, result_B), axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_out().dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        result_A = self.A.compute_adjoint(input_A)
        result_B = self.B.compute_adjoint(input_B)
        return np.concatenate((result_A, result_B), axis=-1)

    def circuit(self) -> Circuit:
        circuit_A = self.A.circuit().tq_circuit
        circuit_B = self.B.circuit().tq_circuit
        control_qubit = self.max_qubits
        circuit_A.add_controls(control_qubit, inpl=True)
        circuit_B.add_controls(control_qubit, inpl=True)

        circuit = tq.QCircuit()
        circuit += tq.gates.X(target=control_qubit)
        circuit += circuit_A
        circuit += tq.gates.X(target=control_qubit)
        circuit += circuit_B

        return Circuit(circuit)


Node.__or__ = lambda A, B: BlockDiagonal(A, B)

class BlockHorizontal(Node):
    pass


class BlockVertical(Node):
    pass
