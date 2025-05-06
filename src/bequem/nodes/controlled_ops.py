import numpy as np
import tequila as tq

from bequem.qubit_map import QubitMap, padded_qubit
from bequem.circuit import Circuit
from bequem.nodes.basic_ops import UnsafeMul
from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.node import Node, Controlled


class BlockDiagonal(ProxyNode):

    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        assert np.isclose(A.normalization(), B.normalization())

        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self):
        A_controlled = self.A.controlled() or Controlled(
            self.A)
        B_controlled = self.B.controlled() or Controlled(
            self.B)
        A_controlled_m = ModifyControl(self.A, A_controlled, B_controlled.qubits_in().case_one(), True)
        B_controlled_m = ModifyControl(self.B, B_controlled, A_controlled.qubits_out().case_one(), False)

        return UnsafeMul(A_controlled_m, B_controlled_m)

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


Node.__or__ = lambda A, B: BlockDiagonal(A, B)


class ModifyControl(Node):
    A: Node
    A_controlled: Node
    qubits_B: QubitMap
    swap_control_state: bool

    def __init__(self, A: Node, A_controlled: Node, qubits_B: QubitMap, swap_control_state: bool):
        self.A = A
        self.A_controlled = A_controlled
        self.qubits_B = qubits_B
        self.swap_control_state = swap_control_state
    
    def children(self) -> list[Node]:
        return [self.A]

    def parameters(self) -> dict:
        return {"qubits_B": self.qubits_B, "swap_control_state": self.swap_control_state}

    def qubits_in(self) -> QubitMap:
        # TODO: Verify
        # assert qubits_in_A.registers == self.A.qubits_in().registers
        # assert self.A_controlled.qubits_in().select_case_zero() == QubitMap(qubits_in_A.total_qubits)
        qubits_in_A = self.A_controlled.qubits_in().case_one()

        if self.swap_control_state:
            return QubitMap([padded_qubit(qubits_in_A, self.qubits_B)])
        else:
            return QubitMap([padded_qubit(self.qubits_B, qubits_in_A)])

    def qubits_out(self) -> QubitMap:
        qubits_out_A = self.A_controlled.qubits_out().case_one()

        if self.swap_control_state:
            return QubitMap([padded_qubit(qubits_out_A, self.qubits_B)])
        else:
            return QubitMap([padded_qubit(self.qubits_B, qubits_out_A)])

    def normalization(self) -> float:
        return self.A_controlled.normalization()

    def phase(self) -> float:
        return self.A_controlled.phase()

    def compute(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_in().dimension
        if self.swap_control_state:
            input_A, remainder = np.split(input, [dim_A], axis=-1)
            result_A = self.A.compute(input_A)
            return np.concatenate((result_A, remainder), axis=-1)
        else:
            remainder, input_A = np.split(input, [-dim_A], axis=-1)
            result_A = self.A.compute(input_A)
            return np.concatenate((remainder, result_A), axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_out().dimension
        if self.swap_control_state:
            input_A, remainder = np.split(input, [dim_A], axis=-1)
            result_A = self.A.compute(input_A)
            return np.concatenate_adojoint((result_A, remainder), axis=-1)
        else:
            remainder, input_A = np.split(input, [-dim_A], axis=-1)
            result_A = self.A.compute_adjoint(input_A)
            return np.concatenate((remainder, result_A), axis=-1)

    def circuit(self) -> Circuit:
        qubits_in_A_controlled = self.A_controlled.qubits_in()
        control_qubit_pre = qubits_in_A_controlled.total_qubits - 1
        control_qubit_post = self.qubits_in().total_qubits - 1

        circuit = Circuit()
        if self.swap_control_state:
            circuit.tq_circuit += tq.gates.X(control_qubit_post)

        qubit_map = dict([(i, i) for i in range(qubits_in_A_controlled.total_qubits)])
        qubit_map[control_qubit_pre] = control_qubit_post
        circuit.tq_circuit += self.A_controlled.circuit().tq_circuit.map_qubits(qubit_map)
        if self.swap_control_state:
            circuit.tq_circuit += tq.gates.X(control_qubit_post)
        circuit.n_qubits = self.qubits_in().total_qubits
        return circuit
