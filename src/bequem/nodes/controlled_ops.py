import numpy as np
import tequila as tq

from bequem.subspace import Subspace, ControlledSubspace
from bequem.nodes.basic_ops import UnsafeMul
from bequem.nodes.identity import Identity
from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.node import Node
from bequem.circuit import Circuit


class BlockDiagonal(ProxyNode):

    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        assert np.isclose(A.normalization, B.normalization)

        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self):
        A_controlled = Controlled(self.A)
        B_controlled = Controlled(self.B)
        subspace_in = _controlled_qubits(A_controlled.subspace_in, B_controlled.subspace_in)
        subspace_mid = _controlled_qubits(A_controlled.subspace_out, B_controlled.subspace_in)
        subspace_out = _controlled_qubits(A_controlled.subspace_out, B_controlled.subspace_out)
        controlled_bits_A = A_controlled.subspace_in.total_qubits - A_controlled.subspace_in.trailing_zeros()
        controlled_bits_B = B_controlled.subspace_in.total_qubits - B_controlled.subspace_in.trailing_zeros()

        A_controlled = ModifyControl(
            A_controlled, max(0, controlled_bits_B - controlled_bits_A), True)
        A_controlled = UnsafeMul(
            UnsafeMul(
                Identity(subspace_in, A_controlled.subspace_in),
                A_controlled,
            ),
            Identity(A_controlled.subspace_out, subspace_mid),
        )
        B_controlled = ModifyControl(
            B_controlled, max(0, controlled_bits_A - controlled_bits_B), False)
        B_controlled = UnsafeMul(
            UnsafeMul(
                Identity(subspace_in, B_controlled.subspace_in),
                B_controlled,
            ),
            Identity(B_controlled.subspace_out, subspace_out),
        )

        return UnsafeMul(A_controlled, B_controlled)

    def _normalization(self) -> float:
        return self.A.normalization

    def compute(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.subspace_in.dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        result_A = self.A.compute(input_A)
        result_B = self.B.compute(input_B)
        return np.concatenate((result_A, result_B), axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.subspace_out.dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        result_A = self.A.compute_adjoint(input_A)
        result_B = self.B.compute_adjoint(input_B)
        return np.concatenate((result_A, result_B), axis=-1)

def _controlled_qubits(A_controlled: Subspace, B_controlled: Subspace) -> Subspace:
    zeros = max(A_controlled.trailing_zeros(), B_controlled.trailing_zeros())
    A = A_controlled.case_one()
    B = B_controlled.case_one()
    controlled_qubits = max(A.total_qubits, B.total_qubits)
    case_zero = Subspace(
        A.registers, max(0, controlled_qubits - A.total_qubits))
    case_one = Subspace(
        B.registers, max(0, controlled_qubits - B.total_qubits))
    return Subspace([ControlledSubspace(case_zero, case_one)], zeros)


Node.__or__ = lambda A, B: BlockDiagonal(A, B)


class Controlled(Node):

    A: Node

    def __init__(self, A: Node):
        self.A = A

    def children(self) -> list[Node]:
        return [self.A]

    def _subspace_in(self) -> Subspace:
        subspace_in_A = self.A.subspace_in
        return Subspace([ControlledSubspace(Subspace(0, subspace_in_A.total_qubits), subspace_in_A)])

    def _subspace_out(self) -> Subspace:
        subspace_out_A = self.A.subspace_out
        return Subspace([ControlledSubspace(Subspace(0, subspace_out_A.total_qubits), subspace_out_A)])

    def _normalization(self) -> float:
        return self.A.normalization

    def _phase(self) -> float:
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def _circuit(self) -> Circuit:
        control_qubit = self.A.subspace_in.total_qubits
        circuit = self.A.circuit.tq_circuit
        circuit = circuit.add_controls(control_qubit)
        circuit += tq.gates.Phase(target=control_qubit, angle=self.A.phase)
        circuit.n_qubits = control_qubit + 1
        return Circuit(circuit)


class ModifyControl(Node):
    A: Node
    expand_control: Subspace
    swap_control_state: bool

    def __init__(self, A: Node, expand_control: Subspace | int = 0, swap_control_state: bool = False):
        self.A = A
        if not isinstance(expand_control, Subspace):
            expand_control = Subspace(expand_control)
        self.expand_control = expand_control
        self.swap_control_state = swap_control_state
    
    def children(self) -> list[Node]:
        return [self.A]

    def parameters(self) -> dict:
        return {"expand_control": self.expand_control, "swap_control_state": self.swap_control_state}

    def _subspace_in(self) -> Subspace:
        subspace_one = Subspace(self.A.subspace_in.case_one().registers + self.expand_control.registers)
        subspace_zero = Subspace(0, subspace_one.total_qubits)

        if self.swap_control_state:
            return Subspace([ControlledSubspace(subspace_one, subspace_zero)], self.A.subspace_in.trailing_zeros())
        else:
            return Subspace([ControlledSubspace(subspace_zero, subspace_one)], self.A.subspace_in.trailing_zeros())

    def _subspace_out(self) -> Subspace:
        subspace_one = Subspace(self.A.subspace_out.case_one().registers + self.expand_control.registers)
        subspace_zero = Subspace(0, subspace_one.total_qubits)

        if self.swap_control_state:
            return Subspace([ControlledSubspace(subspace_one, subspace_zero)], self.A.subspace_out.trailing_zeros())
        else:
            return Subspace([ControlledSubspace(subspace_zero, subspace_one)], self.A.subspace_out.trailing_zeros())

    def _normalization(self) -> float:
        return self.A.normalization

    def _phase(self) -> float:
        return self.A.phase

    def compute(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _circuit(self) -> Circuit:
        subspace = self.A.subspace_in
        control_qubit_pre = subspace.total_qubits - 1
        control_qubit_post = control_qubit_pre + self.expand_control.total_qubits

        circuit = Circuit()
        if self.swap_control_state:
            circuit.tq_circuit += tq.gates.X(control_qubit_post)

        qubit_map = dict([(i, i) for i in range(subspace.total_qubits)])
        qubit_map[control_qubit_pre] = control_qubit_post
        circuit.tq_circuit += self.A.circuit.tq_circuit.map_qubits(qubit_map)
        if self.swap_control_state:
            circuit.tq_circuit += tq.gates.X(control_qubit_post)
        circuit.n_qubits = self.subspace_in.total_qubits
        return circuit
