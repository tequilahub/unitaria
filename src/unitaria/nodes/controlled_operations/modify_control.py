import numpy as np
import tequila as tq

from unitaria.subspace import Subspace, ControlledSubspace
from unitaria.nodes.node import Node
from unitaria.circuit import Circuit


class ModifyControl(Node):
    def __init__(
        self,
        A: Node,
        expand_control: Subspace | int = 0,
        swap_control_state: bool = False,
    ):
        self.A = A
        if not isinstance(expand_control, Subspace):
            expand_control = Subspace(expand_control)
        self.expand_control = expand_control
        self.swap_control_state = swap_control_state

    def children(self) -> list[Node]:
        return [self.A]

    def parameters(self) -> dict:
        return {
            "expand_control": self.expand_control,
            "swap_control_state": self.swap_control_state,
        }

    def _subspace_in(self) -> Subspace:
        subspace_one = Subspace(self.A.subspace_in.case_one().registers + self.expand_control.registers)
        subspace_zero = Subspace(0, subspace_one.total_qubits)

        if self.swap_control_state:
            return Subspace(
                [ControlledSubspace(subspace_one, subspace_zero)],
                self.A.subspace_in.trailing_zeros(),
            )
        else:
            return Subspace(
                [ControlledSubspace(subspace_zero, subspace_one)],
                self.A.subspace_in.trailing_zeros(),
            )

    def _subspace_out(self) -> Subspace:
        subspace_one = Subspace(self.A.subspace_out.case_one().registers + self.expand_control.registers)
        subspace_zero = Subspace(0, subspace_one.total_qubits)

        if self.swap_control_state:
            return Subspace(
                [ControlledSubspace(subspace_one, subspace_zero)],
                self.A.subspace_out.trailing_zeros(),
            )
        else:
            return Subspace(
                [ControlledSubspace(subspace_zero, subspace_one)],
                self.A.subspace_out.trailing_zeros(),
            )

    def _normalization(self) -> float:
        return self.A.normalization

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
