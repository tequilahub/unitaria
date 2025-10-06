from typing import Sequence

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
        super().__init__(
            (A.dimension_in - 1) * expand_control.dimension + 1, (A.dimension_out - 1) * expand_control.dimension + 1
        )
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

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        subspace = self.A.subspace_in
        control_qubit_pre = target[subspace.total_qubits - 1]
        control_qubit_post = target[subspace.total_qubits + self.expand_control.total_qubits - 1]

        circuit = Circuit()
        if self.swap_control_state:
            circuit += tq.gates.X(control_qubit_post)

        original_circuit = self.A.circuit(target, clean_ancillae, borrowed_ancillae)
        qubit_map = {t: t for t in range(original_circuit.n_qubits)}
        qubit_map[control_qubit_pre] = control_qubit_post
        circuit += original_circuit.map_qubits(qubit_map)
        if self.swap_control_state:
            circuit += tq.gates.X(control_qubit_post)
        return circuit

    def clean_ancilla_count(self) -> int:
        return self.A.clean_ancilla_count()

    def borrowed_ancilla_count(self) -> int:
        return self.A.borrowed_ancilla_count()
