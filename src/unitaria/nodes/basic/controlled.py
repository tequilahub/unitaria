from typing import Sequence

import numpy as np
import tequila as tq

from unitaria.subspace import Subspace
from unitaria.nodes.node import Node
from unitaria.circuit import Circuit


class Controlled(Node):
    def __init__(self, A: Node):
        super().__init__(A.dimension_in + 1, A.dimension_out + 1)
        self.A = A

    def children(self) -> list[Node]:
        return [self.A]

    def _subspace_in(self) -> Subspace:
        subspace_in_A = self.A.subspace_in
        return Subspace("0" * subspace_in_A.total_qubits) | subspace_in_A

    def _subspace_out(self) -> Subspace:
        subspace_out_A = self.A.subspace_out
        return Subspace("0" * subspace_out_A.total_qubits) | subspace_out_A

    def _normalization(self) -> float:
        return self.A.normalization

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        control_qubit = target[self.A.target_qubit_count()]
        return self.A.circuit(
            target[: self.A.target_qubit_count()], clean_ancillae, borrowed_ancillae, control=control_qubit
        )

    def _controlled_circuit(
        self, control: int, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        # The controlled version of controls, of course, has two control bits.
        # The last target bit and `control`
        control_qubit = target[self.A.target_qubit_count()]
        if len(clean_ancillae) > self.A.clean_ancilla_count():
            # There is an extra ancilla which we can use to compute the control
            # for A
            ancilla = clean_ancillae[-1]
            circuit = Circuit()
            circuit += tq.gates.Toffoli(control, control_qubit, ancilla)
            circuit += self.A.circuit(
                target[: self.A.target_qubit_count()], clean_ancillae[:-1], borrowed_ancillae, control=ancilla
            )
            circuit += tq.gates.Toffoli(control, control_qubit, ancilla)
            return circuit
        else:
            return self.A.circuit(
                target[: self.A.target_qubit_count()], clean_ancillae, borrowed_ancillae, control=control_qubit
            ).add_controls([control])

    def clean_ancilla_count(self) -> int:
        return self.A.clean_ancilla_count()

    def borrowed_ancilla_count(self) -> int:
        return self.A.borrowed_ancilla_count()
