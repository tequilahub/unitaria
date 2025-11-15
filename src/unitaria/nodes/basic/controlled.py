from typing import Sequence

import numpy as np

from unitaria.subspace import Subspace, ControlledSubspace
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
        return Subspace([ControlledSubspace(Subspace(0, subspace_in_A.total_qubits), subspace_in_A)])

    def _subspace_out(self) -> Subspace:
        subspace_out_A = self.A.subspace_out
        return Subspace([ControlledSubspace(Subspace(0, subspace_out_A.total_qubits), subspace_out_A)])

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
        circuit = self.A.circuit(target[: self.A.target_qubit_count()], clean_ancillae, borrowed_ancillae)
        return circuit.add_controls(control_qubit)

    def clean_ancilla_count(self) -> int:
        return self.A.clean_ancilla_count()

    def borrowed_ancilla_count(self) -> int:
        return self.A.borrowed_ancilla_count()
