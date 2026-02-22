from __future__ import annotations

from typing import Sequence

import numpy as np

from unitaria.circuit import Circuit
from unitaria.subspace import Subspace
from unitaria.nodes.node import Node


class UnsafeMul(Node):
    """
    Node for chaining the circuits of two nodes

    This is mostly for internal usage. To properly multiply two matrices use
    `~unitaria.nodes.prox_node.Mul` instead. The order of operations is such that
    the first argument ``A`` is applied first.

    :param A:
        The first factor
    :param B:
        The second factor
    """

    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        if not A.subspace_out.match_nonzero(B.subspace_in):
            raise ValueError(f"Non matching qubit maps {A.subspace_out} and {B.subspace_in}")

        super().__init__(A.dimension_in, B.dimension_out)

        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        input = self.A.compute(input)
        input = self.B.compute(input)
        return input

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        input = self.B.compute_adjoint(input)
        input = self.A.compute_adjoint(input)
        return input

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        circuit = Circuit()
        circuit += self.A.circuit(target, clean_ancillae, borrowed_ancillae)
        circuit += self.B.circuit(target, clean_ancillae, borrowed_ancillae)
        return circuit

    def _subspace_in(self) -> Subspace:
        max_qubits = max(self.A.subspace_in.total_qubits, self.B.subspace_out.total_qubits)
        return Subspace(
            registers=self.A.subspace_in.registers,
            zero_qubits=max_qubits - self.A.subspace_in.total_qubits,
        )

    def _subspace_out(self) -> Subspace:
        max_qubits = max(self.A.subspace_in.total_qubits, self.B.subspace_out.total_qubits)
        return Subspace(
            registers=self.B.subspace_out.registers,
            zero_qubits=max_qubits - self.B.subspace_out.total_qubits,
        )

    def _normalization(self) -> float:
        return self.A.normalization * self.B.normalization

    def clean_ancilla_count(self) -> int:
        return max(self.A.clean_ancilla_count(), self.B.clean_ancilla_count())

    def borrowed_ancilla_count(self) -> int:
        return max(self.A.borrowed_ancilla_count(), self.B.borrowed_ancilla_count())
