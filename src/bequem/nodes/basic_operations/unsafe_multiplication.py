from __future__ import annotations
import numpy as np

from bequem.circuit import Circuit
from bequem.subspace import Subspace
from bequem.nodes.node import Node


class UnsafeMul(Node):
    """
    Node for chaining the circuits of two nodes

    This is mostly for internal usage. To properly multiply two matrices use
    :py:class:`~bequem.nodes.prox_node.Mul` instead. The order of operations is
    such that the first argument ``A`` is applied first.

    :ivar A:
        The first factor
    :ivar B:
        The second factor
    """

    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        """
        The order of operations is such that the first argument ``A`` is applied
        first.

        :ivar A:
            The first factor
        :ivar B:
            The second factor
        """
        if not A.subspace_out.match_nonzero(B.subspace_in):
            raise ValueError(f"Non matching qubit maps {A.subspace_out} and {B.subspace_in}")

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

    def _circuit(self) -> Circuit:
        circuit = Circuit()
        circuit += self.A.circuit
        circuit += self.B.circuit

        return circuit

    def _subspace_in(self) -> Subspace:
        max_qubits = max(self.A.subspace_in.total_qubits, self.B.subspace_out.total_qubits)
        return Subspace(
            self.A.subspace_in.registers,
            max_qubits - self.A.subspace_in.total_qubits,
        )

    def _subspace_out(self) -> Subspace:
        max_qubits = max(self.A.subspace_in.total_qubits, self.B.subspace_out.total_qubits)
        return Subspace(
            self.B.subspace_out.registers,
            max_qubits - self.B.subspace_out.total_qubits,
        )

    def _normalization(self) -> float:
        return self.A.normalization * self.B.normalization
