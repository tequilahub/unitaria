from __future__ import annotations
import numpy as np

from bequem.circuit import Circuit
from bequem.subspace import Subspace
from bequem.nodes.node import Node


class Adjoint(Node):
    """
    Node representing the adjoint of another node

    :ivar A:
        The node of which to compute the adjoint
    """

    A: Node

    def __init__(self, A: Node):
        """
        :param A:
            The node of which to compute the adjoint
        """
        self.A = A

    def children(self) -> list[Node]:
        return [self.A]

    def _subspace_in(self) -> Subspace:
        return self.A.subspace_out

    def _subspace_out(self) -> Subspace:
        return self.A.subspace_in

    def _normalization(self) -> float:
        return self.A.normalization

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        return self.A.compute_adjoint(input)

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        return self.A.compute(input)

    def _circuit(self) -> Circuit:
        return self.A.circuit.adjoint()
