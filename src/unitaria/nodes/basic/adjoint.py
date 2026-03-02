from __future__ import annotations

from typing import Sequence

import numpy as np

from unitaria.circuit import Circuit
from unitaria.subspace import Subspace
from unitaria.nodes.node import Node


class Adjoint(Node):
    """
    Node representing the adjoint of another node

    :param A:
        The node of which to compute the adjoint
    """

    A: Node

    def __init__(self, A: Node):
        super().__init__(A.dimension_out, A.dimension_in)
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

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        return self.A.circuit(target, clean_ancillae, borrowed_ancillae).adjoint()

    def clean_ancilla_count(self) -> int:
        return self.A.clean_ancilla_count()

    def borrowed_ancilla_count(self) -> int:
        return self.A.borrowed_ancilla_count()


Node.adjoint = lambda x: Adjoint(x)
