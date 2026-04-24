from __future__ import annotations
from typing import Sequence
import numpy as np
import tequila as tq

from unitaria.circuit import Circuit
from unitaria.subspace import Subspace
from unitaria.nodes.node import Node


class Scale(Node):
    """
    Node representing the product of a scalar and another node

    :param A: The node to scale.
    :param scale: The scalar factor (default: 1).
    :param remove_efficiency: (Unused, must be 1).
    :param absolute: If True, A is divided by its normalization first (default: False).
    :raises AssertionError: If remove_efficiency is not 1.
    """

    A: Node
    scale: float
    remove_efficiency: float
    absolute: bool

    def __init__(
        self,
        A: Node,
        scale: float = 1,
        remove_efficiency: None | float = None,
        absolute: bool = False,
    ):
        super().__init__(A.dimension_in, A.dimension_out)
        self.A = A
        assert remove_efficiency is None or remove_efficiency > 1
        self.remove_efficiency = remove_efficiency
        self.scale = np.abs(scale)
        self.global_phase = np.angle(scale)
        self.absolute = absolute

    def children(self) -> list[Node]:
        return [self.A]

    def parameters(self) -> dict:
        return {"scale": self.scale, "absolute": self.absolute}

    def _subspace_in(self) -> Subspace:
        if self.remove_efficiency is None:
            return self.A.subspace_in
        else:
            return Subspace("0") & self.A.subspace_in

    def _subspace_out(self) -> Subspace:
        if self.remove_efficiency is None:
            return self.A.subspace_out
        else:
            return Subspace("0") & self.A.subspace_out

    def _normalization(self) -> float:
        remove_efficiency = 1 if self.remove_efficiency is None else self.remove_efficiency
        if self.absolute:
            return self.scale * remove_efficiency
        else:
            return self.scale * self.A.normalization * remove_efficiency

    def is_guaranteed_unitary(self) -> bool:
        return self.remove_efficiency == 1 and self.A.is_guaranteed_unitary()

    def compute(self, input: np.ndarray | None = None) -> np.ndarray:
        if self.absolute:
            return np.exp(1j * self.global_phase) * self.scale / self.A.normalization * self.A.compute(input)
        else:
            return np.exp(1j * self.global_phase) * self.scale * self.A.compute(input)

    def compute_adjoint(self, input: np.ndarray | None = None) -> np.ndarray:
        if self.absolute:
            return np.exp(-1j * self.global_phase) * self.scale / self.A.normalization * self.A.compute_adjoint(input)
        else:
            return np.exp(-1j * self.global_phase) * self.scale * self.A.compute_adjoint(input)

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        circuit = Circuit()
        A_target = target
        if self.remove_efficiency is not None:
            A_target = target[:-1]
            circuit += tq.gates.Ry(2 * np.arccos(1 / self.remove_efficiency), target[-1])
        circuit += self.A.circuit(A_target, clean_ancillae, borrowed_ancillae)
        circuit += tq.gates.GlobalPhase(self.global_phase)
        return circuit

    def clean_ancilla_count(self) -> int:
        return self.A.clean_ancilla_count()

    def borrowed_ancilla_count(self) -> int:
        return self.A.borrowed_ancilla_count()


Node.__rmul__ = lambda A, s: Scale(A, s)
