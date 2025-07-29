from __future__ import annotations
import numpy as np
import tequila as tq

from bequem.circuit import Circuit
from bequem.subspace import Subspace
from bequem.nodes.node import Node


class Scale(Node):
    """
    Node representing the product of a scalar and another node

    :ivar A:
        The node to scale
    :ivar scale:
        The scalar factor
    :ivar absolute:
        If ``True``, ``A`` is divided by its normalization first
    """

    def __init__(
        self,
        A: Node,
        scale: float = 1,
        remove_efficiency: float = 1,
        absolute: bool = False,
    ):
        self.A = A
        # TODO: assert_efficiency not implemented yet
        assert remove_efficiency == 1
        self.remove_efficiency = remove_efficiency
        self.scale = np.abs(scale)
        self.global_phase = np.angle(scale)
        self.absolute = absolute

    def children(self) -> list[Node]:
        return [self.A]

    def parameters(self) -> dict:
        return {"scale": self.scale, "absolute": self.absolute}

    def _subspace_in(self) -> Subspace:
        return self.A.subspace_in

    def _subspace_out(self) -> Subspace:
        return self.A.subspace_out

    def _normalization(self) -> float:
        if self.absolute:
            return self.scale
        else:
            return self.scale * self.A.normalization

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

    def _circuit(self) -> Circuit:
        circuit = Circuit()
        circuit += self.A.circuit
        circuit.tq_circuit += tq.circuit.gates.GlobalPhase(self.global_phase)
        return circuit


Node.__rmul__ = lambda A, s: Scale(A, s)
