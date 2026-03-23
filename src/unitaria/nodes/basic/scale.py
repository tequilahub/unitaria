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

    :param A:
        The node to scale
    :param scale:
        The scalar factor
    :param absolute:
        If ``True``, ``A`` is divided by its normalization first
    """

    A: Node
    scale: float
    remove_efficiency: float
    absolute: bool

    def __init__(
        self,
        A: Node,
        scale: float = 1,
        remove_efficiency: float = 1,
        absolute: bool = False,
    ):
        super().__init__(A.dimension_in, A.dimension_out)
        self.A = A
        # TODO: remove_efficiency not implemented yet
        assert remove_efficiency == 1
        self.remove_efficiency = remove_efficiency
        self.scale = np.abs(scale)
        self.global_phase = np.angle(scale)
        self.absolute = absolute

    def children(self) -> list[Node]:
        """
        Returns the child nodes of this node.
        """
        return [self.A]

    def parameters(self) -> dict:
        """
        Returns the parameters of the scale node.
        """
        return {"scale": self.scale, "absolute": self.absolute}

    def _subspace_in(self) -> Subspace:
        """
        Returns the input subspace.
        """
        return self.A.subspace_in

    def _subspace_out(self) -> Subspace:
        """
        Returns the output subspace.
        """
        return self.A.subspace_out

    def _normalization(self) -> float:
        """
        Returns the normalization factor for this node.
        """
        if self.absolute:
            return self.scale
        else:
            return self.scale * self.A.normalization

    def compute(self, input: np.ndarray | None = None) -> np.ndarray:
        """
        Computes the scaled output for the given input.

        Args:
            input (np.ndarray, optional): Input state vector.

        Returns:
            np.ndarray: Scaled output state vector.
        """
        if self.absolute:
            return np.exp(1j * self.global_phase) * self.scale / self.A.normalization * self.A.compute(input)
        else:
            return np.exp(1j * self.global_phase) * self.scale * self.A.compute(input)

    def compute_adjoint(self, input: np.ndarray | None = None) -> np.ndarray:
        """
        Computes the adjoint (inverse) scaled output for the given input.

        Args:
            input (np.ndarray, optional): Input state vector.

        Returns:
            np.ndarray: Scaled output state vector.
        """
        if self.absolute:
            return np.exp(-1j * self.global_phase) * self.scale / self.A.normalization * self.A.compute_adjoint(input)
        else:
            return np.exp(-1j * self.global_phase) * self.scale * self.A.compute_adjoint(input)

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        """
        Returns the quantum circuit implementing this scaled node.

        Args:
            target (Sequence[int]): Target qubits.
            clean_ancillae (Sequence[int]): Clean ancilla qubits.
            borrowed_ancillae (Sequence[int]): Borrowed ancilla qubits.

        Returns:
            Circuit: The constructed quantum circuit.
        """
        circuit = Circuit()
        circuit += self.A.circuit(target, clean_ancillae, borrowed_ancillae)
        circuit += tq.gates.GlobalPhase(self.global_phase)
        return circuit

    def clean_ancilla_count(self) -> int:
        """
        Returns the number of clean ancilla qubits required.
        """
        return self.A.clean_ancilla_count()

    def borrowed_ancilla_count(self) -> int:
        """
        Returns the number of borrowed ancilla qubits required.
        """
        return self.A.borrowed_ancilla_count()


Node.__rmul__ = lambda A, s: Scale(A, s)
