from unitaria.nodes.node import Node
from unitaria.subspace import Subspace
from unitaria.circuit import Circuit
import numpy as np


class BlockEncoding(Node):
    """
    this class provides a block encoding of a given matrix using a quantum circuit
    the encoding has the form: subspace_in * circuit * subspace_out
    the user has to specify only incomping and outgoing subspaces, a circuit, and a normalization factor
    """

    def __init__(
        self,
        circuit: Circuit,
        subspace_in: Subspace,
        subspace_out: Subspace,
        normalization: float,
    ):
        super().__init__(subspace_in.dimension, subspace_out.dimension)
        self._circuit_obj = circuit
        self._subspace_in_obj = subspace_in
        self._subspace_out_obj = subspace_out
        self._normalization_value = normalization

    def _subspace_in(self) -> Subspace:
        return self._subspace_in_obj

    def _subspace_out(self) -> Subspace:
        return self._subspace_out_obj

    def _normalization(self) -> float:
        return self._normalization_value

    def _circuit(self) -> Circuit:
        return self._circuit_obj

    def compute(self, input: np.ndarray) -> np.ndarray:
        return self.simulate(input)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        # Adjoint circuit, project into subspace_out, then subspace_in
        projected_input = self.subspace_out.project(input)
        output = self.circuit.adjoint().simulate(projected_input)
        projected_output = self.subspace_in.project(output)
        return self.normalization * projected_output
