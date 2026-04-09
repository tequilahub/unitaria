from typing import Sequence
import numpy as np
import tequila as tq

from unitaria.circuit import Circuit
from unitaria.subspace import Subspace
from unitaria.nodes.node import Node


class Identity(Node):
    """
    Node representing the identity matrix on a given vectorspace

    Should specify exactly one of ``subspace`` or ``dim``

    :param subspace:
        The domain of the identity matrix
    :param dim:
        The dimension of the domain
    """

    subspace: Subspace

    def __init__(self, subspace: Subspace | None = None, *, dim: int | None = None):
        if dim is not None:
            if subspace is not None:
                raise ValueError("Either subspace or dim has to be specified")
            self.subspace = Subspace.from_dim(dim)
        else:
            if subspace is None:
                raise ValueError("Either subspace or dim has to be specified")
            self.subspace = subspace
        super().__init__(self.subspace.dimension, self.subspace.dimension)

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        params = {}
        params["subspace"] = self.subspace
        return params

    def _subspace_in(self) -> Subspace:
        return self.subspace

    def _subspace_out(self) -> Subspace:
        return self.subspace

    def _normalization(self) -> float:
        return 1

    def is_guaranteed_unitary(self) -> bool:
        return True

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        circuit = Circuit()
        for qubit in target:
            # TODO: Replace with identity gate once it's fixed
            circuit += tq.gates.Rx(target=qubit, angle=0)
        return circuit

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 0
