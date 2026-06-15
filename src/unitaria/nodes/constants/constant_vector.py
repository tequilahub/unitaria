from typing import Sequence

import numpy as np
import tequila as tq

from unitaria.circuit import Circuit
from unitaria.circuits.state_prep import prepare_state
from unitaria.nodes.node import Node
from unitaria.subspace import Subspace


class ConstantVector(Node):
    """
    Node representing the given vector

    :param vec: The vector represented by this node
    """

    vec: np.ndarray

    def __init__(self, vec: np.ndarray | int, dim: int | None = None):
        """
        Initialize a ConstantVector node.

        :param vec:
            The vector to be represented/prepared. If this is an integer, it is
            interpreted as the vector with component $1$ at that index, and 0
            elsewhere. In this case, ``dim`` has to be given.
        :param dim:
            The dimension of the vector. Only required if ``vec`` is an integer.
        :raises ValueError:
            If ``vec`` is an integer, but ``dim`` is not given, or if ``vec`` is
            an array with a dimension that does not match ``dim``.
        """
        if dim is None:
            if not isinstance(vec, np.ndarray):
                raise ValueError("If `vec` is an integer, then `dim` must be given.")
            super().__init__(1, vec.shape[0])
        else:
            if isinstance(vec, np.ndarray):
                if dim != vec.shape[0]:
                    raise ValueError(f"dim = {dim} does not match dimension of vector {vec.shape[0]}.")
            else:
                if vec >= dim:
                    raise ValueError(f"{vec} is not a valid index for dimension {dim}.")
            super().__init__(1, dim)
        self.vec = vec
        self.dim = dim
        self.n_qubits = int(np.ceil(np.log2(self.dimension_out)))

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        parameters = {"vec": self.vec}
        if self.dim is not None:
            parameters["dim"] = self.dim
        return parameters

    def _subspace_in(self) -> Subspace:
        return Subspace("0" * self.n_qubits)

    def _subspace_out(self) -> Subspace:
        return Subspace.from_dim(self.dimension_out, bits=self.n_qubits)

    def _normalization(self) -> float:
        if isinstance(self.vec, int):
            return 1
        return np.linalg.norm(self.vec)

    def compute(self, input: np.ndarray) -> np.ndarray:
        vec = self.vec
        if isinstance(self.vec, int):
            vec = np.zeros(self.dimension_out)
            vec[self.vec] = 1
        if input.ndim == 1:
            return vec * input[0]
        else:
            return (np.array([vec]).T @ input.T).T

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        vec = self.vec
        if isinstance(self.vec, int):
            vec = np.zeros(self.dimension_out)
            vec[self.vec] = 1
        return (np.array([np.conj(vec)]) @ input.T).T

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        if isinstance(self.vec, int):
            circuit = Circuit()
            circuit.n_qubits = self.n_qubits
            for i in range(self.n_qubits):
                if (self.vec >> i) & 1 != 0:
                    circuit += tq.gates.X(target=i)
            return circuit
        if self.vec.shape[0] == 1:
            return Circuit()
        if self.normalization < 1e-8:
            circuit = Circuit()
            circuit.n_qubits = self.n_qubits
            return circuit
        normalized = self.vec / self.normalization
        normalized = np.concatenate((normalized, np.zeros(2**self.n_qubits - self.vec.shape[0])))
        tq_circuit = prepare_state(normalized, target)
        return Circuit(tq_circuit)

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 0
