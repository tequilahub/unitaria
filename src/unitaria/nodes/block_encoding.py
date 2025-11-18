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

    def _circuit(self, target, clean_ancillae=None, borrowd_ancillae=None) -> Circuit:
        return self._circuit_obj

    def clean_ancilla_count(self):
        return 0

    def borrowed_ancilla_count(self):
        return 0

    def compute(self, input: np.ndarray) -> np.ndarray:
        # handle both vector and matrix cases
        if input.ndim == 1:
            # extend input vector to full hilbert space size
            full_input = np.zeros(2**self._circuit_obj.n_qubits, dtype=np.complex128)
            basis_indices = self._subspace_in_obj.enumerate_basis()
            full_input[basis_indices] = input
            output = self.simulate(full_input)
            # project output onto subspace_out
            return self._subspace_out_obj.project(output)
        else:
            # multiple dimensions (batch): apply simulate to each vector in the batch
            results = np.zeros((input.shape[0], self._subspace_out_obj.dimension), dtype=np.complex128)
            for idx in np.ndindex(input.shape[:-1]):
                full_input = np.zeros(2**self._circuit_obj.n_qubits, dtype=np.complex128)
                basis_indices = self._subspace_in_obj.enumerate_basis()
                full_input[basis_indices] = input[idx]
                output = self.simulate(full_input)
                # project output onto subspace_out
                results[idx] = self._subspace_out_obj.project(output)
            return results

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        # project input into subspace_out, handle both vector and matrix cases
        if input.ndim == 1:
            projected_input = self._subspace_out_obj.project(input)
            output = self._circuit_obj.adjoint().simulate(projected_input)
            projected_output = self._subspace_in_obj.project(output)
        else:
            results = np.zeros((input.shape[0], self._subspace_in_obj.dimension), dtype=np.complex128)
            for idx in np.ndindex(input.shape[:-1]):
                proj_in = self._subspace_out_obj.project(input[idx])
                out = self._circuit_obj.adjoint().simulate(proj_in)
                results[idx] = self._subspace_in_obj.project(out)
            projected_output = results
        return self._normalization_value * projected_output
