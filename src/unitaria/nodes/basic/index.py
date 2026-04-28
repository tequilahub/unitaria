import numpy as np

from unitaria.subspace import Subspace
from unitaria.nodes.node import Node
from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.basic.adjoint import Adjoint
from unitaria.nodes.basic.projection import Projection
from unitaria.nodes.basic.mul import Mul
from unitaria.nodes.classical.constant_integer_addition import ConstantIntegerAddition


class Index(ProxyNode):
    """
    Used in the implementation of indexing of nodes.

    Specifically, ``A[x]`` is equivalent to ``Index(A.dimension_out, x) @ A``.
    """

    def __init__(self, dim: int, index: slice | int):
        if isinstance(index, (int, np.integer)):
            index = slice(int(index), int(index) + 1, None)

        index = [index.start, index.stop, index.step]

        if index[0] is None:
            index[0] = 0
        if index[1] is None:
            index[1] = dim
        if index[2] is None:
            index[2] = 1

        if index[0] < 0:
            index[0] = dim + index[0]
        if index[1] < 0:
            index[1] = dim + index[1]

        if not (
            index[0] >= 0
            and index[0] < dim
            and index[1] > 0
            and index[1] <= dim
            and index[0] < index[1]
            and index[2] > 0
        ):
            raise IndexError

        self.dim = dim
        self.index = slice(index[0], index[1], index[2])
        len = (self.index.stop - self.index.start + self.index.step - 1) // self.index.step

        super().__init__(dim, len)

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        params = {}
        params["dim"] = self.dim
        params["index"] = self.index
        return params

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        input = input.reshape([-1, self.dim])
        input = input[:, self.index.start : self.index.stop : self.index.step]
        input = input.reshape(outer_shape + [-1])
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        input = input.reshape([-1, input.shape[-1]])
        output = np.zeros([input.shape[0], self.dim], dtype=complex)
        output[:, self.index.start : self.index.stop : self.index.step] = input
        output = output.reshape(outer_shape + [self.dim])
        return output

    def definition(self):
        len = (self.index.stop - self.index.start) // self.index.step
        last_step = self.index.stop - self.index.start - len * self.index.step > 0
        remainder = self.dim - len * self.index.step - self.index.start
        if last_step:
            remainder -= 1
        subspace_len = Subspace.from_dim(len)
        subspace_step = Subspace.from_dim(self.index.step)

        subspace_in = subspace_len & subspace_step
        subspace_out = subspace_len & Subspace.from_dim(1, bits=subspace_step.total_qubits)
        assert subspace_in.total_qubits == subspace_out.total_qubits
        if last_step:
            subspace_last_step = Subspace("0" * subspace_in.total_qubits)
            subspace_in = subspace_in | subspace_last_step
            subspace_out = subspace_out | subspace_last_step
            assert subspace_in.total_qubits == subspace_out.total_qubits
        if remainder > 0:
            remainder_bits = int(np.ceil(np.log2(remainder)))
            if remainder_bits > subspace_in.total_qubits:
                add_bits = remainder_bits - subspace_in.total_qubits
                subspace_in = Subspace("0" * add_bits) & subspace_in
                subspace_out = Subspace("0" * add_bits) & subspace_out
            subspace_in = subspace_in | Subspace.from_dim(remainder, bits=subspace_in.total_qubits)
            subspace_out = Subspace("0") & subspace_out
            assert subspace_in.total_qubits == subspace_out.total_qubits

        projection = Projection(subspace_in, subspace_out)
        if self.index.start == 0:
            return projection

        # Handle offset
        # TODO: If Subspace ever supports 1 qubits, this could be simplified
        # to not use ConstantIntegerAddition

        subspace_dim = Subspace.from_dim(self.dim)
        bits = subspace_dim.total_qubits
        subspace_bits = Subspace("#" * bits)
        shift = Adjoint(ConstantIntegerAddition(bits, self.index.start))
        subspace_shift_out = Subspace.from_dim(subspace_in.dimension, bits=bits)
        shift = Mul(Mul(Projection(subspace_bits, subspace_shift_out), shift), Projection(subspace_dim, subspace_bits))
        return projection @ shift


def _node_getitem(self: Node, indices):
    if isinstance(indices, tuple):
        return Mul(
            Mul(Index(self.dimension_out, indices[0]), self),
            Adjoint(Index(self.dimension_in, indices[1])),
        )
    else:
        return Mul(Index(self.dimension_out, indices), self)


Node.__getitem__ = _node_getitem
