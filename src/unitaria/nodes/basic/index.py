import numpy as np

from unitaria.subspace import Subspace
from unitaria.nodes.node import Node
from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.basic.adjoint import Adjoint
from unitaria.nodes.basic.projection import Projection
from unitaria.nodes.basic.mul import Mul
from unitaria.nodes.classical.constant_integer_addition import ConstantIntegerAddition
from unitaria.util import logreduce


class Index(ProxyNode):
    """
    Used in the implementation of indexing of nodes.

    Specifically, ``A[x]`` is equivalent to ``Index(A.dimension_out, x) @ A``.
    """

    def __init__(self, A: Node, index_in: slice | int, index_out: slice | int):
        self.A = A
        self.index_in = Index._preprocess_index(A.dimension_in, index_in)
        self.index_out = Index._preprocess_index(A.dimension_out, index_out)

        dimension_in = (self.index_in.stop - self.index_in.start + self.index_in.step - 1) // self.index_in.step

        dimension_out = (self.index_out.stop - self.index_out.start + self.index_out.step - 1) // self.index_out.step

        super().__init__(dimension_in, dimension_out)

    def _preprocess_index(dimension: int, index: slice | int) -> slice:
        if isinstance(index, (int, np.integer)):
            index = slice(int(index), int(index) + 1, None)

        index = [index.start, index.stop, index.step]
        if index[0] is None:
            index[0] = 0
        if index[1] is None:
            index[1] = dimension
        if index[2] is None:
            index[2] = 1

        if index[0] < 0:
            index[0] = dimension + index[0]
        if index[1] < 0:
            index[1] = dimension + index[1]

        if not (
            index[0] >= 0
            and index[0] < dimension
            and index[1] > 0
            and index[1] <= dimension
            and index[0] < index[1]
            and index[2] > 0
        ):
            raise IndexError
        return slice(index[0], index[1], index[2])

    def children(self) -> list[Node]:
        return [self.A]

    def parameters(self) -> dict:
        params = {}
        params["index_in"] = self.index_in
        params["index_out"] = self.index_out
        return params

    def _normalization(self) -> float:
        return self.A.normalization

    def _compute_index(input: np.array, index: slice) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        input = input.reshape([-1, input.shape[-1]])
        input = input[:, index.start : index.stop : index.step]
        input = input.reshape(outer_shape + [-1])
        return input

    def _compute_index_adjoint(input: np.array, index: slice, dimension: int) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        input = input.reshape([-1, input.shape[-1]])
        output = np.zeros([input.shape[0], dimension], dtype=complex)
        output[:, index.start : index.stop : index.step] = input
        output = output.reshape(outer_shape + [dimension])
        return output

    def compute(self, input: np.ndarray) -> np.ndarray:
        input = Index._compute_index_adjoint(input, self.index_in, self.A.dimension_in)
        input = self.A.compute(input)
        input = Index._compute_index(input, self.index_out)
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        input = Index._compute_index_adjoint(input, self.index_out, self.A.dimension_out)
        input = self.A.compute_adjoint(input)
        input = Index._compute_index(input, self.index_in)
        return input

    def definition(self) -> Node:
        result = []

        projection_in = Index._projection(self.A.subspace_in, self.index_in)
        if projection_in is not None:
            result.append(Adjoint(projection_in))

        result.append(self.A)

        projection_out = Index._projection(self.A.subspace_out, self.index_out)
        if projection_out is not None:
            result.append(projection_out)

        return logreduce(Mul, result[::-1])

    def _projection(subspace: Subspace, index: slice) -> Node | None:
        result = []

        # 1. Handle stop
        if index.stop < subspace.dimension:
            result.append(Projection(subspace, subspace.truncate(index.stop)))

        # 2. Handle start
        if index.start > 0:
            subspace_add_in = Subspace.from_dim(index.stop)
            bits = subspace_add_in.total_qubits
            subspace_add = Subspace("#" * bits)
            subspace_add_out = Subspace.from_dim(index.stop - index.start, bits=subspace_add_in.total_qubits)
            result.append(Projection(subspace_add_in, subspace_add))
            result.append(Adjoint(ConstantIntegerAddition(bits, index.start)))
            result.append(Projection(subspace_add, subspace_add_out))

        # 3. Handle step
        if index.step != 1:
            subspace_step = Subspace.from_dim(index.step)
            dimension_out = (index.stop - index.start + index.step - 1) // index.step
            subspace_stride_in = (Subspace.from_dim(dimension_out) & subspace_step).truncate(index.stop - index.start)
            subspace_stride_out = Subspace.from_dim(dimension_out) & Subspace("0" * subspace_step.total_qubits)
            result.append(Projection(subspace_stride_in, subspace_stride_out))

        if len(result) == 0:
            return None

        return logreduce(Mul, result[::-1])


def _node_getitem(self: Node, indices):
    if not isinstance(indices, tuple):
        indices = (indices, slice(None, None, None))
    return Index(self, indices[1], indices[0])


Node.__getitem__ = _node_getitem
