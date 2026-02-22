import numpy as np
from unitaria.nodes.node import Node

from unitaria.nodes.basic.unsafe_multiplication import UnsafeMul
from unitaria.nodes.basic.tensor import Tensor
from unitaria.nodes.basic.compute_projection import SubspaceCircuit

from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.permutation.permutation import Permutation
from unitaria.nodes.basic.identity import Identity
from unitaria.subspace import Subspace


class Mul(ProxyNode):
    """
    Node for computing the product of two nodes

    The order of operations is such that the first argument ``A`` is applied
    first, i.e. this implements ``B @ A``. This requires ``A.dimension_out ==
    B.dimension_in``.

    :param A:
        The first factor
    :param B:
        The second factor
    """

    A: Node
    B: Node
    skip_projection: bool

    def __init__(self, A: Node, B: Node, skip_projection: bool = False):
        if A.dimension_out != B.dimension_in:
            raise ValueError(f"dimensions {A.dimension_out} and {B.dimension_in} do not match")
        super().__init__(A.dimension_in, B.dimension_out)
        self.A = A
        self.B = B
        self.skip_projection = skip_projection

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def parameters(self):
        params = {}
        if self.skip_projection:
            params["skip_projection"] = True
        return params

    def definition(self) -> Node:
        permutation = Permutation(self.A.subspace_out, self.B.subspace_in)
        A_permuted = self.A
        B_permuted = UnsafeMul(permutation, self.B)
        projection_subspace = A_permuted.subspace_out
        projection_required = not self.skip_projection
        if projection_subspace == Subspace(projection_subspace.total_qubits):
            projection_required = False
        if projection_required:
            # TODO: This can be probably be done more correctly
            # once match_nonzero is improved
            if A_permuted.subspace_out.total_qubits < B_permuted.subspace_in.total_qubits:
                projection_subspace = B_permuted.subspace_in
            A_permuted = Tensor(A_permuted, Identity(subspace=Subspace(registers=0, zero_qubits=1)))
            B_permuted = Tensor(B_permuted, Identity(subspace=Subspace(registers=0, zero_qubits=1)))
            return UnsafeMul(UnsafeMul(A_permuted, SubspaceCircuit(projection_subspace)), B_permuted)
        else:
            return UnsafeMul(A_permuted, B_permuted)

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        input = self.A.compute(input)
        input = self.B.compute(input)
        return input

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        input = self.B.compute_adjoint(input)
        input = self.A.compute_adjoint(input)
        return input

    def _normalization(self) -> float:
        return self.A.normalization * self.B.normalization


Node.__matmul__ = lambda A, B: Mul(B, A)
