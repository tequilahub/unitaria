import numpy as np

from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.node import Node
from bequem.nodes.basic_ops import Tensor, UnsafeMul, ComputeProjection
from bequem.nodes.permutation import Permutation
from bequem.nodes.identity import Identity
from bequem.qubit_map import QubitMap

class Mul(ProxyNode):
    """
    Node for computing the product of two nodes

    The order of operations is such that the first argument ``A`` is applied
    first, i.e. this implements ``B @ A``.

    :ivar A:
        The first factor
    :ivar B:
        The second factor
    """
    A: Node
    B: Node
    skip_projection_check: bool

    def __init__(self, A: Node, B: Node, skip_projection_check: bool = False):
        if A.qubits_out().dimension != B.qubits_in().dimension:
            raise ValueError
        self.A = A
        self.B = B
        self.skip_projection_check = skip_projection_check

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def parameters(self) -> dict:
        params = {}
        if self.skip_projection_check:
            params["skip_projection_check"] = True
        return params

    def definition(self) -> Node:
        permutation = Permutation(self.A.qubits_out(), self.B.qubits_in())
        if self.skip_projection_check:
            return UnsafeMul(UnsafeMul(self.A, permutation), self.B)
        else:
            A = Tensor(self.A,
                Identity(QubitMap(0, 1)))
            B = Tensor(UnsafeMul(permutation, self.B),
                Identity(QubitMap(0, 1)))
            return UnsafeMul(
                UnsafeMul(A, ComputeProjection(self.A.qubits_out())),
                B)

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        input = self.A.compute(input)
        input = self.B.compute(input)
        return input

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        input = self.B.compute_adjoint(input)
        input = self.A.compute_adjoint(input)
        return input

    def normalization(self) -> float:
        return self.A.normalization() * self.B.normalization()

    def phase(self) -> float:
        return self.A.phase() + self.B.phase()

    def controlled(self):
        return Mul(self.A.controlled(), self.B.controlled(), self.skip_projection_check)


Node.__matmul__ = lambda A, B: Mul(A, B)

