import numpy as np
from bequem.nodes.node import Node
from bequem.nodes.basic_ops import Scale, Adjoint, UnsafeMul, Tensor, ComputeProjection
from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.controlled_ops import BlockDiagonal
from bequem.nodes.permutation import SimplifyZeros, find_permutation
from bequem.nodes.constant import ConstantVector
from bequem.nodes.identity import Identity
from bequem.qubit_map import QubitMap


class Add(ProxyNode):
    """
    Node for computing the sum of two nodes

    :ivar A:
        The first summand
    :ivar B:
        The second summand
    """
    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self) -> Node:
        permutation_in = find_permutation(self.A.qubits_in(),
                                          self.B.qubits_in())
        permutation_out = find_permutation(self.A.qubits_out(),
                                           self.B.qubits_out())

        A_permuted = Scale(UnsafeMul(
            Adjoint(permutation_in.permute_a),
            UnsafeMul(self.A, permutation_out.permute_a)),
                           absolute=True)
        B_permuted = Scale(UnsafeMul(
            Adjoint(permutation_in.permute_b),
            UnsafeMul(self.B, permutation_out.permute_b)),
                           absolute=True)

        diag = BlockDiagonal(A_permuted, B_permuted)
        simplify_in = SimplifyZeros(diag.qubits_in())
        simplify_out = SimplifyZeros(diag.qubits_out())
        diag = UnsafeMul(Adjoint(simplify_in), UnsafeMul(diag, simplify_out))

        sqrt_A = np.sqrt(np.abs(self.A.normalization()))
        sqrt_B = np.sqrt(np.abs(self.B.normalization()))
        rotation_in = Tensor(Identity(permutation_in.target()),
                             ConstantVector(np.array([sqrt_A, sqrt_B])))
        rotation_out = Tensor(
            Identity(permutation_out.target()),
            ConstantVector(
                np.array([
                    self.A.normalization() / sqrt_A,
                    self.B.normalization() / sqrt_B
                ])))

        return UnsafeMul(UnsafeMul(rotation_in, diag), Adjoint(rotation_out))

    def normalization(self) -> float:
        return self.A.normalization() + self.B.normalization()

    def compute(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute(input) + self.B.compute(input)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute_adjoint(input) + self.B.compute_adjoint(input)


Node.__add__ = lambda A, B: Add(A, B)


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

    def __init__(self, A: Node, B: Node):
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self) -> Node:
        permutation = find_permutation(self.A.qubits_out(), self.B.qubits_in())
        A_permuted = Tensor(UnsafeMul(self.A, permutation.permute_a),
                            Identity(QubitMap(0, 1)))
        B_permuted = Tensor(UnsafeMul(Adjoint(permutation.permute_b), self.B),
                            Identity(QubitMap(0, 1)))
        return UnsafeMul(
            A_permuted,
            UnsafeMul(ComputeProjection(permutation.target()), B_permuted))

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


Node.__matmul__ = lambda A, B: Mul(A, B)
