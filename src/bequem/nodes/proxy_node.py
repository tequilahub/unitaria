from abc import abstractmethod

import numpy as np
from rich.panel import Panel

from bequem.nodes.node import Node
from bequem.nodes.basic_ops import UnsafeMul, Adjoint, Tensor, Scale
from bequem.nodes.controlled_ops import BlockDiagonal
from bequem.nodes.constant import ConstantVector
from bequem.nodes.identity import Identity
from bequem.circuit import Circuit
from bequem.qubit_map import QubitMap
from bequem.nodes.permutation import find_permutation, SimplifyZeros

class ProxyNode(Node):
    """
    Abstract class for nodes that are defined in terms of other nodes

    This is useful when the given definition is used in the construction of
    the circuit or QubitMaps, but a simpler representation is avilable for
    :py:func:`compute`. Additionally, the simpler structure is used for printing
    and serialization.

    For an example of how to use this class see :py:class:`Add` and
    :py:class:`Mul`.
    """

    _definition: Node | None = None

    @abstractmethod
    def definition(self) -> Node:
        """
        The definition of this node, which is used to implement all other
        abstract methods of :py:class:`Node`. The other methods can be
        overwritten to give a more efficient implementation.
        """
        raise NotImplementedError

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        # TODO: Use something more concise for lazy calculation, instead of manually
        #  checking for None everywhere
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.compute(input)

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.compute_adjoint(input)

    def circuit(self) -> Circuit:
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.circuit()

    def qubits_in(self) -> QubitMap:
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.qubits_in()

    def qubits_out(self) -> QubitMap:
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.qubits_out()

    def normalization(self) -> float:
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.normalization()

    def tree_label(self, verbose: bool = False):
        label = super().tree_label()
        if not verbose:
            return label
        else:
            if self._definition is None:
                self._definition = self.definition()
            return Panel(self._definition.tree(verbose=True, holes=self.children()), title=label, title_align="left")


class ProjectionNode(Node):
    def __init__(self, qubits: QubitMap):
        self.qubits = QubitMap(qubits.registers, qubits.zero_qubits + 1)

    def children(self) -> list[Node]:
        return []

    def qubits_in(self) -> QubitMap:
        return self.qubits

    def qubits_out(self) -> QubitMap:
        return self.qubits

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return input

    def circuit(self) -> Circuit:
        # TODO
        return Circuit()


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
        permutation = find_permutation(
            self.A.qubits_out(), self.B.qubits_in()
        )
        A_permuted = Tensor(UnsafeMul(self.A, permutation.permute_a), Identity(QubitMap(0, 1)))
        B_permuted = Tensor(UnsafeMul(Adjoint(permutation.permute_b), self.B), Identity(QubitMap(0, 1)))
        return UnsafeMul(A_permuted, UnsafeMul(ProjectionNode(permutation.target()), B_permuted))

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


Node.__matmul__ = lambda A, B: Mul(A, B)


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
        permutation_in = find_permutation(
            self.A.qubits_in(), self.B.qubits_in()
        )
        permutation_out = find_permutation(
            self.A.qubits_out(), self.B.qubits_out()
        )

        A_permuted = Scale(UnsafeMul(Adjoint(permutation_in.permute_a), UnsafeMul(self.A, permutation_out.permute_a)), absolute=True)
        B_permuted = Scale(UnsafeMul(Adjoint(permutation_in.permute_b), UnsafeMul(self.B, permutation_out.permute_b)), absolute=True)

        diag = BlockDiagonal(A_permuted, B_permuted)
        simplify_in = SimplifyZeros(diag.qubits_in())
        simplify_out = SimplifyZeros(diag.qubits_out())
        diag = UnsafeMul(Adjoint(simplify_in), UnsafeMul(diag, simplify_out))

        sqrt_A = np.sqrt(np.abs(self.A.normalization()))
        sqrt_B = np.sqrt(np.abs(self.B.normalization()))
        rotation_in = Tensor(Identity(permutation_in.target()), ConstantVector(np.array([sqrt_A, sqrt_B])))
        rotation_out = Tensor(Identity(permutation_out.target()), ConstantVector(np.array([self.A.normalization() / sqrt_A, self.B.normalization() / sqrt_B])))

        return UnsafeMul(UnsafeMul(rotation_in, diag), Adjoint(rotation_out))

    def normalization(self) -> float:
        return self.A.normalization() + self.B.normalization()

    def compute(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute(input) + self.B.compute(input)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute_adjoint(input) + self.B.compute_adjoint(input)


Node.__add__ = lambda A, B: Add(A, B)


class BlockHorizontal(ProxyNode):
    """
    Node for block matrices of the form ``[A B]``

    :ivar A:
        The left block
    :ivar B:
        The right block
    """
    A: Node
    B: Node

    def __init__(self, A: Node, B: Node):
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self) -> Node:
        permutation = find_permutation(
            self.A.qubits_out(), self.B.qubits_out()
        )

        A_permuted = Scale(UnsafeMul(self.A, permutation.permute_a), absolute=True)
        B_permuted = Scale(UnsafeMul(self.B, permutation.permute_b), absolute=True)

        diag = BlockDiagonal(A_permuted, B_permuted)
        simplify = SimplifyZeros(diag.qubits_out())
        diag = UnsafeMul(diag, simplify)

        rotation_out = Tensor(Identity(permutation.target()), ConstantVector(np.array([self.A.normalization(), self.B.normalization()])))

        return UnsafeMul(diag, Adjoint(rotation_out))


    def normalization(self) -> float:
        return np.sqrt(np.abs(self.A.normalization()) ** 2 + np.abs(self.B.normalization()) ** 2)

    def compute(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_in().dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        return self.A.compute(input_A) + self.B.compute(input_B)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return np.concatenate((self.A.compute_adjoint(input), self.B.compute(input)), axis=-1)


class BlockVertical(ProxyNode):
    """
    Node for block matrices of the form ``[A B]^T``

    :ivar A:
        The top block
    :ivar B:
        The bottom block
    """

    def __init__(self, A: Node, B: Node):
        self.A = A
        self.B = B

    def children(self) -> list[Node]:
        return [self.A, self.B]

    def definition(self) -> Node:
        permutation = find_permutation(
            self.A.qubits_in(), self.B.qubits_in()
        )

        A_permuted = Scale(UnsafeMul(Adjoint(permutation.permute_a), self.A), absolute=True)
        B_permuted = Scale(UnsafeMul(Adjoint(permutation.permute_b), self.B), absolute=True)

        diag = BlockDiagonal(A_permuted, B_permuted)
        simplify = SimplifyZeros(diag.qubits_in())
        diag = UnsafeMul(Adjoint(simplify), diag)

        rotation_in = Tensor(Identity(permutation.target()), ConstantVector(np.array([self.A.normalization(), self.B.normalization()])))

        return UnsafeMul(rotation_in, diag)


    def normalization(self) -> float:
        return np.sqrt(np.abs(self.A.normalization()) ** 2 + np.abs(self.B.normalization()) ** 2)

    def compute(self, input: np.ndarray) -> np.ndarray:
        return np.concatenate((self.A.compute(input), self.B.compute(input)), axis=-1)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        dim_A = self.A.qubits_in().dimension
        input_A, input_B = np.split(input, [dim_A], axis=-1)
        return self.A.compute_adjoint(input_A) + self.B.compute_adjoint(input_B)
