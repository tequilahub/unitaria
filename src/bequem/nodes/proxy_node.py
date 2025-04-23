from abc import abstractmethod

import numpy as np

from bequem.nodes.node import Node
from bequem.nodes.basic_ops import UnsafeMul, Adjoint, Tensor, Scale
from bequem.nodes.controlled_ops import BlockDiagonal
from bequem.nodes.constant import ConstantVector
from bequem.nodes.identity import Identity
from bequem.circuit import Circuit
from bequem.qubit_map import QubitMap
from bequem.nodes.permutation import find_permutation, SimplifyZeros

class ProxyNode(Node):

    _definition: Node | None = None

    @abstractmethod
    def definition(self) -> Node:
        raise NotImplementedError

    def compute(self, input: np.ndarray | None) -> np.ndarray:
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


class ProjectionNode(Node):
    def __init__(self, qubits: QubitMap):
        self.qubits = QubitMap(qubits.registers, qubits.zero_qubits + 1)

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
    def __init__(self, A: Node, B: Node):
        self.A = A
        self.B = B

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
    def __init__(self, A: Node, B: Node):
        self.A = A
        self.B = B

    def definition(self) -> Node:
        permutation_in = find_permutation(
            self.A.qubits_in(), self.B.qubits_in()
        )
        permutation_out = find_permutation(
            self.A.qubits_out(), self.B.qubits_out()
        )

        A_permuted = Scale(UnsafeMul(Adjoint(permutation_in.permute_a), UnsafeMul(self.A, permutation_out.permute_a)), scale_absolute=True)
        B_permuted = Scale(UnsafeMul(Adjoint(permutation_in.permute_b), UnsafeMul(self.B, permutation_out.permute_b)), scale_absolute=True)

        diag = BlockDiagonal(A_permuted, B_permuted)
        simplify_in = SimplifyZeros(diag.qubits_in())
        simplify_out = SimplifyZeros(diag.qubits_out())
        diag = UnsafeMul(Adjoint(simplify_in), UnsafeMul(diag, simplify_out))

        sqrt_A = np.sqrt(np.abs(self.A.normalization()))
        sqrt_B = np.sqrt(np.abs(self.B.normalization()))
        # print(np.array([sqrt_A, sqrt_B]))
        # print(np.conj(np.array([self.A.normalization() / sqrt_A, self.B.normalization() / sqrt_B])))
        rotation_in = Tensor(Identity(permutation_in.target()), ConstantVector(np.array([sqrt_A, sqrt_B])))
        rotation_out = Tensor(Identity(permutation_out.target()), ConstantVector(np.conj(np.array([self.A.normalization() / sqrt_A, self.B.normalization() / sqrt_B]))))

        return UnsafeMul(UnsafeMul(rotation_in, diag), Adjoint(rotation_out))


    def normalization(self) -> float:
        return self.A.normalization() + self.B.normalization()

    def compute(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute(input) + self.B.compute(input)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return self.A.compute_adjoint(input) + self.B.compute_adjoint(input)


Node.__add__ = lambda A, B: Add(A, B)


