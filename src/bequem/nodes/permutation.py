from __future__ import annotations
from dataclasses import dataclass

import tequila as tq
import numpy as np

from bequem.nodes.node import Node
from bequem.nodes.identity import Identity
from bequem.nodes.basic_ops import Tensor, UnsafeMul, Adjoint
from bequem.nodes.controlled_ops import BlockDiagonal
from bequem.qubit_map import QubitMap, Qubit
from bequem.circuit import Circuit


def find_permutation(a: QubitMap,
                     b: QubitMap) -> Permutation:
    if a.dimension != b.dimension:
        raise ValueError()

    perm_a = Identity(QubitMap([]))
    perm_b = Identity(QubitMap([]))

    for (sub_a, sub_b) in _find_matching_subdivision(a, b):
        if sub_a == sub_b:
            sub_permutation = Permutation(Identity(sub_a), Identity(sub_b))
        else:
            sub_permutation = _find_permutation_brute_force(sub_a, sub_b)
        perm_a = Tensor(perm_a, sub_permutation.permute_a)
        perm_b = Tensor(perm_b, sub_permutation.permute_b)

    assert a.registers == perm_a.qubits_in().registers
    assert b.registers == perm_b.qubits_in().registers

    return Permutation(perm_a, perm_b)


def _find_matching_subdivision(a: QubitMap,
                               b: QubitMap) -> list[tuple[QubitMap, QubitMap]]:
    if a.registers == [] and b.registers == []:
        return []

    assert a.dimension == b.dimension
    subdivisions = []
    last_breakpoint_a = 0
    last_breakpoint_b = 0
    i_a = 1
    i_b = 1
    submap_a = QubitMap(a.registers[last_breakpoint_a:i_a])
    submap_b = QubitMap(b.registers[last_breakpoint_b:i_b])
    while i_a < len(a.registers) and i_b < len(b.registers):
        if submap_a.dimension == submap_b.dimension:
            subdivisions.append((submap_a, submap_b))
            last_breakpoint_a = i_a
            last_breakpoint_b = i_b
            i_a += 1
            i_b += 1
            submap_a = QubitMap(a.registers[last_breakpoint_a:i_a])
            submap_b = QubitMap(b.registers[last_breakpoint_b:i_b])
        elif submap_a.dimension < submap_b.dimension:
            i_a += 1
            submap_a = QubitMap(a.registers[last_breakpoint_a:i_a])
        else:
            i_b += 1
            submap_b = QubitMap(b.registers[last_breakpoint_b:i_b])

    submap_a = QubitMap(a.registers[last_breakpoint_a:])
    submap_b = QubitMap(b.registers[last_breakpoint_b:])
    subdivisions.append((submap_a, submap_b))

    return subdivisions


class SimplifyZeros(Node):

    def __init__(self, qubits: QubitMap):
        assert len(qubits.registers) > 0 and isinstance(qubits.registers[-1], Qubit)

        case_zero = qubits.registers[-1].case_zero
        case_one = qubits.registers[-1].case_one
        self.common_zeros = min(case_zero.zero_qubits, case_one.zero_qubits)
        self.qubits = qubits

        case_zero = QubitMap(case_zero.registers,
                             case_zero.zero_qubits - self.common_zeros)
        case_one = QubitMap(case_one.registers,
                             case_one.zero_qubits - self.common_zeros)
        self._qubits_out = QubitMap(
            self.qubits.registers[:-1] + [Qubit(case_zero, case_one)],
            self.qubits.zero_qubits + self.common_zeros)

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"qubits": self.qubits}

    def qubits_in(self) -> QubitMap:
        return self.qubits

    def qubits_out(self) -> QubitMap:
        return self._qubits_out

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        return input

    def circuit(self) -> Circuit:
        circuit = Circuit()

        if self.common_zeros > 0:
            circuit += Circuit(
                tq.gates.SWAP(
                    self.qubits.total_qubits - self.qubits.zero_qubits - 1,
                    self.qubits.total_qubits - self.qubits.zero_qubits - self.common_zeros -
                    1))

        circuit.tq_circuit.n_qubits = self.qubits.total_qubits
        
        return circuit

class QubitMapRotation(Node):

    def __init__(self, qubits: QubitMap, right: bool):
        assert len(qubits.registers) > 0 and isinstance(qubits.registers[-1], Qubit)

        if right:
            pivot = _left_child(qubits)
            l = _extend_zeros(_left_child(pivot), 2)
            m = _extend_zeros(_right_child(pivot), 1)
            r = _right_child(qubits)
            self._qubits_out = QubitMap([Qubit(l, QubitMap([Qubit(m, r)]))])
        else:
            pivot = _right_child(qubits)
            l = _left_child(qubits)
            m = _extend_zeros(_left_child(pivot), 1)
            r = _extend_zeros(_right_child(pivot), 2)
            self._qubits_out = QubitMap([Qubit(QubitMap([Qubit(l, m)]), r)])

        self.qubits = qubits
        self.right = right

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"qubits": self.qubits, "right": self.right}

    def qubits_in(self) -> QubitMap:
        return QubitMap(self.qubits.registers, self.qubits.zero_qubits + 1)

    def qubits_out(self) -> QubitMap:
        return self._qubits_out

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        return input

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        return input

    def circuit(self) -> Circuit:
        circuit = Circuit()

        last_bit = self._qubits_out.total_qubits - 1
        if self.right:
            circuit.tq_circuit += tq.gates.CNOT(last_bit - 1, last_bit)
            circuit.tq_circuit += tq.gates.X(last_bit - 1)
            circuit.tq_circuit += tq.gates.CNOT(
                [last_bit - 1, last_bit - 2],
                last_bit,
            )
            circuit.tq_circuit += tq.gates.CNOT(
                [last_bit - 1, last_bit],
                last_bit - 2,
            )
            circuit.tq_circuit += tq.gates.X(last_bit - 1)
        else:
            circuit.tq_circuit += tq.gates.CNOT(
                [last_bit - 1, last_bit - 2],
                last_bit,
            )
            circuit.tq_circuit += tq.gates.CNOT(
                last_bit, [last_bit-1, last_bit-2]
            )

        circuit.tq_circuit.n_qubits = self._qubits_out.total_qubits

        return circuit


def _find_permutation_brute_force(
        a: QubitMap, b: QubitMap) -> Permutation:
    assert a.dimension == b.dimension
    assert len(a.registers) != 0
    assert len(b.registers) != 0

    # TODO: Potentially switch a and b
    perm_b = SimplifyZeros(b)

    while _split_dimension(a) != _split_dimension(perm_b.qubits_out()):
        if _split_dimension(a) < _split_dimension(perm_b.qubits_out()):
            pivot = _left_child(perm_b.qubits_out())
            if _split_dimension(a) <= _split_dimension(pivot):
                # Right rotation
                rot = QubitMapRotation(perm_b.qubits_out(), True)
                simp = SimplifyZeros(rot.qubits_out())
                perm_b = UnsafeMul(perm_b, UnsafeMul(rot, simp))
            else:
                # Left-right rotation
                pivot2 = _right_child(pivot)
                raise NotImplementedError
        else:
            pivot = _right_child(perm_b.qubits_out())
            if _split_dimension(a) >= _split_dimension(pivot):
                # Left rotation
                rot = QubitMapRotation(perm_b.qubits_out(), False)
                simp = SimplifyZeros(rot.qubits_out())
                perm_b = UnsafeMul(perm_b, UnsafeMul(rot, simp))
            else:
                # Right-left rotation
                pivot2 = _left_child(pivot)
                raise NotImplementedError

    b = perm_b.qubits_out()

    a_zero = _left_child(a)
    b_zero = _left_child(b)
    a_one = _right_child(a)
    b_one = _right_child(b)

    perm_zero = find_permutation(a_zero, b_zero)

    perm_one = find_permutation(a_one, b_one)

    diag_a = BlockDiagonal(perm_zero.permute_a, perm_one.permute_a)
    simp_in = SimplifyZeros(diag_a.qubits_in())
    simp_out = SimplifyZeros(diag_a.qubits_out())
    perm_a = UnsafeMul(Adjoint(simp_in), UnsafeMul(diag_a, simp_out))

    diag_b = BlockDiagonal(perm_zero.permute_b, perm_one.permute_b)

    simp_in = SimplifyZeros(diag_b.qubits_in())
    simp_out = SimplifyZeros(diag_b.qubits_out())
    diag_b = UnsafeMul(Adjoint(simp_in), UnsafeMul(diag_b, simp_out))
    perm_b = UnsafeMul(perm_b, diag_b)

    return Permutation(perm_a, perm_b)


def _split_dimension(qubits: QubitMap) -> int:
    assert len(qubits.registers) != 0
    assert isinstance(qubits.registers[-1], Qubit)

    total_dim = qubits.dimension
    zero_dim = qubits.registers[-1].case_zero.dimension
    one_dim = qubits.registers[-1].case_one.dimension

    return total_dim / (zero_dim + one_dim) * zero_dim


def _left_child(qubits: QubitMap) -> QubitMap:
    assert len(qubits.registers) != 0
    assert isinstance(qubits.registers[-1], Qubit)

    return QubitMap(
        qubits.registers[:-1] + qubits.registers[-1].case_zero.registers,
        qubits.zero_qubits + qubits.registers[-1].case_zero.zero_qubits)


def _right_child(qubits: QubitMap) -> QubitMap:
    assert len(qubits.registers) != 0
    assert isinstance(qubits.registers[-1], Qubit)

    return QubitMap(
        qubits.registers[:-1] + qubits.registers[-1].case_one.registers,
        qubits.zero_qubits + qubits.registers[-1].case_one.zero_qubits)


def _extend_zeros(qubits: QubitMap, n: int) -> QubitMap:
    return QubitMap(qubits.registers[:], qubits.zero_qubits + n)


@dataclass(init=False)
class Permutation:
    permute_a: Node
    permute_b: Node

    def __init__(self, permute_a: Node, permute_b: Node):
        if permute_a.qubits_out().registers != permute_b.qubits_out().registers:
            raise ValueError
        self.permute_a = permute_a
        self.permute_b = permute_b

    def target(self) -> QubitMap:
        return self.permute_a.qubits_out()

    def verify(self):
        assert self.permute_a.qubits_out().registers == self.permute_b.qubits_out().registers
        assert self.permute_a.normalization() == 1
        A = self.permute_a.verify_recursive()
        if len(A.shape) == 1:
            A = np.array([A])
        np.testing.assert_allclose(A, np.eye(A.shape[0]))
        assert self.permute_b.normalization() == 1
        B = self.permute_b.verify_recursive()
        if len(B.shape) == 1:
            B = np.array([B])
        np.testing.assert_allclose(B, np.eye(B.shape[0]))
