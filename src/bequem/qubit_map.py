from __future__ import annotations
from typing import ClassVar
from dataclasses import dataclass
from enum import Enum

import numpy as np

from bequem.circuit import Circuit


class QubitMap:
    def __init__(self, registers: list[Register]):
        for register in registers:
            if not isinstance(register, Register):
                raise ValueError(f"{register} is not valid in a QubitMap")
        self.registers = registers
        self._dimension = None
        self._total_qubits = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = 1
            for register in self.registers:
                match register:
                    case Qubit(_):
                        pass
                    case Controlled(case_zero, case_one):
                        self._dimension *= case_zero.dimension + case_one.dimension
                    case _:
                        raise NotImplementedError

        return self._dimension

    @property
    def total_qubits(self) -> int:
        if self._total_qubits is None:
            self._total_qubits = 0
            for register in self.registers:
                match register:
                    case Controlled(case_zero):
                        self._total_qubits += 1 + case_zero.total_qubits
                    case Projection(circuit):
                        # One bit in the circuit corresponds to the result of the
                        # operation
                        self._total_qubits += len(circuit.tq_circuit.qubits) - 1
                    case Qubit():
                        self._total_qubits += 1
                    case _:
                        raise NotImplementedError
        return self._total_qubits

    def __eq__(self, other) -> bool:
        return self.registers == other.registers

    def simplify(self) -> QubitMap:
        simplified = []

        for register in self.registers:
            match register:
                case Controlled(case_zero, case_one):
                    case_zero = register.case_zero.simplify()
                    case_one = register.case_one.simplify()
                    simplified.extend(Controlled(case_zero, case_one).simplify())
                case Qubit() | Projection():
                    simplified.append(register)
                case _:
                    raise NotImplementedError

        return QubitMap(simplified)

    def cast_to(self, other: QubitMap) -> CastResult:
        if self.dimension != other.dimension:
            # TODO
            raise CastException()
        raise NotImplementedError

    def _find_partial_permutation(self, other: QubitMap) -> Permutation:
        assert self.dimension == other.dimension

        if len(self.registers) == 0:
            return dict()

        head = QubitMap(self.registers[:-1])

        match self.registers[-1]:
            case Qubit(_):
                return head._find_partial_permutation(other)
            case Controlled(case_zero, case_one):
                s = other._split(case_zero.dimension)
                match s:
                    case QubitMap(_), QubitMap(_):
                        raise NotImplementedError
                    case SimplifiedSplit(_):
                        if head.dimension == s.case_zero.dimension:
                            perm0 = case_zero._find_partial_permutation(s.case_zero)
                            perm1 = case_one._find_partial_permutation(s.case_one)
                            if perm0 == perm1:
                                perm_head = head._find_partial_permutation(
                                    s.common_head
                                )
                                perm_head[self.total_qubits - 1] = (
                                    s.common_head.total_qubits
                                    + s.case_zero.total_qubits
                                )
                                return perm_head
                        # Maps onto Controlled in `other`, but cases have different permutations
                        raise NotImplementedError
                        perm0 = case_zero._find_partial_permutation(s.case_zero)
            case _:
                raise NotImplementedError

    def _split(
        self, dimension: int, simplify=True
    ) -> tuple[QubitMap, QubitMap] | SimplifiedSplit:
        assert dimension >= 0 and dimension < self.dimension

        if len(self.registers) == 0:
            return (self, self)

        match self.registers[-1]:
            case Qubit(_):
                match QubitMap(self.registers[:-1])._split(dimension):
                    case (QubitMap(a), QubitMap(b)):
                        return QubitMap(a + [self.registers[-1]]), QubitMap(
                            b + [self.registers[-1]]
                        )
                    case SimplifiedSplit(common_head, case_zero, case_one, common_tail):
                        return SimplifiedSplit(
                            common_head,
                            case_zero,
                            case_one,
                            QubitMap(common_tail.registers + [self.registers[-1]]),
                        )
                    case _:
                        raise NotImplementedError
            case Controlled(case_zero, case_one):
                common_head = QubitMap(self.registers[:-1])
                if dimension == case_zero.dimension * common_head.dimension:
                    if simplify:
                        return SimplifiedSplit(
                            common_head, case_zero, case_one, QubitMap([])
                        )
                    else:
                        return QubitMap(
                            common_head.registers + case_zero.registers + [ZERO]
                        ), QubitMap(common_head.registers + case_one.registers + [ONE])
                elif dimension < case_zero.dimension * common_head.dimension:
                    (temp0, temp1) = case_zero._split(dimension, simplify=False)
                    return (
                        QubitMap(common_head.registers + temp0.registers + [ZERO]),
                        QubitMap(
                            common_head.registers
                            + Controlled(temp1, case_one).simplify()
                        ),
                    )
                else:
                    (temp0, temp1) = case_one._split(dimension, simplify=False)
                    return (
                        QubitMap(
                            common_head.registers
                            + Controlled(case_zero, temp0).simplify()
                        ),
                        QubitMap(common_head.registers + temp1.registers + [ONE]),
                    )
            case _:
                raise NotImplementedError

    def is_trivial(self) -> bool:
        for register in self.registers:
            match register:
                case Qubit(QubitType.ZERO) | Qubit(QubitType.ONE):
                    pass
                case Controlled() | Projection():
                    return False
        return True

    def test_basis(self, bits: int) -> bool:
        for register in self.registers:
            match register:
                case Qubit(QubitType.ZERO):
                    if bits & 1 != 0:
                        return False
                    bits = bits >> 1
                case Qubit(QubitType.ONE):
                    if bits & 1 != 1:
                        return False
                    bits = bits >> 1
                case Controlled(case_zero, case_one):
                    num_qubits = case_zero.total_qubits
                    relevant_bits = bits & ((1 << num_qubits) - 1)
                    control_bit = (bits >> num_qubits) & 1
                    result = None
                    if control_bit == 0:
                        result = case_zero.test_basis(relevant_bits)
                    else:
                        result = case_one.test_basis(relevant_bits)
                    if not result:
                        return False
                    bits = bits >> (num_qubits + 1)
                case _:
                    raise NotImplementedError
        return True

    def enumerate_basis(self) -> np.ndarray:
        return np.fromiter(
            filter(self.test_basis, range(2**self.total_qubits)), dtype=np.int32
        )

    def project(self, vector: np.ndarray) -> np.ndarray:
        return vector[self.enumerate_basis()]


class QubitType(Enum):
    ZERO = 0
    ONE = 1
    # TODO: HELPER_CLEAN, HELPER_DIRTY


@dataclass(frozen=True)
class Qubit:
    qubit_type: QubitType


ZERO = Qubit(QubitType.ZERO)
ONE = Qubit(QubitType.ONE)


@dataclass(frozen=True)
class Controlled:
    case_zero: QubitMap
    case_one: QubitMap

    def simplify(self) -> list[Register]:
        min_len = min(len(self.case_zero.registers), len(self.case_one.registers))
        for i in range(min_len):
            if self.case_zero.registers[i] != self.case_one.registers[i]:
                return self.case_zero.registers[:i] + [
                    Controlled(
                        QubitMap(self.case_zero.registers[i:]),
                        QubitMap(self.case_one.registers[i:]),
                    )
                ]
        return self.case_zero.registers[:min_len] + [
            Controlled(
                QubitMap(self.case_zero.registers[min_len:]),
                QubitMap(self.case_one.registers[min_len:]),
            )
        ]


ID = Controlled(QubitMap([]), QubitMap([]))


@dataclass(frozen=True)
class Projection:
    circuit: Circuit


Register = Qubit | Controlled | Projection

Permutation = Circuit | list[int]


@dataclass(frozen=True)
class CastResult:
    new_other: QubitMap
    permutation: Permutation


@dataclass(frozen=True)
class SimplifiedSplit:
    common_head: QubitMap
    case_zero: QubitMap
    case_one: QubitMap
    common_tail: QubitMap


class CastException(Exception):
    pass
