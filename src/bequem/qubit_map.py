from __future__ import annotations
from typing import ClassVar
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .circuit import Circuit


@dataclass(frozen=True)
class QubitMap:
    registers: list[Register]

    def __init__(self, registers: list[Register]):
        for register in registers:
            if not isinstance(register, Register):
                raise ValueError(f"{register} is not valid in a QubitMap")
        object.__setattr__(self, "registers", registers)

    def simplify(self) -> QubitMap:
        simplified = []

        for register in self.registers:
            match register:
                case Controlled(case_zero, case_one):
                    case_zero = register.case_zero.simplify()
                    case_one = register.case_one.simplify()
                    if case_zero == case_one:
                        simplified.extend(case_zero.registers)
                        simplified.append(Qubit.ID)
                    else:
                        simplified.append(Controlled(case_zero, case_one))
                case Qubit() | Projection():
                    simplified.append(register)
                case _:
                    raise NotImplementedError

        return QubitMap(simplified)

    def reduce(self) -> QubitMap:
        result = []
        for register in self.registers:
            match register:
                case Qubit(QubitType.ID) | Controlled() | Projection():
                    result.append(register)
                case Qubit(QubitType.ZERO):
                    pass
                case _:
                    raise NotImplementedError
        return QubitMap(result)

    def is_all_zeros(self) -> bool:
        for register in self.registers:
            match register:
                case Qubit(QubitType.ZERO):
                    pass
                case Qubit(QubitType.ID) | Controlled() | Projection():
                    return False
        return True

    def test_basis(self, bits: int) -> bool:
        for register in self.registers:
            match register:
                case Qubit(QubitType.ZERO):
                    if bits & 1 != 0:
                        return False
                    bits = bits >> 1
                case Qubit(QubitType.ID):
                    bits = bits >> 1
                case _:
                    raise NotImplementedError
        return True

    def enumerate_basis(self) -> np.ndarray:
        return np.fromiter(
            filter(self.test_basis, range(2 ** self.total_bits())), dtype=np.int32
        )

    def project(self, vector: np.ndarray) -> np.ndarray:
        return vector[self.enumerate_basis()]

    def total_bits(self) -> int:
        sum = 0
        for register in self.registers:
            match register:
                case Controlled(case_one):
                    sum += 1 + case_one.total_bits()
                case Projection(circuit):
                    # One bit in the circuit corresponds to the result of the
                    # operation
                    sum += len(circuit.tq_circuit.qubits) - 1
                case Qubit():
                    sum += 1
                case _:
                    raise NotImplementedError
        return sum


class QubitType(Enum):
    ZERO = 0
    ID = 1


@dataclass(frozen=True)
class Qubit:
    qubit_type: QubitType

    ZERO: ClassVar[QubitType] = None
    ID: ClassVar[QubitType] = None


Qubit.ZERO = Qubit(QubitType.ZERO)
Qubit.ID = Qubit(QubitType.ID)


@dataclass(frozen=True)
class Controlled:
    case_zero: QubitMap
    case_one: QubitMap


@dataclass(frozen=True)
class Projection:
    circuit: Circuit


Register = Qubit | Controlled | Projection
