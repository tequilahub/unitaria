from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .circuit import Circuit


@dataclass(frozen=True)
class QubitMap:
    registers: list[Register]

    def simplify(self) -> QubitMap:
        simplified = []

        for register in self.registers:
            if type(register) is Controlled:
                case_zero = register.case_zero.simplify()
                case_one = register.case_one.simplify()
                if case_zero == case_one:
                    simplified.append(case_zero)
                    simplified.append(IdBit)
                else:
                    simplified.append(Controlled(case_zero, case_one))
            else:
                simplified.append(register)

        return QubitMap(simplified)

    def reduce(self) -> QubitMap:
        return QubitMap([
            register for register in self.registers if type(register) is IdBit
            or type(register) is Controlled or type(register) is Projection
        ])

    def is_all_zeros(self) -> bool:
        return all([register is ZeroBit for register in self.registers])

    def test_basis(self, bits: int):
        raise NotImplementedError

    def enumerate_basis(self) -> list[int]:
        raise NotImplementedError

    def project(self, vector: np.array) -> np.array:
        return vector[self.enumerate_basis()]


@dataclass(frozen=True)
class ZeroBit:
    pass


@dataclass(frozen=True)
class IdBit:
    pass


@dataclass(frozen=True)
class Controlled:
    case_zero: QubitMap
    case_one: QubitMap


@dataclass(frozen=True)
class Projection:
    circuit: Circuit


Register = ZeroBit | IdBit | Controlled | Projection
