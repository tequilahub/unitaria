from __future__ import annotations
from typing import Iterable
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

    def test_basis(self, bits: int) -> bool:
        for register in self.registers:
            if type(register) is ZeroBit:
                if bits & 0 != 0:
                    return False
                bits = bits >> 1
            elif type(register) is IdBit:
                bits = bits >> 1
            else:
                NotImplementedError
        return True


    def enumerate_basis(self) -> np.ndarray:
        return np.fromiter(filter(self.test_basis, range(2**self.total_bits())), dtype=np.int32)
        
    def project(self, vector: np.ndarray) -> np.ndarray:
        return vector[self.enumerate_basis()]

    def total_bits(self) -> int:
        sum = 0
        for register in self.registers:
            if type(register) is Controlled:
                sum += 1 + register.case_one.total_bits()
            elif type(register) is Projection:
                sum += len(register.circuit.tq_circuit.qubits) - 1
            else:
                sum += 1
        return sum


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
