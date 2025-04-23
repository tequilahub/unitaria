from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from bequem.circuit import Circuit


class QubitMap:

    zero_qubits: int
    registers: list[Register]

    def __init__(self, registers: list[Register] | int, zero_qubits: int = 0):
        if isinstance(registers, (int, np.integer)):
            registers = [ID] * registers
        for register in registers:
            if not isinstance(register, Register):
                raise ValueError(f"{register} is not valid in a QubitMap")
        simplified_registers = []
        for register in registers:
            if isinstance(register, Qubit):
                simplified_registers += register.simplify()
            else:
                simplified_registers.append(register)
        self.registers = simplified_registers
        self.zero_qubits = zero_qubits
        self._dimension = None
        self._total_qubits = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = 1
            for register in self.registers:
                self._dimension *= register.dimension()

        return self._dimension

    @property
    def total_qubits(self) -> int:
        if self._total_qubits is None:
            self._total_qubits = self.zero_qubits
            for register in self.registers:
                self._total_qubits += register.total_qubits()
        return self._total_qubits

    def __eq__(self, other) -> bool:
        return self.registers == other.registers and self.zero_qubits == other.zero_qubits

    def __repr__(self) -> str:
        registers = str(len(self.registers))
        for register in self.registers:
            if register != ID:
                registers = str(self.registers)
                break
        if self.zero_qubits == 0:
            return f"QubitMap({registers})"
        return f"QubitMap({registers}, zero_qubits={self.zero_qubits})"

    def is_trivial(self) -> bool:
        return len(self.registers) == 0

    def test_basis(self, bits: int) -> bool:
        if bits >= 2 ** (self.total_qubits - self.zero_qubits):
            return False
        for register in self.registers:
            match register:
                case Qubit(case_zero, case_one):
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

    def circuit(self) -> Circuit:
        # TODO
        return Circuit()


@dataclass(frozen=True, repr=False)
class Qubit:
    case_zero: QubitMap
    case_one: QubitMap

    def __repr__(self) -> str:
        if self == ID:
            return "ID"
        return f"Qubit(case_zero={self.case_zero}, case_one={self.case_one})"

    def total_qubits(self) -> int:
        return 1 + self.case_zero.total_qubits

    def dimension(self) -> int:
        return self.case_zero.dimension + self.case_one.dimension

    def simplify(self) -> list[Register]:
        min_len = min(len(self.case_zero.registers), len(self.case_one.registers))
        for i in range(min_len):
            if self.case_zero.registers[i] != self.case_one.registers[i]:
                return self.case_zero.registers[:i] + [
                    Qubit(
                        QubitMap(self.case_zero.registers[i:], self.case_zero.zero_qubits),
                        QubitMap(self.case_one.registers[i:], self.case_one.zero_qubits),
                    )
                ]
        return self.case_zero.registers[:min_len] + [
            Qubit(
                QubitMap(self.case_zero.registers[min_len:], self.case_zero.zero_qubits),
                QubitMap(self.case_one.registers[min_len:], self.case_one.zero_qubits),
            )
        ]


ID = Qubit(QubitMap([]), QubitMap([]))


@dataclass(frozen=True)
class Projection:
    circuit: Circuit

    def total_qubits(self) -> int:
        return len(self.circuit.tq_circuit.qubits) - 1


Register = Qubit | Projection
