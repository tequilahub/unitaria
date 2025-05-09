from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

from bequem.circuit import Circuit


class Subspace:
    # TODO: More documentation
    """
    Embedding of a vectorspace into the statespace of a number of qubits.

    :ivar registers:
        List of registers, each describing an embedding. The total embedding of
        all registers is the tensor product of the individual embeddings.
    :ivar zero_qubits:
        The number of additional qubits (always the most significant) which are
        always zero in the embedding
    """

    registers: list[Register]

    def __init__(self, registers: list[Register] | int, zero_qubits: int = 0):
        if isinstance(registers, (int, np.integer)):
            registers = [ID] * registers
        for register in registers:
            if not isinstance(register, Register):
                raise ValueError(f"{register} is not valid in a QubitMap")
        simplified_registers = []
        for register in registers:
            if isinstance(register, ControlledSubspace):
                simplified_registers += register.simplify()
            else:
                simplified_registers.append(register)
        self.registers = simplified_registers + [ZeroQubit()] * zero_qubits
        self._dimension = None
        self._total_qubits = None

    @property
    def dimension(self) -> int:
        """
        The dimension of the embedded vector space
        """
        if self._dimension is None:
            self._dimension = 1
            for register in self.registers:
                self._dimension *= register.dimension()

        return self._dimension

    @property
    def total_qubits(self) -> int:
        """
        The number of qubits of the state space in which the embedding lives

        The dimension of the state space is ``2 ** total_qubits``
        """
        if self._total_qubits is None:
            self._total_qubits = 0
            for register in self.registers:
                self._total_qubits += register.total_qubits()
        return self._total_qubits

    def __eq__(self, other) -> bool:
        return self.registers == other.registers

    def match_nonzero(self, other: Subspace) -> bool:
        return self.nonzero_registers() == other.nonzero_registers()

    def __repr__(self) -> str:
        trailing_zeros = self.trailing_zeros()
        registers = self.nonzero_registers()
        str_registers = str(len(registers))
        for register in registers:
            if register != ID:
                str_registers = str(registers)
                break
        if trailing_zeros == 0:
            return f"QubitMap({str_registers})"
        return f"QubitMap({str_registers}, zero_qubits={trailing_zeros})"

    def is_trivial(self) -> bool:
        """
        Tests whether the embedded vector space only contains the ``|0>`` state
        """
        return self.trailing_zeros() == len(self.registers)

    def trailing_zeros(self) -> int:
        for i in reversed(range(len(self.registers))):
            if not isinstance(self.registers[i], ZeroQubit):
                return len(self.registers) - i - 1
        return len(self.registers)

    def nonzero_registers(self) -> list[Register]:
        trailing_zeros = self.trailing_zeros()
        if trailing_zeros == 0:
            return self.registers
        else:
            return self.registers[:-trailing_zeros]

    def test_basis(self, bits: int) -> bool:
        """
        Tests whether the given basis state is inside the embedding
        """
        if bits >= 2 ** self.total_qubits:
            raise ValueError
        for register in self.registers:
            match register:
                case ControlledSubspace(case_zero, case_one):
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
                case ZeroQubit():
                    if bits & 1 != 0:
                        return False
                    bits = bits >> 1
                case _:
                    raise NotImplementedError
        return True

    def enumerate_basis(self) -> np.ndarray:
        """
        Enumerates the basis states inside the embedding
        """
        return np.fromiter(
            filter(self.test_basis, range(2**self.total_qubits)), dtype=np.int32
        )

    def project(self, vector: np.ndarray) -> np.ndarray:
        """
        Projects a state vector the embedding 
        """
        return vector[self.enumerate_basis()]

    def circuit(self) -> Circuit:
        """
        A circuit which checks, whether a number of qubits represent a state
        inside the embedding.

        The result of the check will be stored in an additional qubit, meangin
        the total number of qubits in the circuit is ``total_qubits + 1``.
        Specifically, the extra qubit will be flipped, if the other qubits
        represent a state outside the embedded vector space.
        """
        # TODO: Oliver
        return Circuit()

    def case_zero(self) -> Subspace:
        trailing_zeros = self.trailing_zeros()
        assert isinstance(self.registers[-(trailing_zeros + 1)], ControlledSubspace)

        return Subspace(
            self.registers[:-(trailing_zeros + 1)] + self.registers[-(trailing_zeros + 1)].case_zero.registers,
        )

    def case_one(self) -> Subspace:
        trailing_zeros = self.trailing_zeros()
        assert isinstance(self.registers[-(trailing_zeros + 1)], ControlledSubspace)

        return Subspace(
            self.registers[:-(trailing_zeros + 1)] + self.registers[-(trailing_zeros + 1)].case_one.registers,
        )


class Register(ABC):

    @abstractmethod
    def total_qubits(self) -> int:
        """
        The number of qubits of the state space in which the embedding lives

        The dimension of the state space is ``2 ** total_qubits``
        """
        raise NotImplementedError

    @abstractmethod
    def dimension(self) -> int:
        """
        The dimension of the embedded vector space
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ZeroQubit(Register):

    def total_qubits(self) -> int:
        return 1

    def dimension(self) -> int:
        return 1


@dataclass(frozen=True, repr=False)
class ControlledSubspace(Register):
    """
    Register where the most significant qubit determines the embedding of the
    lower qubits

    :ivar case_zero:
        Embedding of lower qubits, if highest qubit is ``|0>``
    :ivar case_one:
        Embedding of lower qubits, if highest qubit is ``|1>``
    """

    case_zero: Subspace
    case_one: Subspace

    def __repr__(self) -> str:
        if self == ID:
            return "ID"
        return f"Qubit(case_zero={self.case_zero}, case_one={self.case_one})"

    def total_qubits(self) -> int:
        return 1 + self.case_zero.total_qubits

    def dimension(self) -> int:
        return self.case_zero.dimension + self.case_one.dimension

    def simplify(self) -> list[Register]:
        """
        Returns a potentially simpler representation of this register

        Specifically, if :py:attr:`case_one` and :py:attr:`case_zero` agree in a number of lowest
        qubits, this common part can be factored out. E.g.

            >>> from bequem.qubit_map import Qubit, QubitMap
            >>> Qubit(QubitMap(2), QubitMap(1)).simplify()
            [ID, Qubit(case_zero=QubitMap(1), case_one=QubitMap(0))]
        """
        min_len = min(len(self.case_zero.registers), len(self.case_one.registers))
        for i in range(min_len):
            if self.case_zero.registers[i] != self.case_one.registers[i]:
                return self.case_zero.registers[:i] + [
                    ControlledSubspace(
                        Subspace(self.case_zero.registers[i:]),
                        Subspace(self.case_one.registers[i:]),
                    )
                ]
        return self.case_zero.registers[:min_len] + [
            ControlledSubspace(
                Subspace(self.case_zero.registers[min_len:]),
                Subspace(self.case_one.registers[min_len:]),
            )
        ]


ID = ControlledSubspace(Subspace([]), Subspace([]))
