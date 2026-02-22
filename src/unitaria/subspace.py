"""
Objects for specifying a subspace of a quantum statespace.
"""

from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from unitaria.circuit import Circuit
import tequila as tq


class Subspace:
    # TODO: More documentation
    """
    Subspace of the statespace of a number of qubits.

    :param registers:
        List of registers, each describing a subspace. The total subspace of
        all registers is the tensor product of the individual subspaces.
    :param zero_qubits:
        Add an additional ``zero_qubits * [ZeroQubit()]`` to ``registers``.
    """

    registers: list[Register]

    def __init__(
        self, registers: list[Register] | int = None, *args, bits: int = None, dim: int = None, zero_qubits: int = 0
    ):
        if len(args) > 0:
            raise TypeError(
                "Subspace constructor only accepts keyword arguments: registers=... or zero_qubits=... or bits=...(not positional arguments)"
            )
        if bits is not None:
            self.registers = [ID] * bits
        elif dim is not None:
            min_bits = int(np.ceil(np.log2(dim)))
            subspace = Subspace.from_dim(dim, bits=min_bits)
            self.registers = subspace.registers
        elif registers is not None:
            # Convert integer to list of IDs before any use
            if isinstance(registers, (int, np.integer)):
                registers = [ID] * registers
            self.registers = list(registers)
        else:
            raise TypeError(
                "Must specify registers=... or bits=... or dim=... as a keyword argument in Subspace constructor"
            )

        for register in self.registers:
            if not isinstance(register, Register):
                raise ValueError(f"{register} is not a valid register in a Subspace")
        simplified_registers = []
        for register in self.registers:
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
        The dimension of the subspace
        """
        if self._dimension is None:
            self._dimension = 1
            for register in self.registers:
                self._dimension *= register.dimension()

        return self._dimension

    @property
    def total_qubits(self) -> int:
        """
        The number of qubits of the state space in which the subspace lives

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
            return f"Subspace({str_registers})"
        return f"Subspace({str_registers}, zero_qubits={trailing_zeros})"

    def is_trivial(self) -> bool:
        """
        Tests whether the subspace only contains the ``|0>`` state
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
        Tests whether the given basis state is inside the subspace
        """
        if bits >= 2**self.total_qubits:
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
        Enumerates the basis states inside the subspace
        """
        return np.fromiter(filter(self.test_basis, range(2**self.total_qubits)), dtype=np.int32)

    def project(self, vector: np.ndarray) -> np.ndarray:
        """
        Projects a state vector the embedding
        """
        return vector[self.enumerate_basis()]

    def circuit(self, target: Sequence[int], flag: int, ancillae: Sequence[int]) -> Circuit:
        """
        A circuit which checks whether a state is inside the subspace.

        The result of the check will be stored in a flag qubit where
        the index is the output of ``total_qubits``.
        Specifically, this qubit will be flipped if the other qubits
        represent a state outside the embedded vector space.
        """
        # Index of the current register
        index = 0
        # Next unused ancilla index
        ancilla_index = 0
        # Set of intermediate flags, if one of them is set, the result flag will be set
        intermediate_flags = []

        circuit = Circuit()

        for register in self.registers:
            if isinstance(register, ZeroQubit):
                # Zero qubits already behave like flag qubits, so we only need to toggle it
                circuit += tq.gates.X(target=target[index])
                intermediate_flags.append(target[index])
                index += 1
                continue

            if register == ID:
                # No need to do anything here
                index += 1
                continue

            if not isinstance(register, ControlledSubspace):
                raise ValueError("Registers that aren't of type Qubit are unsupported")

            circuit += register.circuit(
                target=target[index : index + register.total_qubits()],
                flag=ancillae[ancilla_index],
                ancillae=ancillae[ancilla_index + 1 :],
            )
            circuit += tq.gates.X(target=ancillae[ancilla_index])
            intermediate_flags.append(ancillae[ancilla_index])

            index += register.total_qubits()
            ancilla_index += 1

        # Apply the constructed circuit, add a multi-controlled NOT to determine
        # if the result flag is set, then uncompute the rest of the circuit
        circuit = (
            circuit + tq.gates.X(target=flag, control=intermediate_flags) + tq.gates.X(target=flag) + circuit.adjoint()
        )

        return circuit

    def clean_ancilla_count(self) -> int:
        controlled_subspaces = filter(lambda r: isinstance(r, ControlledSubspace), self.registers)
        return max(
            [i + r.clean_ancilla_count() for (i, r) in enumerate(controlled_subspaces, start=1)],
            default=0,
        )

    def verify_circuit(self):
        circuit = self.circuit(
            range(self.total_qubits),
            self.total_qubits,
            range(
                self.total_qubits + 1,
                self.total_qubits + 1 + self.clean_ancilla_count(),
            ),
        )
        for i in range(2**self.total_qubits):
            result = circuit.simulate(i)
            if self.test_basis(i):
                assert result[i] == 1
            else:
                assert result[i + 2**self.total_qubits] == 1

    def case_zero(self) -> Subspace:
        trailing_zeros = self.trailing_zeros()
        assert isinstance(self.registers[-(trailing_zeros + 1)], ControlledSubspace)

        return Subspace(
            self.registers[: -(trailing_zeros + 1)] + self.registers[-(trailing_zeros + 1)].case_zero.registers,
        )

    def case_one(self) -> Subspace:
        trailing_zeros = self.trailing_zeros()
        assert isinstance(self.registers[-(trailing_zeros + 1)], ControlledSubspace)

        return Subspace(
            self.registers[: -(trailing_zeros + 1)] + self.registers[-(trailing_zeros + 1)].case_one.registers,
        )

    @staticmethod
    def from_dim(dim: int, bits: int | None = None) -> Subspace:
        if bits is None:
            bits = int(np.ceil(np.log2(dim)))
        if dim == 1:
            return Subspace(registers=0, zero_qubits=bits)
        min_bits = int(np.ceil(np.log2(dim)))
        case_zero = Subspace(registers=min_bits - 1)
        case_one = Subspace.from_dim(dim - 2 ** (min_bits - 1), min_bits - 1)
        return Subspace(registers=[ControlledSubspace(case_zero, case_one)], zero_qubits=bits - min_bits)


class Register(ABC):
    @abstractmethod
    def total_qubits(self) -> int:
        """
        The number of qubits of the state space in which the subspace lives

        The dimension of the state space is ``2 ** total_qubits``
        """
        raise NotImplementedError

    @abstractmethod
    def dimension(self) -> int:
        """
        The dimension of the subspace
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
    Register where the most significant qubit determines the subspace
    corresponding to lower qubits

    :param case_zero:
        Embedding of lower qubits, if highest qubit is ``|0>``
    :param case_one:
        Embedding of lower qubits, if highest qubit is ``|1>``
    """

    case_zero: Subspace
    case_one: Subspace

    def __post_init__(self):
        assert self.case_zero.total_qubits == self.case_one.total_qubits

    def __repr__(self) -> str:
        if self == ID:
            return "ID"
        return f"ControlledSubspace(case_zero={self.case_zero}, case_one={self.case_one})"

    def total_qubits(self) -> int:
        return 1 + self.case_zero.total_qubits

    def dimension(self) -> int:
        return self.case_zero.dimension + self.case_one.dimension

    def simplify(self) -> list[Register]:
        """
        Returns a potentially simpler representation of this register

        Specifically, if `case_one` and `case_zero` agree in a number of lowest
        qubits, this common part can be factored out. E.g.

            >>> from unitaria.subspace import Subspace, ControlledSubspace
            >>> ControlledSubspace(Subspace(2), Subspace(1, 1)).simplify()
            [ID, ControlledSubspace(case_zero=Subspace(1), case_one=Subspace(0, zero_qubits=1))]
        """
        min_len = min(len(self.case_zero.registers), len(self.case_one.registers))
        for i in range(min_len):
            if self.case_zero.registers[i] != self.case_one.registers[i]:
                return self.case_zero.registers[:i] + [
                    ControlledSubspace(
                        Subspace(registers=self.case_zero.registers[i:]),
                        Subspace(registers=self.case_one.registers[i:]),
                    )
                ]
        return self.case_zero.registers[:min_len] + [
            ControlledSubspace(
                Subspace(registers=self.case_zero.registers[min_len:]),
                Subspace(registers=self.case_one.registers[min_len:]),
            )
        ]

    def circuit(self, target: Sequence[int], flag: int, ancillae: Sequence[int]) -> Circuit:
        """
        A circuit which checks whether a state is inside the subspace.

        The result of the check will be stored in the flag qubit.
        Specifically, this qubit will be flipped if the other qubits
        represent a state outside the embedded vector space.
        """
        circuit = Circuit()
        control = target[-1]
        circuit_zero = self.case_zero.circuit(target=target[:-1], flag=flag, ancillae=ancillae)
        circuit_one = self.case_one.circuit(target=target[:-1], flag=flag, ancillae=ancillae)
        circuit += tq.gates.X(target=control)
        circuit += circuit_zero.add_controls(control)
        circuit += tq.gates.X(target=control)
        circuit += circuit_one.add_controls(control)
        return circuit

    def clean_ancilla_count(self) -> int:
        return max(self.case_zero.clean_ancilla_count(), self.case_one.clean_ancilla_count())


ID = ControlledSubspace(Subspace(registers=[]), Subspace(registers=[]))
